import logging
import os
from pathlib import Path

import numpy as np
import pycolmap
import torch
from tqdm import tqdm

from gluemap.estimators.track_establishment import TrackEstablishment

logger = logging.getLogger(__name__)


def camera_from_intrinsics_matrix(
    intrinsics_matrix: np.ndarray | torch.Tensor,
    camera_model: str = "SIMPLE_PINHOLE",
    width: int | None = None,
    height: int | None = None,
    camera_id: int = 0,
) -> pycolmap.Camera:
    """
    Create a pycolmap.Camera from a 3x3 intrinsics matrix.

    Args:
        intrinsics_matrix: 3x3 numpy array or torch tensor
        camera_model: Camera model string
        width: Image width (defaults to 2 * cx)
        height: Image height (defaults to 2 * cy)
        camera_id: Camera ID

    Returns:
        pycolmap.Camera
    """
    if torch.is_tensor(intrinsics_matrix):
        intrinsics_matrix = intrinsics_matrix.cpu().numpy()

    fx, fy = float(intrinsics_matrix[0, 0]), float(intrinsics_matrix[1, 1])
    cx, cy = float(intrinsics_matrix[0, 2]), float(intrinsics_matrix[1, 2])

    if width is None:
        width = int(2 * cx)
    if height is None:
        height = int(2 * cy)

    camera = pycolmap.Camera.create_from_model_id(
        camera_id, pycolmap.CameraModelId(camera_model), 1.0, width, height
    )
    if len(camera.focal_length_idxs()) == 2:
        camera.focal_length_x = fx
        camera.focal_length_y = fy
    else:
        camera.focal_length = (fx + fy) / 2
    camera.principal_point_x = cx
    camera.principal_point_y = cy
    return camera


def extract_gt_intrinsics(
    gt_path: str,
    images_list: list[str],
    intrinsics_mapping: dict[int, int],
    match_by_basename: bool = False,
) -> list[torch.Tensor | None]:
    """Extract GT intrinsics from a COLMAP reconstruction directory.

    Returns List[Optional[torch.Tensor]] indexed by camera_id,
    each tensor shape (1, 3, 3), matching global_intrinsics format.
    """
    gt_recon = pycolmap.Reconstruction()
    gt_recon.read(gt_path)

    name_to_gt_cam = {}
    for _img_id, img in gt_recon.images.items():
        key = os.path.basename(img.name) if match_by_basename else img.name
        name_to_gt_cam[key] = gt_recon.cameras[img.camera_id]

    num_cameras = max(intrinsics_mapping) + 1
    gt_intrinsics = [None] * num_cameras

    for i, name in enumerate(images_list):
        cam_id = intrinsics_mapping[i]
        key = os.path.basename(name) if match_by_basename else name
        if key in name_to_gt_cam and gt_intrinsics[cam_id] is None:
            gt_cam = name_to_gt_cam[key]
            K = gt_cam.calibration_matrix()
            gt_intrinsics[cam_id] = torch.tensor(
                K, dtype=torch.float32
            ).unsqueeze(0)

    num_set = sum(1 for x in gt_intrinsics if x is not None)
    logger.info(
        f"GT intrinsics loaded: {num_set}/{num_cameras} cameras matched"
    )
    return gt_intrinsics


def merge_colmap_databases(
    db_path_primary: str,
    db_path_secondary: str,
    output_path: str = None,
    primary_features_first: bool = True,
) -> str:
    """
    Merge two COLMAP databases by creating a fresh database from scratch.

    Images are matched by name. Features and matches from both databases
    are combined. Cameras from primary are used for common images.

    Args:
        db_path_primary: Path to the primary database (cameras preserved
            for existing images)
        db_path_secondary: Path to the secondary database (features/matches
            merged)
        output_path: Path for output database. If None, modifies primary
            in-place.
        primary_features_first: If True, primary keypoints come first in
            the merged array (default). If False, secondary keypoints come
            first.

    Returns:
        Path to the merged database
    """
    target_path = output_path if output_path is not None else db_path_primary

    # Remove existing output database if it exists
    if os.path.exists(target_path):
        os.remove(target_path)

    # Open source databases (read-only)
    db_primary = pycolmap.Database.open(db_path_primary)

    db_secondary = pycolmap.Database.open(db_path_secondary)

    # Create fresh output database
    db_output = pycolmap.Database.open(target_path)

    # Read all data from primary
    primary_cameras = {
        cam.camera_id: cam for cam in db_primary.read_all_cameras()
    }
    primary_images = {img.name: img for img in db_primary.read_all_images()}

    # Read all data from secondary
    secondary_cameras = {
        cam.camera_id: cam for cam in db_secondary.read_all_cameras()
    }
    secondary_images = {img.name: img for img in db_secondary.read_all_images()}

    # Determine common and unique images
    all_names = set(primary_images.keys()) | set(secondary_images.keys())

    # Step 1: Write cameras from primary (use primary cameras for common images)
    logger.info("Writing cameras...")
    for cam_id, cam in primary_cameras.items():
        new_cam = pycolmap.Camera(
            camera_id=cam_id,
            model=cam.model_name,
            params=cam.params,
            width=cam.width,
            height=cam.height,
        )
        db_output.write_camera(new_cam)

    # Add any additional cameras from secondary (for images only in secondary)
    max_camera_id = max(primary_cameras.keys()) if primary_cameras else 0
    secondary_cam_to_output_cam = {}
    for name in secondary_images:
        if name not in primary_images:
            sec_img = secondary_images[name]
            if sec_img.camera_id not in secondary_cam_to_output_cam:
                max_camera_id += 1
                secondary_cam_to_output_cam[sec_img.camera_id] = max_camera_id
                sec_cam = secondary_cameras[sec_img.camera_id]
                new_cam = pycolmap.Camera(
                    camera_id=max_camera_id,
                    model=sec_cam.model_name,
                    params=sec_cam.params,
                    width=sec_cam.width,
                    height=sec_cam.height,
                )
                db_output.write_camera(new_cam)

    # Step 2: Write images and track keypoint offsets
    name_to_output_image_id = {}
    primary_offsets = {}  # image_name -> index offset for primary keypoints
    secondary_offsets = {}  # image_name -> index offset for secondary keypoints

    logger.info("Writing images and merging keypoints...")
    for name in all_names:
        if name in primary_images:
            pri_img = primary_images[name]
            new_img = pycolmap.Image()
            new_img.name = name
            new_img.image_id = pri_img.image_id
            new_img.camera_id = pri_img.camera_id
            db_output.write_image(new_img, use_image_id=True)
            name_to_output_image_id[name] = pri_img.image_id
        else:
            sec_img = secondary_images[name]
            new_img = pycolmap.Image()
            new_img.name = name
            new_img.image_id = sec_img.image_id
            new_img.camera_id = secondary_cam_to_output_cam[sec_img.camera_id]
            db_output.write_image(new_img, use_image_id=True)
            name_to_output_image_id[name] = sec_img.image_id

    # Step 3: Merge and write keypoints
    logger.info("Merging and writing keypoints...")
    for name in all_names:
        output_image_id = name_to_output_image_id[name]

        pri_kp = None
        sec_kp = None

        if name in primary_images:
            pri_kp = db_primary.read_keypoints(primary_images[name].image_id)
        if name in secondary_images:
            sec_kp = db_secondary.read_keypoints(
                secondary_images[name].image_id
            )

        # Compute per-source offsets and merge keypoints
        pri_kp_xy = (
            pri_kp[:, :2] if pri_kp is not None and len(pri_kp) > 0 else None
        )
        sec_kp_xy = (
            sec_kp[:, :2] if sec_kp is not None and len(sec_kp) > 0 else None
        )

        if pri_kp_xy is not None and sec_kp_xy is not None:
            if primary_features_first:
                merged_kp = np.vstack([pri_kp_xy, sec_kp_xy])
                primary_offsets[name] = 0
                secondary_offsets[name] = len(pri_kp_xy)
            else:
                merged_kp = np.vstack([sec_kp_xy, pri_kp_xy])
                secondary_offsets[name] = 0
                primary_offsets[name] = len(sec_kp_xy)
        elif pri_kp_xy is not None:
            merged_kp = pri_kp_xy
            primary_offsets[name] = 0
            secondary_offsets[name] = 0
        elif sec_kp_xy is not None:
            merged_kp = sec_kp_xy
            primary_offsets[name] = 0
            secondary_offsets[name] = 0
        else:
            merged_kp = None
            primary_offsets[name] = 0
            secondary_offsets[name] = 0

        if merged_kp is not None and len(merged_kp) > 0:
            db_output.write_keypoints(output_image_id, merged_kp)

    # Step 4: Collect all matches from both databases
    all_matches = {}  # (out_id1, out_id2) -> merged matches array

    # Read primary matches
    primary_id_to_name = {
        img.image_id: name for name, img in primary_images.items()
    }
    pair_ids_pri, matches_list_pri = db_primary.read_all_matches()
    for pair_id, matches in zip(pair_ids_pri, matches_list_pri, strict=False):
        id1, id2 = pycolmap.pair_id_to_image_pair(pair_id)
        if matches is None or len(matches) == 0:
            continue
        name1 = primary_id_to_name.get(id1)
        name2 = primary_id_to_name.get(id2)
        if name1 is None or name2 is None:
            continue
        out_id1 = name_to_output_image_id[name1]
        out_id2 = name_to_output_image_id[name2]
        offset1 = primary_offsets.get(name1, 0)
        offset2 = primary_offsets.get(name2, 0)
        key = (min(out_id1, out_id2), max(out_id1, out_id2))
        remapped = matches.copy()
        if offset1 > 0 or offset2 > 0:
            if out_id1 <= out_id2:
                remapped[:, 0] += offset1
                remapped[:, 1] += offset2
            else:
                remapped[:, 0] += offset2
                remapped[:, 1] += offset1
        if key not in all_matches:
            all_matches[key] = []
        all_matches[key].append(remapped)

    logger.info("Merging matches from secondary database...")

    # Read secondary matches (with offset)
    secondary_id_to_name = {
        img.image_id: name for name, img in secondary_images.items()
    }
    pair_ids_sec, matches_list_sec = db_secondary.read_all_matches()
    for pair_id, matches in zip(pair_ids_sec, matches_list_sec, strict=False):
        id1, id2 = pycolmap.pair_id_to_image_pair(pair_id)
        if matches is None or len(matches) == 0:
            continue
        name1 = secondary_id_to_name.get(id1)
        name2 = secondary_id_to_name.get(id2)
        if name1 is None or name2 is None:
            continue
        out_id1 = name_to_output_image_id[name1]
        out_id2 = name_to_output_image_id[name2]
        offset1 = secondary_offsets.get(name1, 0)
        offset2 = secondary_offsets.get(name2, 0)
        key = (min(out_id1, out_id2), max(out_id1, out_id2))
        remapped = matches.copy()
        if offset1 > 0 or offset2 > 0:
            if out_id1 <= out_id2:
                remapped[:, 0] += offset1
                remapped[:, 1] += offset2
            else:
                remapped[:, 0] += offset2
                remapped[:, 1] += offset1
        if key not in all_matches:
            all_matches[key] = []
        all_matches[key].append(remapped)

    # Write merged matches
    for (id1, id2), match_list in all_matches.items():
        if len(match_list) > 0:
            merged = np.vstack(match_list)
            db_output.write_matches(id1, id2, merged)

    # Step 5: Collect all two-view geometries
    all_geometries = {}  # (out_id1, out_id2) -> (inliers_list, config)

    # Read primary geometries
    for pair_id in pair_ids_pri:
        id1, id2 = pycolmap.pair_id_to_image_pair(pair_id)
        name1 = primary_id_to_name.get(id1)
        name2 = primary_id_to_name.get(id2)
        if name1 is None or name2 is None:
            continue
        try:
            geom = db_primary.read_two_view_geometry(id1, id2)
            if (
                geom is not None
                and geom.inlier_matches is not None
                and len(geom.inlier_matches) > 0
            ):
                out_id1 = name_to_output_image_id[name1]
                out_id2 = name_to_output_image_id[name2]
                offset1 = primary_offsets.get(name1, 0)
                offset2 = primary_offsets.get(name2, 0)
                key = (min(out_id1, out_id2), max(out_id1, out_id2))
                remapped = geom.inlier_matches.copy()
                if offset1 > 0 or offset2 > 0:
                    if out_id1 <= out_id2:
                        remapped[:, 0] += offset1
                        remapped[:, 1] += offset2
                    else:
                        remapped[:, 0] += offset2
                        remapped[:, 1] += offset1
                if key not in all_geometries:
                    all_geometries[key] = ([], geom.config)
                all_geometries[key][0].append(remapped)
        except Exception:
            pass

    # Read secondary geometries (with offset)
    for pair_id in pair_ids_sec:
        id1, id2 = pycolmap.pair_id_to_image_pair(pair_id)
        name1 = secondary_id_to_name.get(id1)
        name2 = secondary_id_to_name.get(id2)
        if name1 is None or name2 is None:
            continue
        try:
            geom = db_secondary.read_two_view_geometry(id1, id2)
            if (
                geom is not None
                and geom.inlier_matches is not None
                and len(geom.inlier_matches) > 0
            ):
                out_id1 = name_to_output_image_id[name1]
                out_id2 = name_to_output_image_id[name2]
                offset1 = secondary_offsets.get(name1, 0)
                offset2 = secondary_offsets.get(name2, 0)
                key = (min(out_id1, out_id2), max(out_id1, out_id2))
                remapped = geom.inlier_matches.copy()
                if offset1 > 0 or offset2 > 0:
                    if out_id1 <= out_id2:
                        remapped[:, 0] += offset1
                        remapped[:, 1] += offset2
                    else:
                        remapped[:, 0] += offset2
                        remapped[:, 1] += offset1
                if key not in all_geometries:
                    all_geometries[key] = ([], geom.config)
                all_geometries[key][0].append(remapped)
        except Exception:
            pass

    # Write merged two-view geometries
    for (id1, id2), (inliers_list, config) in all_geometries.items():
        if len(inliers_list) > 0:
            merged_inliers = np.vstack(inliers_list)
            new_geom = pycolmap.TwoViewGeometry()
            new_geom.inlier_matches = merged_inliers
            new_geom.config = config
            db_output.write_two_view_geometry(id1, id2, new_geom)

    # Close all databases
    db_primary.close()
    db_secondary.close()
    db_output.close()

    logger.info(f"Created merged database with {len(all_names)} images")
    return target_path


def remap_cameras_to_intrinsics(
    database: pycolmap.Database,
    images_list: list[str],
    intrinsics_mapping: dict[int, int],
) -> dict[int, int]:
    """
    Remap camera IDs to be consistent with intrinsics_mapping values.

    After pycolmap.import_images/extract_features creates cameras with
    arbitrary IDs, this function:
    1. Reads existing cameras and their parameters
    2. Updates/writes cameras with IDs = intrinsics_mapping_value + 1
    3. Updates all images to reference the new camera IDs

    Args:
        database: pycolmap.Database instance
        images_list: List of image names
        intrinsics_mapping: Dict mapping image index to intrinsics group index

    Returns:
        idx_to_image_id: Dict mapping image index to database image_id
    """
    # Read cameras and build mapping from old camera_id to camera params
    cameras = database.read_all_cameras()
    old_camera_params = {cam.camera_id: cam for cam in cameras}
    existing_camera_ids = set(old_camera_params.keys())

    # Track which old camera to use for each intrinsics group (first occurrence)
    intrinsics_to_old_camera = {}
    idx_to_image_id = {}
    for i in range(len(images_list)):
        image = database.read_image_with_name(images_list[i])
        idx_to_image_id[i] = image.image_id
        intrinsics_val = intrinsics_mapping[i]
        if intrinsics_val not in intrinsics_to_old_camera:
            intrinsics_to_old_camera[intrinsics_val] = image.camera_id

    # Create/update cameras with correct IDs (intrinsics_val + 1)
    new_cameras = {}
    for intrinsics_val, old_cam_id in intrinsics_to_old_camera.items():
        new_cam_id = intrinsics_val + 1  # 1-indexed for COLMAP
        old_cam = old_camera_params[old_cam_id]

        new_cam = pycolmap.Camera(
            camera_id=new_cam_id,
            model=old_cam.model_name,
            params=old_cam.params,
            width=old_cam.width,
            height=old_cam.height,
        )

        if new_cam_id in existing_camera_ids:
            # Update existing camera with new params
            database.update_camera(new_cam)
        else:
            # Write new camera
            database.write_camera(new_cam)
        new_cameras[new_cam_id] = new_cam

    # Create rigs for new camera IDs (rig_id == camera_id in trivial setup)
    existing_rigs = {rig.rig_id for rig in database.read_all_rigs()}
    for new_cam_id, cam in new_cameras.items():
        if new_cam_id not in existing_rigs:
            rig = pycolmap.Rig()
            rig.rig_id = new_cam_id
            rig.add_ref_sensor(cam.sensor_id)
            database.write_rig(rig, use_rig_id=True)

    # Update images and frames to reference new camera/rig IDs
    for i in range(len(images_list)):
        image = database.read_image_with_name(images_list[i])
        image.camera_id = intrinsics_mapping[i] + 1
        frame = database.read_frame(image.frame_id)
        frame.rig_id = intrinsics_mapping[i] + 1
        frame.clear_data_ids()
        frame.add_data_id(image.data_id)
        database.update_frame(frame)
        database.update_image(image)

    return idx_to_image_id


def prepare_glomap_prior(
    dir_write: str,
    images_shape_ori: list[tuple[int, int]],
    images_list: list[str] | None,
    global_intrinsics: list[torch.Tensor | None],
    predictions_dict: dict,
    intrinsics_mapping: dict[int, int],
    camera_model: str = "SIMPLE_RADIAL",
    add_tracks: bool = True,
    add_virtual_points: bool = True,
    database_name: str = "database.db",
) -> None:
    """Build a COLMAP database to use as a glomap prior.

    Writes cameras (one per entry in ``global_intrinsics``), images, and
    keypoints/two-view-matches derived from ``predictions_dict`` via
    :class:`TrackEstablishment` into a fresh database at
    ``dir_write/database_name``. Any existing database at that path is
    removed first.

    Args:
        dir_write: Output directory (created if missing).
        images_shape_ori: Per-image ``(height, width)`` of the originals.
        images_list: Image names written to the database; if ``None``,
            images are named by their integer index.
        global_intrinsics: Per-camera intrinsics tensors (shape ``(1, 3, 3)``);
            ``None`` slots are skipped.
        predictions_dict: Track dictionary consumed by
            :meth:`TrackEstablishment.establish_keypoints_and_correspondences`.
        intrinsics_mapping: Maps each image index to a camera id.
        camera_model: COLMAP camera model name.
        add_tracks: Forwarded to ``TrackEstablishment``.
        add_virtual_points: Forwarded to ``TrackEstablishment``.
        database_name: Filename of the database under ``dir_write``.
    """
    if not os.path.exists(dir_write):
        os.makedirs(dir_write)

    database_path = dir_write + "/" + database_name
    if os.path.exists(database_path):
        os.remove(database_path)
    database = pycolmap.Database.open(database_path)

    camera_sizes = [None for _ in range(len(global_intrinsics))]
    for i in range(len(intrinsics_mapping)):
        camera_id = intrinsics_mapping[i]
        if camera_sizes[camera_id] is None:
            camera_sizes[camera_id] = images_shape_ori[i]

    # Write cameras
    cameras_colmap = {}
    for i in range(len(global_intrinsics)):
        if global_intrinsics[i] is None:
            continue

        camera = camera_from_intrinsics_matrix(
            global_intrinsics[i][0],
            camera_model,
            camera_sizes[i][-1],
            camera_sizes[i][0],
            i + 1,
        )

        database.write_camera(camera)
        cameras_colmap[i] = camera

    logger.info("Write cameras to database done")

    N = len(images_shape_ori)

    # Establish keypoints and correspondences
    (
        keypoints_per_image,
        correspondences,
        _,
        _,
        _,
    ) = TrackEstablishment().establish_keypoints_and_correspondences(
        predictions_dict,
        N,
        add_tracks,
        add_virtual_points,
    )

    # Add images
    logger.info("Add images to database...")
    for idx in tqdm(range(N)):
        image = pycolmap.Image()
        image.camera_id = intrinsics_mapping[idx] + 1

        if images_list is not None:
            image.name = images_list[idx]
        else:
            image.name = str(idx)
        image.image_id = idx + 1

        database.write_image(image, use_image_id=True)

    # Write keypoints to database
    for idx, keypoints in keypoints_per_image.items():
        database.write_keypoints(idx + 1, keypoints)

    # For each correspondence, collect into a numpy array and write to the
    # database
    logger.info("Write matches to database...")
    for key in tqdm(correspondences.keys()):
        image_id1 = key[0]
        image_id2 = key[1]

        if len(correspondences[key]) < 3:
            continue

        two_view_geometry = pycolmap.TwoViewGeometry()
        two_view_geometry.inlier_matches = np.array(correspondences[key])
        two_view_geometry.config = 2

        database.write_two_view_geometry(
            image_id1, image_id2, two_view_geometry
        )
        database.write_matches(
            image_id1, image_id2, np.array(correspondences[key])
        )

    database.close()


def prepare_sift_database(
    dir_write: str,
    images_path: str,
    images_list: list[str],
    intrinsics_mapping: dict[int, int],
    pairs: np.ndarray,
    device: str = "cuda",
    camera_model: str = "SIMPLE_RADIAL",
    skip_matching: bool = False,
    remove_existing: bool = True,
) -> bool:
    """Extract SIFT features and match the given pairs into a COLMAP database.

    Runs ``pycolmap.extract_features`` on ``images_list`` (under
    ``images_path``) into ``dir_write/database_sift.db``, then remaps
    camera IDs to follow ``intrinsics_mapping`` and (unless
    ``skip_matching``) runs SIFT matching on ``pairs``.

    Args:
        dir_write: Output directory holding the database and pairs file.
        images_path: Root directory of the input images.
        images_list: Image names relative to ``images_path``.
        intrinsics_mapping: Maps each image index to a camera id.
        pairs: ``(M, 2)`` int array of image-index pairs to match.
        device: Where to run SIFT extraction and matching. ``"cuda"`` uses
            all visible CUDA devices, ``"cuda:N"`` selects a single device,
            and ``"cpu"`` disables GPU.
        camera_model: COLMAP camera model used during feature extraction.
        skip_matching: If set, return immediately after feature extraction.
        remove_existing: If set, delete any existing database at the target
            path before writing.

    Returns:
        ``True`` on success.
    """

    if device == "cpu":
        gpu_index = "-1"
        use_gpu = False
    elif device == "cuda":
        gpu_index = ",".join(str(i) for i in range(torch.cuda.device_count()))
        use_gpu = True
    elif device.startswith("cuda:"):
        gpu_index = device.split(":", 1)[1]
        use_gpu = True
    else:
        raise ValueError(f"Unsupported device: {device!r}")

    datbase_dir = dir_write + "/database_sift.db"
    if os.path.exists(datbase_dir) and remove_existing:
        os.remove(datbase_dir)

    camera_mode = "PER_IMAGE"

    images_list = [str(Path(p)) for p in images_list]

    # Extract the features for all images
    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = camera_model
    pycolmap.extract_features(
        datbase_dir,
        images_path,
        images_list,
        camera_mode,
        reader_opts,
        extraction_options=pycolmap.FeatureExtractionOptions(
            num_threads=16, gpu_index=gpu_index, use_gpu=use_gpu
        ),
    )  # use the same camera model for all images

    database = pycolmap.Database.open(datbase_dir)
    remap_cameras_to_intrinsics(database, images_list, intrinsics_mapping)

    if skip_matching:
        return True

    matched_pairs = list({(int(i), int(j)) for i, j in pairs.tolist()})

    # Write the match pair to a file
    with open(dir_write + "/pairs.txt", "w") as f:
        for i, j in matched_pairs:
            f.write(f"{images_list[i]} {images_list[j]}\n")

    pairs_path = dir_write + "/pairs.txt"
    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.gpu_index = gpu_index
    matching_options.use_gpu = use_gpu
    pairing_options = pycolmap.ImportedPairingOptions()
    pairing_options.match_list_path = pairs_path
    pycolmap.match_image_pairs(
        database_path=datbase_dir,
        matching_options=matching_options,
        pairing_options=pairing_options,
    )

    return True


def convert_to_colmap_format(
    images_shape_ori: list[tuple[int, int]],
    global_rotations: dict[int, np.ndarray],
    global_centers: dict[int, np.ndarray],
    global_intrinsics: list[torch.Tensor | None],
    intrinsics_mapping: dict[int, int],
    images_list: list[str] | None = None,
    camera_type: str | None = None,
) -> pycolmap.Reconstruction:
    """Assemble a :class:`pycolmap.Reconstruction` from pipeline outputs.

    Adds one camera per non-``None`` entry in ``global_intrinsics``, then
    one image per key in ``global_rotations`` (with image-to-world poses
    derived from ``global_rotations``/``global_centers``).

    Args:
        images_shape_ori: Per-image ``(height, width)`` of the originals.
        global_rotations: Image-id -> 3x3 world-from-camera rotation.
        global_centers: Image-id -> 3-vector camera center in world.
        global_intrinsics: Per-camera intrinsics tensors (shape
            ``(1, 3, 3)``); ``None`` slots are skipped.
        intrinsics_mapping: Maps each image index to a camera id.
        images_list: Image names; if ``None``, images are named by their
            integer index.
        camera_type: COLMAP camera model name (defaults to ``"PINHOLE"``).

    Returns:
        The constructed :class:`pycolmap.Reconstruction`. Camera and
        image ids are 1-indexed (``camera_id = intrinsics_mapping[i] + 1``,
        ``image_id = idx + 1``) to match COLMAP's on-disk convention.
    """
    reconstruction = pycolmap.Reconstruction()

    camera_sizes = [None for _ in range(len(global_intrinsics))]
    for i in range(len(intrinsics_mapping)):
        camera_id = intrinsics_mapping[i]
        if camera_sizes[camera_id] is None:
            camera_sizes[camera_id] = images_shape_ori[i]

    # Write cameras
    for i in range(len(global_intrinsics)):
        if global_intrinsics[i] is None:
            continue
        model = camera_type if camera_type is not None else "PINHOLE"
        camera = camera_from_intrinsics_matrix(
            global_intrinsics[i][0],
            model,
            camera_sizes[i][-1],
            camera_sizes[i][0],
            i + 1,
        )
        reconstruction.add_camera_with_trivial_rig(camera)

    # Add images
    for idx in global_rotations:
        image = pycolmap.Image()
        image.camera_id = intrinsics_mapping[idx] + 1

        if images_list is not None:
            image.name = images_list[idx]
        else:
            image.name = str(idx)
        image.image_id = idx + 1

        cam_from_world = pycolmap.Rigid3d(
            np.concatenate(
                [
                    global_rotations[idx][:3, :3].T,
                    global_centers[idx].reshape(3, 1),
                ],
                axis=1,
            )
        ).inverse()
        reconstruction.add_image_with_trivial_frame(image, cam_from_world)

    return reconstruction


def colorize_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    images_path: str,
    images_list: list[str] | None = None,
) -> int:
    """Sample point colors from observing images.

    For each 3D point with zero (black) color, samples the pixel color from
    each observing image at the 2D keypoint location and sets the median.

    Args:
        reconstruction: COLMAP reconstruction to colorize in place.
        images_path: Directory containing the source images.
        images_list: Optional list of image filenames to restrict lookup.

    Returns:
        Number of points whose color was updated.
    """
    from PIL import Image

    images_path = Path(images_path)
    image_files = None
    if images_list is not None:
        image_files = {Path(f).name for f in images_list}

    image_cache: dict[str, Image.Image] = {}
    points_updated = 0

    for point3D in reconstruction.points3D.values():
        if not np.all(point3D.color == 0):
            continue

        colors = []
        for element in point3D.track.elements:
            image_id = element.image_id
            if image_id not in reconstruction.images:
                continue
            image = reconstruction.images[image_id]

            if image_files is not None and image.name not in image_files:
                continue

            try:
                if image.name not in image_cache:
                    image_cache[image.name] = Image.open(
                        images_path / image.name
                    ).convert("RGB")
                img = image_cache[image.name]

                if element.point2D_idx >= len(image.points2D):
                    continue
                kp = image.points2D[element.point2D_idx]
                x = max(0, min(int(round(kp.xy[0])), img.width - 1))
                y = max(0, min(int(round(kp.xy[1])), img.height - 1))
                colors.append(np.array(img.getpixel((x, y)), dtype=np.uint8))
            except Exception:
                continue

        if colors:
            point3D.color = np.median(colors, axis=0).astype(np.uint8)
            points_updated += 1

    # Free image cache
    image_cache.clear()

    return points_updated


def write_to_colmap_format(
    dir_write: str,
    images_shape_ori: list[tuple[int, int]],
    global_rotations: dict[int, np.ndarray],
    global_centers: dict[int, np.ndarray],
    global_intrinsics: list[torch.Tensor | None],
    intrinsics_mapping: dict[int, int],
    images_list: list[str] | None = None,
    camera_type: str | None = None,
) -> None:
    """Write a reconstruction in COLMAP text format to ``dir_write``.

    Thin wrapper that calls :func:`convert_to_colmap_format` and then
    :meth:`pycolmap.Reconstruction.write` on the result. See
    :func:`convert_to_colmap_format` for parameter semantics.
    """
    os.makedirs(dir_write, exist_ok=True)

    reconstruction = convert_to_colmap_format(
        images_shape_ori,
        global_rotations,
        global_centers,
        global_intrinsics,
        intrinsics_mapping,
        images_list=images_list,
        camera_type=camera_type,
    )

    reconstruction.write(dir_write)
