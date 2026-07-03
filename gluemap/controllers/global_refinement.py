import contextlib
import logging
import os
import shutil
import time
from copy import deepcopy

import numpy as np
import pycolmap
import pygluemap
import torch

from gluemap.controllers.augmented_bundle_adjustment import (
    IterativeBAOptions,
    build_negative_depth_observations,
    build_reconstruction_for_ba,
    initialize_world_points,
    iterative_bundle_adjustment,
)
from gluemap.estimators.track_establishment import (
    TrackEstablishmentOptions,
    establish_tracks_from_predictions_dict,
)
from gluemap.math.reprojection_error import (
    ReprojectionErrorType,
    filter_reconstruction_by_reprojection_error,
)
from gluemap.utils.colmap import (
    camera_from_intrinsics_matrix,
    colorize_reconstruction,
    merge_colmap_databases,
    prepare_glomap_prior,
)

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_native_stdio():
    """Redirect fd 1 and fd 2 to /dev/null for the duration of the block.

    Needed because pycolmap.triangulate_points writes via C++ std::cout
    and glog directly to file descriptors, so contextlib.redirect_stdout
    is not enough.
    """
    saved_fds = [os.dup(1), os.dup(2)]
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_fds[0], 1)
        os.dup2(saved_fds[1], 2)
        os.close(devnull)
        os.close(saved_fds[0])
        os.close(saved_fds[1])


def _extract_track_csr(
    reconstruction: pycolmap.Reconstruction,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract track data from a reconstruction as CSR numpy arrays."""
    point3d_ids = []
    track_img_ids = []
    track_pt2d_idxs = []
    track_lengths = []
    for p3d_id, p3d in reconstruction.points3D.items():
        elems = list(p3d.track.elements)
        point3d_ids.append(p3d_id)
        track_lengths.append(len(elems))
        for e in elems:
            track_img_ids.append(e.image_id)
            track_pt2d_idxs.append(e.point2D_idx)
    return (
        np.array(point3d_ids, dtype=np.int64),
        np.array(track_img_ids, dtype=np.int64),
        np.array(track_pt2d_idxs, dtype=np.int64),
        np.array(track_lengths, dtype=np.int32),
    )


def _apply_deletions(
    reconstruction: pycolmap.Reconstruction,
    ids_to_delete: np.ndarray,
) -> None:
    """Delete point3D entries returned by a C++ track selector."""
    for p3d_id in ids_to_delete:
        reconstruction.delete_point3D(int(p3d_id))


def select_tracks_from_merged(
    reconstruction: pycolmap.Reconstruction,
    sift_count: dict[int, int],
    min_num_support_abs: int = 512,
) -> dict[int, int]:
    """
    Selectively prune non-SIFT tracks from a merged reconstruction.
    Returns the image-pair coverage map {canonical_pair_key: count}.
    The canonical key encodes (img_low, img_high) as (img_low << 32) | img_high.
    """
    point3d_ids, track_img_ids, track_pt2d_idxs, track_lengths = (
        _extract_track_csr(reconstruction)
    )
    sc = {int(k): int(v) for k, v in sift_count.items()}

    ids_to_delete, pair_count = pygluemap.compute_tracks_to_delete(
        point3d_ids,
        track_img_ids,
        track_pt2d_idxs,
        track_lengths,
        sc,
        min_num_support_abs,
    )
    _apply_deletions(reconstruction, ids_to_delete)
    return pair_count


def select_virtual_tracks_from_merged(
    virtual_reconstruction: pycolmap.Reconstruction,
    pair_count: dict[int, int],
    min_num_support_abs: int = 512,
) -> dict[int, int]:
    """
    Selectively prune virtual tracks using existing pair coverage.
    Removes tracks whose image pairs are all already sufficiently covered.
    Returns the updated pair_count.
    """
    if len(virtual_reconstruction.points3D) == 0:
        return pair_count

    point3d_ids, track_img_ids, track_pt2d_idxs, track_lengths = (
        _extract_track_csr(virtual_reconstruction)
    )

    ids_to_delete, updated_pair_count = (
        pygluemap.compute_virtual_tracks_to_delete(
            point3d_ids,
            track_img_ids,
            track_pt2d_idxs,
            track_lengths,
            pair_count,
            min_num_support_abs,
        )
    )
    _apply_deletions(virtual_reconstruction, ids_to_delete)
    return updated_pair_count


def triangulate_with_pycolmap(
    reconstruction: pycolmap.Reconstruction,
    database_path: str,
    triangulated_output_path: str,
    options: pycolmap.IncrementalPipelineOptions,
) -> pycolmap.Reconstruction:
    """Run pycolmap.triangulate_points silently on a deep copy.

    The reconstruction is already 1-indexed (matching the COLMAP database
    written by prepare_glomap_prior), so no reindexing is required. The input
    reconstruction is deep-copied first so the returned reconstruction does
    not overwrite the caller's. Native stdout/stderr is suppressed because
    triangulate_points is verbose.
    """
    reconstruction = deepcopy(reconstruction)
    with _suppress_native_stdio():
        reconstruction = pycolmap.triangulate_points(
            reconstruction,
            database_path,
            ".",  # skip color extraction
            triangulated_output_path,
            clear_points=True,
            refine_intrinsics=False,
            options=options,
        )
    logger.info(
        f"pycolmap.triangulate_points produced "
        f"{len(reconstruction.points3D)} 3D points"
    )
    return reconstruction


def run_refinement_pipeline(
    args,
    predictions_dict: dict,
    global_rotations,
    global_centers,
    global_intrinsics,
    dataset_pair,
    num_images: int,
    use_triangulation_first: bool = True,
    angular_error_threshold_deg: float = 0.5,
    num_refinement_iterations: int = 2,
    track_mode: str = "SPV",
) -> pycolmap.Reconstruction:
    """
    Run the refinement pipeline.

    Triangulation, track establishment, and bundle adjustment.

    Args:
        args: Argument namespace (needs curr_path, images_path)
        predictions_dict: Predictions from star inference
        global_rotations: Global rotation matrices for all images
        global_centers: Global camera center positions
        global_intrinsics: Camera intrinsic parameters for all images
        dataset_pair: Dataset pair object (needs camera_model,
            intrinsics_mapping, images_shape_ori, images_list)
        num_images: Number of images in the dataset
        use_triangulation_first: If True, triangulate SIFT + real tracks
            first, then add only virtual points. If False (default),
            triangulate SIFT only and establish both real tracks and virtual
            points.
        track_mode: Combination of S(IFT), P(rior), V(irtual) tracks to use.
            Valid modes: "SPV", "SP", "SV", "PV", "S", "P".

    Returns:
        pycolmap.Reconstruction: The bundle-adjusted reconstruction
    """
    t_refinement_start = time.perf_counter()
    refinement_timing = {}

    # Indexing convention: all data stored in COLMAP format (the database
    # written by prepare_glomap_prior, the pycolmap.Reconstruction returned
    # by build_reconstruction_for_ba, and anything keyed against either) is
    # 1-indexed for image_id and camera_id. The upstream Python-side
    # structures (global_rotations, keypoints_per_image, points3D track
    # elements, negative_depth_observations, etc.) remain 0-indexed and are
    # shifted at the COLMAP boundary.

    # Parse track mode flags
    use_sift = "S" in track_mode
    use_prior = "P" in track_mode
    use_virtual = "V" in track_mode
    logger.info(
        f"Track mode: {track_mode} (SIFT={use_sift}, "
        f"Prior={use_prior}, Virtual={use_virtual})"
    )

    # Step 1: Triangulate 3D points
    logger.info("Triangulating points with pycolmap...")
    t0 = time.perf_counter()
    suffix = getattr(args, "output_suffix", "")
    coarse_dir = f"coarse{suffix}"
    coarse_reconstruction = pycolmap.Reconstruction()
    coarse_reconstruction.read(args.curr_path + "/" + coarse_dir)
    refinement_timing["load_coarse"] = time.perf_counter() - t0

    # Step 1b: Determine parameters based on track mode
    if use_prior:
        database_name = "database_tracks.db"
        add_tracks = True
        log_message = "Creating tracks database (with prior tracks)..."
    else:
        database_name = "database_empty.db"
        add_tracks = False
        log_message = "Creating tracks database (empty, no prior tracks)..."

    # Step 1c: Create database with tracks (or empty)
    logger.info(log_message)
    t0 = time.perf_counter()
    prepare_glomap_prior(
        args.curr_path,
        dataset_pair.images_shape_ori,
        dataset_pair.images_list,
        global_intrinsics,
        predictions_dict,
        dataset_pair.intrinsics_mapping,
        camera_model=dataset_pair.camera_model,
        add_tracks=add_tracks,
        add_virtual_points=False,
        database_name=database_name,
    )
    refinement_timing["prepare_prior"] = time.perf_counter() - t0

    # Step 1c.5: Read SIFT DB keypoint counts (= sift_count per image)
    t0 = time.perf_counter()
    if use_sift:
        sift_db = pycolmap.Database.open(args.curr_path + "/database_sift.db")
        sift_count_by_name = {}
        for img in sift_db.read_all_images():
            kp = sift_db.read_keypoints(img.image_id)
            sift_count_by_name[img.name] = (
                len(kp) if kp is not None and len(kp) > 0 else 0
            )
    else:
        sift_count_by_name = {}
    refinement_timing["read_sift"] = time.perf_counter() - t0

    # Step 1d: Merge SIFT database with the created database (or copy if
    # no SIFT)
    t0 = time.perf_counter()
    merged_db_path = args.curr_path + "/database_merged.db"
    if use_sift:
        logger.info("Merging SIFT and tracks databases...")
        merge_colmap_databases(
            db_path_primary=args.curr_path + "/" + database_name,
            db_path_secondary=args.curr_path + "/database_sift.db",
            output_path=merged_db_path,
            # SIFT features should be at the front for correct indexing
            primary_features_first=False,
        )
    else:
        logger.info("Copying tracks database (no SIFT merge)...")
        shutil.copy2(args.curr_path + "/" + database_name, merged_db_path)
    refinement_timing["merge_databases"] = time.perf_counter() - t0

    # Step 2: Establish tracks from predictions
    t0 = time.perf_counter()
    track_options = TrackEstablishmentOptions(track_min_num_views_per_track=2)

    add_virtual_points_flag = use_virtual
    if use_triangulation_first and use_prior:
        # Real tracks already in DB for triangulation; only establish
        # virtual points
        add_tracks_flag = False
    elif use_prior:
        # Establish real tracks into reconstruction directly
        add_tracks_flag = True
    else:
        # No prior tracks requested
        add_tracks_flag = False

    (
        points3D,
        keypoints_per_image,
        pts2d_idx_inv,
        pts2d_idx_virtual_inv,
        images_points2d_virtual_isnegative,
    ) = establish_tracks_from_predictions_dict(
        predictions_dict=predictions_dict,
        num_images=num_images,
        options=track_options,
        add_tracks=add_tracks_flag,
        add_virtual_points=add_virtual_points_flag,
        device="cuda",
    )
    torch.cuda.empty_cache()
    refinement_timing["establish_tracks"] = time.perf_counter() - t0

    # Step 3: Initialize 3D world points
    t0 = time.perf_counter()
    cameras = [
        (
            camera_from_intrinsics_matrix(intr[0], dataset_pair.camera_model)
            if intr is not None
            else None
        )
        for intr in global_intrinsics
    ]
    negative_depth_observations = build_negative_depth_observations(
        pts2d_idx_inv, images_points2d_virtual_isnegative
    )
    points3D = initialize_world_points(
        predictions_dict,
        global_rotations,
        global_centers,
        points3D,
        pts2d_idx_inv,
        pts2d_idx_virtual_inv,
        keypoints_per_image=keypoints_per_image,
        cameras=cameras,
        intrinsics_mapping=dataset_pair.intrinsics_mapping,
        angular_error_threshold_deg=angular_error_threshold_deg,
        negative_depth_observations=negative_depth_observations,
    )
    refinement_timing["initialize_points"] = time.perf_counter() - t0

    # Step 4: Configure bundle adjustment
    ba_options = IterativeBAOptions(
        max_ba_iterations=200,
        max_filter_iterations=3,
        normalized_reproj_threshold=1e-2,
        min_track_length=2,
        fix_rotations_first_pass=False,
    )

    # Step 5: Build reconstruction from current data
    t0 = time.perf_counter()
    virtual_reconstruction = build_reconstruction_for_ba(
        global_rotations,
        global_centers,
        global_intrinsics,
        dataset_pair.intrinsics_mapping,
        points3D,
        keypoints_per_image,
        image_sizes=dataset_pair.images_shape_ori,
        images_list=dataset_pair.images_list,
        camera_model=dataset_pair.camera_model,
    )
    refinement_timing["build_reconstruction"] = time.perf_counter() - t0

    # build_reconstruction_for_ba emits 1-indexed image_ids, so consumers that
    # join against the reconstruction (BA, reprojection-error filter) need a
    # 1-indexed view of negative_depth_observations.
    negative_depth_observations_1indexed = {
        image_id + 1: pt_set
        for image_id, pt_set in negative_depth_observations.items()
    }

    database_path = args.curr_path + "/database_merged.db"
    triangulated_output_path = args.curr_path + "/coarse_triangulated"

    iteration_timings = []
    for outer_iter in range(num_refinement_iterations):
        logger.info(f"{'=' * 60}")
        logger.info(
            f"Refinement iteration {outer_iter + 1}/{num_refinement_iterations}"
        )
        logger.info(f"{'=' * 60}")
        t_iter_start = time.perf_counter()

        # Step 1e: Triangulate on merged database
        t_tri_start = time.perf_counter()
        opt_triang = pycolmap.IncrementalPipelineOptions()
        opt_triang.triangulation.min_angle = 1.0
        opt_triang.triangulation.merge_max_reproj_error = 15.0
        opt_triang.triangulation.complete_max_reproj_error = 15.0
        opt_triang.triangulation.ignore_two_view_tracks = False
        opt_triang.triangulation.create_max_angle_error = (
            angular_error_threshold_deg
        )
        opt_triang.ba_global_max_refinements = 0

        reconstruction = triangulate_with_pycolmap(
            virtual_reconstruction,
            database_path,
            triangulated_output_path,
            options=opt_triang,
        )
        t_tri_end = time.perf_counter()

        # Step 7a: Selectively prune prior/virtual tracks (SelectTrack logic)
        sift_count = {}
        for recon_id, img in reconstruction.images.items():
            sift_count[recon_id] = sift_count_by_name.get(img.name, 0)

        pair_count = select_tracks_from_merged(
            reconstruction=reconstruction,
            sift_count=sift_count,
            min_num_support_abs=512,
        )

        # Step 7a.2: Prune virtual tracks using pair coverage from real
        # selection
        if virtual_reconstruction is not None:
            select_virtual_tracks_from_merged(
                virtual_reconstruction=virtual_reconstruction,
                pair_count=pair_count,
                min_num_support_abs=512,
            )

        # Step 7.5: Filter tracks before bundle adjustment
        t_filter_start = time.perf_counter()
        if angular_error_threshold_deg > 0:
            for recon, neg_depth, label in (
                (reconstruction, None, "real: "),
                (
                    virtual_reconstruction,
                    negative_depth_observations_1indexed,
                    "virtual: ",
                ),
            ):
                if recon is None:
                    continue
                filter_reconstruction_by_reprojection_error(
                    recon,
                    ReprojectionErrorType.ANGULAR,
                    angular_error_threshold_deg,
                    negative_depth_observations=neg_depth,
                    log_prefix=label,
                )

        t_filter_end = time.perf_counter()

        # Step 7c: Limit number of tracks before BA
        max_num_tracks = getattr(args, "max_num_tracks", None)
        if (
            max_num_tracks is not None
            and len(reconstruction.points3D) > max_num_tracks
        ):
            sorted_ids = sorted(
                reconstruction.points3D.keys(),
                key=lambda pid: len(
                    list(reconstruction.points3D[pid].track.elements)
                ),
                reverse=True,
            )
            ids_to_remove = sorted_ids[max_num_tracks:]
            for pid in ids_to_remove:
                reconstruction.delete_point3D(pid)
            logger.info(
                f"  Track limit: kept {max_num_tracks}, "
                f"removed {len(ids_to_remove)} tracks"
            )

        # Step 8: Run iterative bundle adjustment
        t_ba_start = time.perf_counter()
        reconstruction, virtual_reconstruction = iterative_bundle_adjustment(
            reconstruction,
            virtual_reconstruction,
            negative_depth_observations_1indexed,
            options=ba_options,
        )
        t_ba_end = time.perf_counter()

        iter_timing = {
            "triangulation": t_tri_end - t_tri_start,
            "filter": t_filter_end - t_filter_start,
            "ba": t_ba_end - t_ba_start,
            "total": t_ba_end - t_iter_start,
        }
        iteration_timings.append(iter_timing)
        logger.info(
            f"[Profiling] Iteration {outer_iter + 1}: "
            f"triangulation={iter_timing['triangulation']:.2f}s, "
            f"filter={iter_timing['filter']:.2f}s, "
            f"ba={iter_timing['ba']:.2f}s, total={iter_timing['total']:.2f}s"
        )

    # Clean up triangulated reconstruction output
    if os.path.exists(triangulated_output_path):
        shutil.rmtree(triangulated_output_path)

    # Step 9: Write bundle adjusted results to COLMAP format
    t0 = time.perf_counter()
    suffix = getattr(args, "output_suffix", "")
    file_dir = f"gluemap_aba{suffix}"
    logger.info(
        "Writing bundle adjusted reconstruction: %s",
        args.curr_path + "/" + file_dir,
    )
    os.makedirs(args.curr_path + "/" + file_dir, exist_ok=True)

    # Colorize points from source images before writing
    num_colorized = colorize_reconstruction(
        reconstruction, args.images_path, dataset_pair.images_list
    )
    logger.info(f"Colorized {num_colorized} points from source images")

    reconstruction.write(args.curr_path + "/" + file_dir)
    refinement_timing["write_output"] = time.perf_counter() - t0

    refinement_timing["iterations"] = iteration_timings
    refinement_timing["total"] = time.perf_counter() - t_refinement_start

    logger.info("[Profiling] Refinement Summary:")
    logger.info(
        f"  Setup: load_coarse={refinement_timing['load_coarse']:.2f}s, "
        f"prepare_prior={refinement_timing['prepare_prior']:.2f}s, "
        f"merge_db={refinement_timing['merge_databases']:.2f}s, "
        f"establish_tracks={refinement_timing['establish_tracks']:.2f}s, "
        f"init_points={refinement_timing['initialize_points']:.2f}s, "
        f"build_recon={refinement_timing['build_reconstruction']:.2f}s"
    )
    logger.info(
        f"  Iterations: {sum(it['total'] for it in iteration_timings):.2f}s "
        f"({len(iteration_timings)} iters)"
    )

    logger.info(f"  Total refinement: {refinement_timing['total']:.2f}s")

    return file_dir, refinement_timing
