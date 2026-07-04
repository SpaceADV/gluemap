import numpy as np
import torch

from gluemap.ff_inference.local_inference import LocalInference


def _make_valid_pixels(images_change_np, img_h, img_w):
    """Build a valid-pixel mask from the pipeline's ``images_change``.

    The pipeline pads images with white (1.0).  DVLT expects black (0.0)
    padding and a boolean mask marking real pixels.  This function computes
    which pixels in the pipeline's preprocessed image correspond to actual
    image content (after resize + crop + pad), so DVLT can ignore the rest.

    Args:
        images_change_np: ``(N, 4)`` array of
            ``[scale_x, scale_y, x_offset, y_offset]`` per view.
        img_h: Preprocessed image height.
        img_w: Preprocessed image width.

    Returns:
        ``(1, N, H, W)`` bool tensor (``True`` = real pixel).
    """
    N = images_change_np.shape[0]
    masks = []
    for i in range(N):
        ox, oy = float(images_change_np[i, 2]), float(images_change_np[i, 3])
        # Content region in preprocessed coordinates:
        #   x_content = x_orig * scale_x + x_offset
        # The valid range is where x_orig >= 0 and x_orig < W_orig,
        # i.e. x_content in [-offset_x, -offset_x + W_orig*scale_x).
        # W_orig*scale_x = img_w - offset_x  (for square-padded images).
        x_min = max(0, int(round(-ox)))
        x_max = min(img_w, int(round(img_w + ox)))
        y_min = max(0, int(round(-oy)))
        y_max = min(img_h, int(round(img_h + oy)))
        m = torch.zeros(img_h, img_w, dtype=torch.bool)
        m[y_min:y_max, x_min:x_max] = True
        masks.append(m)
    return torch.stack(masks, dim=0).unsqueeze(0)  # (1, N, H, W)


class DVLTLocalInference(LocalInference):
    """Local inference adapter for the DVLT (Deja View) backbone.

    Uses the pipeline's preprocessed images (from ``batch["images"]``)
    directly — matching how Pi3 and VGGT adapters work.  White padding
    from the pipeline is zeroed out and a valid-pixel mask is passed to
    DVLT so it ignores synthetic border regions.

    This keeps DVLT's intrinsics and depth in the same coordinate space
    as the pipeline's ``images_change``, avoiding the projection drift
    caused by a coordinate-space mismatch.
    """

    def __init__(self, model, device, dtype, accelerator):
        super().__init__(model, device, dtype)
        self.accelerator = accelerator

    def predict(self, batch: dict) -> dict:
        """Run DVLT on a batch using the pipeline's preprocessed images.

        Instead of loading raw PIL images and applying DVLT's own
        preprocessing, this reuses ``batch["images"]`` (already
        preprocessed by the pipeline) and constructs a valid-pixel
        mask from ``batch["images_change"]``.  Intrinsics and depth
        are therefore in the pipeline's standard coordinate space.
        """
        from dvlt.common.constants import (
            DataField,
            PredictionField,
        )

        images = batch["images"]  # (B, N, 3, H, W) pipeline-preprocessed
        images_change_raw = batch["images_change"]

        if isinstance(images_change_raw, torch.Tensor):
            ic_np = images_change_raw.numpy()
        elif isinstance(images_change_raw, np.ndarray):
            ic_np = images_change_raw
        else:
            ic_np = np.array([images_change_raw])

        B, N, _, H, W = images.shape

        all_depth = []
        all_depth_conf = []
        all_extrinsics = []
        all_intrinsics = []

        for i in range(B):
            imgs = images[i].to(self.device)  # (N, 3, H, W)

            # Zero out white padding so DVLT sees black borders
            ic_i = ic_np[i] if ic_np.ndim == 3 else ic_np
            valid_pixels = _make_valid_pixels(ic_i, H, W).to(self.device)
            # Expand mask to (N, 3, H, W) for broadcasting with images
            mask_3d = valid_pixels[0].unsqueeze(1)  # (N, 1, H, W)
            imgs = imgs * mask_3d

            dvlt_batch = {
                DataField.IMAGES: imgs.unsqueeze(0),  # (1, N, 3, H, W)
                "gradio_valid_pixels": valid_pixels,
            }

            with torch.no_grad():
                predictions = self.model.predict(
                    dvlt_batch, self.accelerator
                )

            cameras = predictions[PredictionField.CAMERAS][0]
            extrinsics_c2w = cameras.camera_to_worlds
            intrinsics = cameras.get_intrinsics_matrices()

            S = extrinsics_c2w.shape[0]
            bottom_row = torch.zeros(
                S, 1, 4,
                device=self.device,
                dtype=extrinsics_c2w.dtype,
            )
            bottom_row[:, 0, 3] = 1.0
            extrinsics_c2w_4x4 = torch.cat(
                [extrinsics_c2w, bottom_row], dim=1
            )
            extrinsics = torch.linalg.inv(
                extrinsics_c2w_4x4
            ).unsqueeze(0)

            depth = predictions[PredictionField.DEPTHS].unsqueeze(-1)
            depth_conf = predictions[PredictionField.DEPTHS_CONF]

            all_depth.append(depth)
            all_depth_conf.append(depth_conf)
            all_extrinsics.append(extrinsics)
            all_intrinsics.append(intrinsics.unsqueeze(0))

        return {
            "depth": torch.cat(all_depth, dim=0),
            "depth_conf": torch.cat(all_depth_conf, dim=0),
            "extrinsics": torch.cat(all_extrinsics, dim=0),
            "intrinsics": torch.cat(all_intrinsics, dim=0),
        }
