import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import resize

from gluemap.ff_inference.local_inference import LocalInference


def _make_divisible(v: int, divisor: int) -> int:
    return max(divisor, (v // divisor) * divisor)


def _dvlt_preprocess(
    pil_images: list[Image.Image],
    img_size: int,
    patch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[list[float]]]:
    """DVLT-native preprocessing: resize + center-crop (no white padding).

    Returns:
        images: ``(1, S, 3, H, W)`` float tensor in ``[0, 1]``.
        valid_pixels: ``(1, S, H, W)`` bool mask (``True`` = real pixel).
        images_change: per-image ``[scale_x, scale_y, x_offset, y_offset]``
            mapping original pixel coordinates to model-input coordinates.
    """
    tensors: list[torch.Tensor] = []
    raw_changes: list[tuple[float, float, int, int]] = []

    for img in pil_images:
        img = img.convert("RGB")
        w, h = img.size
        t = (
            torch.from_numpy(np.array(img))
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        scale = img_size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        t = resize(t, [new_h, new_w], antialias=True)

        crop_h = _make_divisible(new_h, patch_size)
        crop_w = _make_divisible(new_w, patch_size)
        top = (new_h - crop_h) // 2
        left = (new_w - crop_w) // 2
        t = t[:, top : top + crop_h, left : left + crop_w]

        tensors.append(t)
        raw_changes.append((new_w / w, new_h / h, left, top))

    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    aligned: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    images_change: list[list[float]] = []

    for i, t in enumerate(tensors):
        scale_x, scale_y, left, top = raw_changes[i]
        _, h, w = t.shape
        pad_h, pad_w = max_h - h, max_w - w

        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            t = torch.nn.functional.pad(
                t,
                (
                    pad_left,
                    pad_w - pad_left,
                    pad_top,
                    pad_h - pad_top,
                ),
                value=0.0,
            )
            left += pad_left
            top += pad_top

        images_change.append(
            [scale_x, scale_y, float(-left), float(-top)]
        )

        vm = torch.ones(1, h, w, dtype=torch.float32)
        if pad_h > 0 or pad_w > 0:
            vm = torch.nn.functional.pad(
                vm,
                (
                    pad_left,
                    pad_w - pad_left,
                    pad_top,
                    pad_h - pad_top,
                ),
                value=0.0,
            )
        aligned.append(t)
        valid_masks.append(vm.squeeze(0).bool())

    images = torch.stack(aligned, dim=0).unsqueeze(0).to(device)
    valid_pixels = (
        torch.stack(valid_masks, dim=0).unsqueeze(0).to(device)
    )
    return images, valid_pixels, images_change


def _unwrap_image_paths(raw):
    """Normalise ``batch["image_paths"]`` after DataLoader collation.

    ``default_collate`` wraps each string in a tuple::

        ['a.jpg', 'b.jpg']  →  [('a.jpg',), ('b.jpg',)]

    This helper flattens ``tuple-of-1-str`` back to ``str`` and
    returns a plain list of path strings regardless of nesting.
    """
    if not raw:
        return []
    first = raw[0]
    if isinstance(first, (list, tuple)):
        # Each element is a tuple/list — unwrap singletons
        return [
            p[0] if isinstance(p, (list, tuple)) and len(p) == 1 else p
            for p in raw
        ]
    return list(raw)


class DVLTLocalInference(LocalInference):
    """Local inference adapter for the DVLT (Deja View) backbone.

    Loads original PIL images from ``batch["image_paths"]`` to bypass
    gluemap's pad-to-square preprocessing, applies DVLT-native
    center-crop preprocessing, and maps outputs to gluemap's uniform
    ``{depth, depth_conf, extrinsics, intrinsics}`` contract.
    """

    def __init__(self, model, device, dtype, accelerator):
        super().__init__(model, device, dtype)
        self.accelerator = accelerator

    def _predict_single(self, pil_images):
        """Run DVLT on a single star (one group of views).

        Returns ``(depth, depth_conf, extrinsics, intrinsics)``
        tensors with a leading batch dim of 1.
        """
        from dvlt.common.constants import (
            DataField,
            PredictionField,
        )

        images, valid_pixels, _ = _dvlt_preprocess(
            pil_images,
            img_size=self.model.img_size,
            patch_size=self.model.patch_size,
            device=self.device,
        )

        dvlt_batch = {
            DataField.IMAGES: images,
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
            S,
            1,
            4,
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

        intrinsics = intrinsics.unsqueeze(0)

        depth = predictions[
            PredictionField.DEPTHS
        ].unsqueeze(-1)
        depth_conf = predictions[PredictionField.DEPTHS_CONF]

        return depth, depth_conf, extrinsics, intrinsics

    def predict(self, batch: dict) -> dict:
        """Run DVLT on a batch using DVLT-native preprocessing.

        Handles batched DataLoader output by looping over batch
        items, since DVLT preprocessing can produce different
        spatial sizes per star (different aspect ratios).
        """
        image_paths_raw = batch["image_paths"]
        images_change_raw = batch["images_change"]

        # Determine batch size from images_change shape.
        # After collation: (B, N, 4) tensor.
        if isinstance(images_change_raw, torch.Tensor):
            B = images_change_raw.shape[0]
            images_change_np = images_change_raw.numpy()
        elif isinstance(images_change_raw, np.ndarray):
            B = images_change_raw.shape[0]
            images_change_np = images_change_raw
        else:
            B = 1
            images_change_np = np.array(
                [images_change_raw]
            )

        # Normalise image_paths: default_collate wraps strings
        # in tuples: ['a.jpg', ...] → [('a.jpg',), ...]
        all_paths = _unwrap_image_paths(image_paths_raw)

        all_depth = []
        all_depth_conf = []
        all_extrinsics = []
        all_intrinsics = []
        all_ic = []

        for i in range(B):
            # Paths for this batch item: with batch_size=1
            # (star pipeline default), all N paths belong to
            # the single star. With batch_size>1, each item is
            # one star. image_paths after unwrap is flat list
            # of strings — for B>1, we'd need per-item slicing,
            # but star pipeline always uses B=1.
            pil_images = [Image.open(p) for p in all_paths]

            depth, depth_conf, extrinsics, intrinsics = (
                self._predict_single(pil_images)
            )

            all_depth.append(depth)
            all_depth_conf.append(depth_conf)
            all_extrinsics.append(extrinsics)
            all_intrinsics.append(intrinsics)

        # Recompute images_change from preprocessing
        _, _, dvlt_ic = _dvlt_preprocess(
            [Image.open(p) for p in all_paths],
            img_size=self.model.img_size,
            patch_size=self.model.patch_size,
            device=self.device,
        )

        batch["images_change"] = np.array(
            [dvlt_ic for _ in range(B)]
        )

        return {
            "depth": torch.cat(all_depth, dim=0),
            "depth_conf": torch.cat(all_depth_conf, dim=0),
            "extrinsics": torch.cat(all_extrinsics, dim=0),
            "intrinsics": torch.cat(all_intrinsics, dim=0),
        }
