import argparse
import os

import numpy as np
import torch
from safetensors.torch import load_file

import thirdparty.path_to_thirdparty  # noqa: F401  (adds all thirdparty submodules to sys.path)


def _import_vpr_model() -> type:
    """Import salad's :class:`VPRModel` while isolating its ``models`` package.

    Both ``thirdparty/salad/models/`` and
    ``thirdparty/doppelgangers-plusplus/dust3r/croco/models/`` claim the
    top-level ``models`` package, so we swap ``sys.modules`` around the
    import to keep the two from clobbering each other.
    """
    import sys

    if "vpr_model" in sys.modules:
        return sys.modules["vpr_model"].VPRModel

    salad_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../../thirdparty/salad")
    )
    saved = {
        k: v
        for k, v in sys.modules.items()
        if k == "models" or k.startswith("models.")
    }
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, salad_path)
    try:
        from vpr_model import VPRModel
    finally:
        if salad_path in sys.path:
            sys.path.remove(salad_path)
        salad_models = {
            k: v
            for k, v in sys.modules.items()
            if k == "models" or k.startswith("models.")
        }
        for k in salad_models:
            del sys.modules[k]
        sys.modules.update(saved)
    return VPRModel


def load_models(
    args: argparse.Namespace,
    keys: set[str] | None = None,
) -> tuple[dict[str, torch.nn.Module], torch.device]:
    """Load and prepare the requested set of pipeline models.

    Each entry in ``keys`` triggers loading of one model:

    * ``"pi3"`` / ``"pi3x"`` / ``"vggt"`` / ``"map_anything"`` — multi-view
      pose backbones, weights from ``args.path_feedforward``.
    * ``"dg"`` — Doppelganger++ (MASt3R), weights from ``args.path_dg``.
    * ``"vggsfm"`` — VGGSfM tracker, weights from ``args.path_tracker``.
    * ``"salad"`` — SALAD VPR descriptor, weights from
      ``args.path_retrieval``.

    Models are moved to CPU when ``args.cpu`` is set, otherwise to
    ``cuda`` when available. Under ``args.distributed`` they are wrapped
    in ``DistributedDataParallel``.

    Args:
        args: Namespace produced by :func:`gluemap.utils.cli.get_args_parser`,
            extended with distributed flags by :func:`init_distributed_mode`.
        keys: Which models to load. ``args.chosen_model`` is auto-added.

    Returns:
        ``(models, device)`` where ``models[args.chosen_model]`` is also
        aliased as ``models["mv"]`` for downstream callers.

    Note:
        ``keys`` defaults to ``None`` (treated as an empty set). The function
        does mutate the set passed in (adding ``args.chosen_model``);
        callers that pass their own set should expect that mutation.
    """
    if keys is None:
        keys = set()
    chosen_model = getattr(args, "chosen_model", "pi3")
    keys.add(chosen_model)

    models = {}

    # Load the chosen multi-view model
    if chosen_model == "pi3" and chosen_model in keys:
        from pi3.models.pi3 import Pi3

        models["pi3"] = Pi3()
        models["pi3"].load_state_dict(load_file(args.path_feedforward))
    elif chosen_model == "pi3x" and chosen_model in keys:
        from pi3.models.pi3x import Pi3X

        models["pi3x"] = Pi3X(use_multimodal=False)
        models["pi3x"].load_state_dict(
            load_file(args.path_feedforward), strict=False
        )
    elif chosen_model == "vggt" and chosen_model in keys:
        from vggt.models.vggt import VGGT

        models["vggt"] = VGGT()
        models["vggt"].load_state_dict(
            torch.load(
                args.path_feedforward, map_location="cpu", weights_only=False
            )
        )
    elif chosen_model == "map_anything" and chosen_model in keys:
        from mapanything.models.mapanything.model import MapAnything

        models["map_anything"] = MapAnything.from_pretrained(
            args.path_feedforward
        )
    elif chosen_model == "dvlt" and chosen_model in keys:
        from accelerate import Accelerator
        from dvlt.model.dvlt.model import DVLT

        accelerator = Accelerator(mixed_precision="bf16")
        models["dvlt"] = DVLT(img_size=518)
        models["dvlt"].load_pretrained("nvidia/dvlt", strict=True)
        models["dvlt"].model = accelerator.prepare(
            models["dvlt"].model
        )
        models["dvlt"].setup_test(accelerator)
        models["_dvlt_accelerator"] = accelerator

    # Load Doppelganger++
    if "dg" in keys:
        from mast3r.model import AsymmetricMASt3R

        models["dg"] = AsymmetricMASt3R(
            pos_embed="RoPE100",
            patch_embed_cls="ManyAR_PatchEmbed",
            img_size=(512, 512),
            head_type="catmlp+dpt",
            head_type_dg="transformer",
            output_mode="pts3d+desc24",
            output_mode_dg="dg_score",
            depth_mode=("exp", -np.inf, np.inf),
            conf_mode=("exp", 1, np.inf),
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
            two_confs=True,
            desc_conf_mode=("exp", 0, np.inf),
            add_dg_pred_head=True,
            freeze=["mask", "encoder", "decoder", "head"],
        ).from_pretrained(args.path_dg)

    # Load VGG-SfM Tracker
    if "vggsfm" in keys:
        from vggsfm.vggsfm_tracker import TrackerPredictor

        models["vggsfm"] = TrackerPredictor()
        models["vggsfm"].load_state_dict(
            torch.load(
                args.path_tracker, map_location="cpu", weights_only=False
            )
        )

    # Load SALAD
    if "salad" in keys:
        VPRModel = _import_vpr_model()
        models["salad"] = VPRModel(
            backbone_arch="dinov2_vitb14",
            backbone_config={
                "num_trainable_blocks": 4,
                "return_token": True,
                "norm_layer": True,
            },
            agg_arch="SALAD",
            agg_config={
                "num_channels": 768,
                "num_clusters": 64,
                "cluster_dim": 128,
                "token_dim": 256,
            },
        )

        models["salad"].load_state_dict(
            torch.load(
                args.path_retrieval, map_location="cpu", weights_only=False
            ),
            strict=False,
        )

    if getattr(args, "cpu", False):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for _model_name, model in models.items():
        if isinstance(model, torch.nn.Module):
            model.eval()
            model.to(device)

    if args.distributed:
        for model_name, model in models.items():
            models[model_name] = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.gpu],
                find_unused_parameters=True,
                static_graph=True,
            )

    # Create uniform alias for the chosen multi-view model
    if chosen_model in models:
        models["mv"] = models[chosen_model]

    return models, device


def load_all_models(
    args: argparse.Namespace,
) -> tuple[dict[str, torch.nn.Module], torch.device]:
    """Load all pipeline models at once for reuse across datasets.

    Respects args.use_dummy_tracks to skip VGGSfM if not needed.
    When direct_inference is enabled, only loads the backbone model.
    """
    chosen_model = getattr(args, "chosen_model", "pi3")
    if getattr(args, "direct_inference", False):
        keys = {chosen_model}
    else:
        keys = {"salad", "dg", chosen_model}
        if not getattr(args, "use_dummy_tracks", False):
            keys.add("vggsfm")
    return load_models(args, keys=keys)
