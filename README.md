# GLUEMAP: Global Structure-from-Motion Meets Feedforward Reconstruction

> **This is a fork of [colmap/gluemap](https://github.com/colmap/gluemap)** with the following additions:
>
> - **DVLT backbone** — NVIDIA's [Déjà View Looping Transformer](https://research.nvidia.com/labs/dvl/projects/dvlt/) as a `chosen_model: dvlt` option. Uses `nvidia/dvlt` weights from HuggingFace Hub.
> - **`--low_vram` mode** — sequential CPU↔GPU model swapping for GPUs with limited VRAM (tested on 8 GB GPUs). Use with `--skip_doppelgangers` on 8 GB cards.
> - **Point colorization** — output COLMAP reconstructions include per-point RGB colors sampled from observing images.
> - **Bug fixes** — empty-neighbor-index crash in global merger, GPU OOM when COLMAP SIFT extraction follows feedforward inference, and other edge cases.

[Project page](https://lpanaf.github.io/cvpr26_gluemap/) | [Paper](https://arxiv.org/abs/2605.26103)
---

## About

GLUEMAP is the first Structure-from-Motion pipeline to integrate
feed-forward reconstruction backbones into a global SfM framework.
It takes a collection of images as input and outputs a COLMAP sparse reconstruction. 
It matches the scalability and accuracy of traditional SfM pipelines while inheriting the local robustness of a feed-forward backbone.
It pairs their resilience on hard local geometry (low overlap, repetitive structures, weak texture) with the accuracy, global consistency, and large-scene tractability that classical SfM provides.

The pipeline comprises:

1. **Retrieval** — SALAD global descriptors build the image neighbor graph.
2. **Two-view inference (optional)** — Doppelgangers++ (MAST3R-based) estimates covisibility of pairs. Can be skipped via `skip_doppelgangers` when the scene has no repetitive structure or when a quick result is preferred (see [Configuration](#configuration)).
3. **Multi-view inference** — a configurable backbone (Pi3 / Pi3X / VGGT / MapAnything) estimates poses and geometry in local star configurations.
4. **Global mapping** — rotation averaging, intrinsics averaging,
   similarity averaging, and global bundle adjustment fuse the local solutions.
5. **Refinement** — SIFT track snapping with iterative augmented bundle adjustment.

The orchestrator is [gluemap/controllers/gluemap_impl.py](gluemap/controllers/gluemap_impl.py).

If you use this project for your research, please cite

```
@inproceedings{pan2026gluemap,
    author={Pan, Linfei and Sch\"{o}nberger, Johannes Lutz and Pollefeys, Marc},
    title={Global Structure-from-Motion Meets Feedforward Reconstruction},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2026},
}
```


## Getting Started

GLUEMAP is a Python package with a C++/pybind11 extension (`pygluemap`)
that links against Ceres, Eigen, Boost, and OpenMP, plus several
feed-forward models vendored as git submodules.

```bash
git clone https://github.com/colmap/gluemap.git
cd gluemap
git submodule update --init --recursive

# In a Python ≥ 3.10 environment with Ceres / Eigen / METIS / Boost / OpenMP available:
CMAKE_PREFIX_PATH=$CONDA_PREFIX pip install -e .
```

For the reproducible conda/micromamba recipe (pinned versions, CUDA
notes, libstdc++ ABI troubleshooting), see [INSTALL.md](INSTALL.md).

The default config expects four model checkpoints under `checkpoints/`
(Pi3, SALAD, VGGSfM tracker, Doppelgangers++). Download commands are in
[INSTALL.md §4](INSTALL.md#4-download-model-checkpoints).

Verify the install:

```bash
python -c "import gluemap; import pygluemap; print(pygluemap.__file__)"
gluemap-demo --help
```

## Usage

### Single image collection

```bash
gluemap-demo \
    --config configs/example.yaml \
    --images_path /path/to/images \
    --intrinsics_mode SHARED \
    --write_path results/
```

The reconstruction is written under `--write_path` in COLMAP format.

### Multi-sequence

When `--images_path` holds several video sequences of the same scene as
sibling subfolders (e.g. LAMAR's `ios*` sequences), enable multi-sequence
mode and supply a regex that selects the subfolders to process:

```bash
gluemap-demo \
    --config configs/example.yaml \
    --images_path /path/to/scene \
    --subfolder_regex '^ios' \
    --is_multi_sequence \
    --intrinsics_mode PER_CAMERA \
    --write_path results/
```

This auto-enables sequential pairing within each subfolder.

### Multi-GPU

Two-view and star-inference stages parallelize across ranks via
`init_distributed()` in [gluemap/utils/gpu.py](gluemap/utils/gpu.py).
Global mapping and refinement run on rank 0 only. Launch with
`torchrun --nproc_per_node=N gluemap-demo ...` (or set `RANK` /
`WORLD_SIZE` manually).

### Benchmarks

`gluemap-benchmark` runs a config-driven sweep over multiple datasets
(see [configs/lamar.yaml](configs/lamar.yaml) for the LAMAR layout).

## Configuration

Configs live in [configs/](configs/) and inherit shared defaults from
[configs/base.yaml](configs/base.yaml) via `_base_:`. CLI flags override
config values, so anything in `base.yaml` can be tweaked without editing
the file.

The most-touched knobs:

| Key | Description |
|---|---|
| `images_path` / `write_path` | Input directory / output directory. |
| `chosen_model` | Multi-view backbone: `pi3` (default), `pi3x`, `vggt`, `map_anything`, `dvlt`. |
| `--low_vram` | CLI flag: keep models in CPU RAM and swap to/from GPU per batch. Required for ≤8 GB GPUs. |
| `path_feedforward` | Checkpoint for the chosen multi-view model. |
| `path_retrieval` / `path_tracker` / `path_dg` | SALAD / VGGSfM / Doppelgangers++ checkpoints. |
| `camera_model` | COLMAP camera model (default `SIMPLE_PINHOLE`). |
| `intrinsics_mode` | Intrinsics-bucketing strategy: `SHARED` (one camera per unique image shape, default), `PER_FOLDER` (one camera per `(dirname, shape)` pair), or `PER_CAMERA` (one camera per image). |
| `num_neighbors` | Neighbors per image in the retrieval graph (default `100`). |
| `is_sequential` / `sample_frequency` | Use temporal pairing for ordered video; subsample every Nth frame. |
| `is_multi_sequence` / `subfolder_regex` | Process several sibling sequences into a single reconstruction. |
| `rerun_from` | Resume from `retrieval`, `twoview`, or `star` to skip earlier stages. |
| `coarse_only` | Stop after global mapping; skip the refinement stage. |
| `skip_doppelgangers` | Skip the Doppelgangers++ two-view disambiguator and treat all retrieval pairs as valid. Useful when the scene has no repetitive structure or for a quick first result. Default `false`. |

See [configs/base.yaml](configs/base.yaml) for the complete surface and
default values.

<details>
<summary><b>Choosing a backbone</b></summary>

`chosen_model` selects the multi-view star-inference model. Each
backbone needs its own `path_feedforward` checkpoint:

- **`pi3`** — Pi3, default; `checkpoints/pi3.safetensors`.
- **`pi3x`** — Pi3X variant; same Pi3 checkpoint family.
- **`vggt`** — Facebook VGGT-1B; download `model.pt` from
  HuggingFace `facebook/VGGT-1B`.
- **`dvlt`** — NVIDIA DVLT (Déjà View Looping Transformer); weights
  downloaded automatically from `nvidia/dvlt` on HuggingFace Hub.
- **`map_anything`** — Facebook MapAnything; set
  `path_feedforward: facebook/map-anything` (HF repo id, not a file path).

Dispatch lives in [gluemap/utils/model_loader.py](gluemap/utils/model_loader.py).

</details>

<details>
<summary><b>Adding a custom backbone</b></summary>

To plug in a different multi-view model, add a wrapper under
[gluemap/ff_inference/](gluemap/ff_inference/) that inherits the
`LocalInference` abstract base class from
[gluemap/ff_inference/local_inference.py](gluemap/ff_inference/local_inference.py)
and implements `predict(batch: dict) -> dict`.

The contract:

- **Input** `batch["images"]` of shape `(B, N, 3, H, W)`. Subclasses may
  read additional keys.
- **Output** dict with at minimum `depth`, `depth_conf`, `extrinsics`,
  `intrinsics`.

Use the existing wrappers as references:

- [gluemap/ff_inference/pi3_inference.py](gluemap/ff_inference/pi3_inference.py)
- [gluemap/ff_inference/vggt_inference.py](gluemap/ff_inference/vggt_inference.py)
- [gluemap/ff_inference/dvlt_inference.py](gluemap/ff_inference/dvlt_inference.py)
- [gluemap/ff_inference/mapanything_inference.py](gluemap/ff_inference/mapanything_inference.py)

Then register the new wrapper in `create_local_inference()` in
[local_inference.py](gluemap/ff_inference/local_inference.py) so it
dispatches on the `chosen_model` string set in your config.

> **Note:** the input image size and patch size are fixed in
> [gluemap/datasets/star.py](gluemap/datasets/star.py) (`image_size=518`,
> `patch_size=14`), matching the Pi3/DINOv2-style encoders used by the
> default backbones. If your custom backbone uses a different encoder
> (e.g. DINOv3, which uses a patch size of 16), update `self.patch_size`
> (and `self.image_size` if needed) accordingly so the preprocessed
> images align with the encoder's patch grid.

</details>

## Acknowledgments

GLUEMAP stands on a stack of upstream feed-forward and geometry models:

- [COLMAP](https://github.com/colmap/colmap) — output format and broader SfM ecosystem
- [Doppelgangers++](https://github.com/doppelgangers25/doppelgangers-plusplus) — two-view disambiguator (with MAST3R / DUSt3R / CroCo)
- [SALAD](https://github.com/serizba/salad) — DINO-based image retrieval
- [VGGSfM](https://github.com/facebookresearch/vggsfm) — point tracker
- [LightGlue](https://github.com/cvg/LightGlue) — feature extractor

Multi-view feedforward backbones:
- [Pi3](https://github.com/yyfz/Pi3) — multi-view pose estimation
- [VGGT](https://github.com/facebookresearch/vggt) — multi-view geometry transformer
- [DVLT](https://github.com/nv-tlabs/dvlt) — looping transformer for multi-view geometry
- [MapAnything](https://github.com/facebookresearch/map-anything) — feed-forward 3D mapping

## Support

Please, use GitHub Discussions at https://github.com/colmap/gluemap/discussions
for questions and the GitHub issue tracker at https://github.com/colmap/gluemap
for bug reports, feature requests/additions, etc.

## Contribution

Contributions (bug reports, bug fixes, improvements, etc.) are very welcome and
should be submitted in the form of new issues and/or pull requests on GitHub.

## License

GLUEMAP is licensed under the new BSD license. Note that this text refers
only to the license for GLUEMAP itself, independent of its thirdparty
dependencies, which are separately licensed. Building GLUEMAP with these
dependencies may affect the resulting GLUEMAP license.

    Copyright (c), ETH Zurich.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.

        * Neither the name of ETH Zurich nor the names of its contributors
          may be used to endorse or promote products derived from this
          software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
