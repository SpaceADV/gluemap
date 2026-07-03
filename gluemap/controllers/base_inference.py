import abc
import argparse
import logging
import os
import time
from typing import Any, ClassVar

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from gluemap.utils.gpu import all_gather_object_cpu

logger = logging.getLogger(__name__)


class BaseInferencePipeline(abc.ABC):
    """Shared skeleton for dataloader-driven distributed inference pipelines.

    Concrete subclasses (``TwoViewInferencePipeline``,
    ``StarInferencePipeline``) declare three class attributes and override six
    hooks; the base owns the dataloader construction, model lifecycle, run()
    orchestration, distributed gather + index-reorder, timing accumulation,
    caching, and rank-0 save.
    """

    _index_key: ClassVar[str]
    _rerun_from_triggers: ClassVar[frozenset[str] | None]
    _profiling_label: ClassVar[str]

    def __init__(
        self,
        args: argparse.Namespace,
        world_size: int,
        rank: int,
        file_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        preloaded_models: dict[str, torch.nn.Module] | None = None,
    ):
        self.args = args
        self.world_size = world_size
        self.rank = rank
        self.file_name = file_name
        self.device = device
        self.dtype = dtype
        self.preloaded_models = preloaded_models
        self.models: dict[str, torch.nn.Module] | None = None
        self._owns_models = False

    # ------------------------------------------------------------------ hooks

    @abc.abstractmethod
    def _load_models(self) -> dict[str, torch.nn.Module]:
        """Return the model dict, loading lazily on first call.

        Implementations must populate ``self.models`` and set
        ``self._owns_models`` to ``True`` when models were freshly loaded (i.e.
        not supplied via ``preloaded_models``), so that ``_release_models`` can
        free GPU memory at end of ``run()``.
        """

    @abc.abstractmethod
    def _create_batch_inference(
        self, models: dict[str, torch.nn.Module]
    ) -> Any:
        """Construct the per-batch inference driver bound to ``models``."""

    @abc.abstractmethod
    def _run_batch_step(
        self, batch_inference: Any, batch: dict
    ) -> tuple[dict, dict[str, float]]:
        """Run one batch and return ``(outputs, extra_timings)``.

        ``extra_timings`` may carry additional per-batch timings (e.g.
        ``"forward_times"``, ``"tracking_times"``); the base accumulates them
        into parallel ``list[float]`` columns in the timing dict.
        """

    @abc.abstractmethod
    def _pack_local_outputs(
        self, all_outputs: list[dict], all_indices: list[int]
    ) -> dict:
        """Pack per-rank outputs into a single dict.

        The result is consumed by ``all_gather_object_cpu``.

        Implementations decide per-key tensor-vs-list shape. The base adds
        ``self._index_key`` to the returned dict; subclasses must not.
        """

    @abc.abstractmethod
    def _merge_gathered_outputs(
        self,
        data_list: list[dict],
        index_mapping: np.ndarray,
        dataset_size: int,
    ) -> dict:
        """Combine per-rank dicts into the final global dict.

        ``index_mapping[i]`` is the position within the concatenated per-rank
        outputs of the item whose original dataset index is ``i``.
        """

    @abc.abstractmethod
    def _postprocess_global_outputs(
        self, global_outputs: dict, dataset
    ) -> dict:
        """Optional final pass over the combined dict (e.g. attach metadata)."""

    def _batch_size(self) -> int:
        """DataLoader batch size; defaults to ``args.batch_size``."""
        return self.args.batch_size

    def _profiling_extra(
        self,
        batch_times: list[float],
        extra_timings: dict[str, list[float]],
    ) -> str:
        """Optional trailing fragment for the rank-0 profiling log line."""
        return ""

    # --------------------------------------------------------------- skeleton

    def _make_dataloader(self, dataset) -> torch.utils.data.DataLoader:
        """Build a (distributed when applicable) DataLoader."""
        if self.args.distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        return torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self._batch_size(),
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def _release_models(self) -> None:
        """Free models if we own them (i.e. not caller-supplied)."""
        if not self._owns_models or self.models is None:
            return
        for name in list(self.models):
            del self.models[name]
        self.models = None
        self._owns_models = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_inference(
        self, data_loader: torch.utils.data.DataLoader
    ) -> tuple[
        list[dict], list[int], list[float], dict[str, list[float]], float
    ]:
        """Iterate ``data_loader`` and run inference batch-by-batch.

        Returns ``(all_outputs, all_indices, batch_times, extra_timings,
        t_model_load)``.
        """
        all_outputs: list[dict] = []
        all_indices: list[int] = []
        batch_times: list[float] = []
        extra_timings: dict[str, list[float]] = {}

        t0_load = time.perf_counter()
        models = self._load_models()
        t_model_load = time.perf_counter() - t0_load

        batch_inference = self._create_batch_inference(models)

        with torch.no_grad():
            for batch in tqdm(
                data_loader,
                desc=f"Inference (Rank {self.rank})",
                disable=self.rank != 0,
            ):
                torch.cuda.synchronize()
                t_batch_start = time.perf_counter()

                outputs, extras = self._run_batch_step(batch_inference, batch)

                torch.cuda.synchronize()
                t_batch_end = time.perf_counter()
                batch_times.append(t_batch_end - t_batch_start)

                for key, val in extras.items():
                    extra_timings.setdefault(key, []).append(val)

                all_outputs.append(outputs)
                all_indices.extend(
                    batch[self._index_key].cpu().numpy().tolist()
                )

        # Release any GPU-resident models (e.g. backbone in low-VRAM mode)
        if hasattr(batch_inference, "unload_all"):
            batch_inference.unload_all()

        return (
            all_outputs,
            all_indices,
            batch_times,
            extra_timings,
            t_model_load,
        )

    def _gather_outputs(
        self,
        all_outputs: list[dict],
        all_indices: list[int],
        dataset,
    ) -> dict:
        """Gather per-rank outputs into the final global dict."""
        local_outputs = self._pack_local_outputs(all_outputs, all_indices)
        local_outputs[self._index_key] = all_indices

        if self.args.distributed:
            data_list = all_gather_object_cpu(
                local_outputs,
                tmpdir=self.args.temp_path + "/tmp_save",
                rank_zero_return_only=False,
                use_system_tmp=False,
            )
            order = [idx for out in data_list for idx in out[self._index_key]]
            index_mapping = np.zeros(len(dataset), dtype=np.int64)
            for i, idx in enumerate(order):
                index_mapping[idx] = i
            global_outputs = self._merge_gathered_outputs(
                data_list, index_mapping, len(dataset)
            )
        else:
            global_outputs = local_outputs

        return self._postprocess_global_outputs(global_outputs, dataset)

    def run(self, dataset) -> tuple[dict, dict[str, float | list[float] | int]]:
        """Run inference end-to-end and return ``(global_outputs, timing)``."""
        args = self.args
        cache_path = os.path.join(args.curr_path, self.file_name)

        rerun = getattr(args, "rerun_from", None)
        trig = self._rerun_from_triggers
        if (
            rerun is not None
            and (trig is None or rerun in trig)
            and os.path.exists(cache_path)
        ):
            os.remove(cache_path)
            logger.info(f"[rerun_from={rerun}] Deleted {cache_path}")

        batch_times: list[float] = []
        extra_timings: dict[str, list[float]] = {}
        t_model_load = 0.0

        if not (args.force_load and os.path.exists(cache_path)):
            data_loader = self._make_dataloader(dataset)
            (
                all_outputs,
                all_indices,
                batch_times,
                extra_timings,
                t_model_load,
            ) = self._run_inference(data_loader)

            global_outputs = self._gather_outputs(
                all_outputs, all_indices, dataset
            )

            if self.rank == 0:
                os.makedirs(args.curr_path, exist_ok=True)
                torch.save(global_outputs, cache_path)
        else:
            logger.info("Loading existing results...")
            global_outputs = torch.load(cache_path, weights_only=False)

        timing: dict[str, float | list[float] | int] = {
            "batch_times": batch_times,
            "num_batches": len(batch_times),
            "total": sum(batch_times) if batch_times else 0.0,
            "model_loading": t_model_load,
            **extra_timings,
        }
        if self.rank == 0 and batch_times:
            mean = sum(batch_times) / len(batch_times)
            logger.info(
                f"[Profiling] {self._profiling_label}: "
                f"{len(batch_times)} batches, "
                f"total={sum(batch_times):.2f}s, "
                f"mean={mean:.3f}s/batch"
                f"{self._profiling_extra(batch_times, extra_timings)}"
            )

        self._release_models()
        return global_outputs, timing
