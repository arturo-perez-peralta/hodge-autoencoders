from collections.abc import Callable, Iterator
import gc
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from src.hodge_sae.datasets.base import ActivationBatch, ActivationDataLoader


class CachedActivationDataLoader(ActivationDataLoader):
    def __init__(
        self,
        cache_dir: str | Path,
        batch_size: int,
        device: str | torch.device,
        source_fn: Callable[[], ActivationDataLoader] | None = None,
        max_samples: int | None = None,
        shard_size: int = 250_000,        
        shuffle: bool = True,
        tensor_key: str = "activations",
    ):
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.shuffle = shuffle
        self.tensor_key = tensor_key

        meta_path = self.cache_dir / "metadata.json"

        if not meta_path.exists():
            if source_fn is None:
                raise FileNotFoundError(
                    f"No valid cache in {self.cache_dir} y and source_fn was not provided."
                )
            self._build_cache(source_fn, max_samples=max_samples, shard_size=shard_size)

        self.shards = sorted(self.cache_dir.glob("shard_*.safetensors"))
        if not self.shards:
            raise FileNotFoundError(f"Corrupted cache in {self.cache_dir}: no shards.")

        with safe_open(self.shards[0], framework="pt") as f:
            self._d_in = int(f.get_slice(self.tensor_key).get_shape()[-1])

    @property
    def d_in(self) -> int:
        return self._d_in

    def _build_cache(
        self,
        source_fn: Callable[[], ActivationDataLoader],
        max_samples: int | None,
        shard_size: int,
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        source = source_fn()

        buffer: list[torch.Tensor] = []
        buffered_count = 0
        total_saved = 0
        shard_idx = 0

        def flush_shard() -> None:
            nonlocal buffer, buffered_count, shard_idx
            if not buffer:
                return
            chunk = torch.cat(buffer, dim=0)
            shard_path = self.cache_dir / f"shard_{shard_idx:04d}.safetensors"
            save_file({self.tensor_key: chunk}, shard_path)
            shard_idx += 1
            buffer = []
            buffered_count = 0

        for batch in source:
            acts = batch.activations.detach().cpu()

            if max_samples is not None and total_saved + acts.shape[0] > max_samples:
                acts = acts[: max_samples - total_saved]

            buffer.append(acts)
            buffered_count += acts.shape[0]
            total_saved += acts.shape[0]

            if buffered_count >= shard_size:
                flush_shard()

            if max_samples is not None and total_saved >= max_samples:
                break

        flush_shard()

        # Write the sentinel only when the shards are safe in disk
        meta = {
            "d_in": source.d_in,
            "total_samples": total_saved,
            "num_shards": shard_idx,
        }
        (self.cache_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        # Delete the LLM from memory
        del source
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __iter__(self) -> Iterator[ActivationBatch]:
        shard_order = list(self.shards)
        if self.shuffle:
            perm = torch.randperm(len(shard_order)).tolist()
            shard_order = [shard_order[i] for i in perm]

        for shard_path in shard_order:
            acts = load_file(shard_path)[self.tensor_key]
            if self.shuffle:
                acts = acts[torch.randperm(acts.shape[0])]

            for i in range(0, acts.shape[0] - self.batch_size + 1, self.batch_size):
                batch = acts[i : i + self.batch_size].to(self.device, non_blocking=True)
                yield ActivationBatch(activations=batch)