from collections.abc import Iterator
from typing import Any

from datasets import load_dataset
import torch
from transformer_lens import HookedTransformer

from src.hodge_sae.datasets.base import ActivationBatch, ActivationDataLoader


class TextDataLoader(ActivationDataLoader):
    def __init__(
        self,
        model_name: str,
        hook_name: str,
        dataset_name: str,
        device: str | torch.device,
        llm_batch_size: int = 32,   
        sae_batch_size: int = 4096, 
        buffer_batches: int = 16,   
        seq_len: int = 128,
        text_column: str = "text",
        split: str = "train",
    ):
        self.model_name = model_name
        self.hook_name = hook_name
        self.dataset_name = dataset_name
        self.device = torch.device(device)
        self.llm_batch_size = llm_batch_size
        self.sae_batch_size = sae_batch_size
        self.buffer_batches = buffer_batches
        self.seq_len = seq_len
        self.text_column = text_column
        self.split = split

        self.model = HookedTransformer.from_pretrained(model_name, device=self.device)
        self.model.eval()
        if self.model.tokenizer.pad_token is None:
            self.model.tokenizer.pad_token = self.model.tokenizer.eos_token

        self._d_in = (
            self.model.cfg.d_mlp if "mlp" in hook_name else self.model.cfg.d_model
        )

    @property
    def d_in(self) -> int:
        return self._d_in

    def _get_token_batch(self, data_iter: Iterator[Any]) -> torch.Tensor:
        texts = [next(data_iter)[self.text_column] for _ in range(self.llm_batch_size)]
        return self.model.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.seq_len,
            padding="max_length",
        )["input_ids"].to(self.device)

    @torch.no_grad()
    def _fill_buffer(self, data_iter: Iterator[Any]) -> torch.Tensor:
        chunks = []
        for _ in range(self.buffer_batches):
            tokens = self._get_token_batch(data_iter)
            _, cache = self.model.run_with_cache(
                tokens,
                names_filter=lambda name: name == self.hook_name,
            )
            chunks.append(cache[self.hook_name].reshape(-1, self.d_in))

        buffer = torch.cat(chunks, dim=0)
        perm = torch.randperm(buffer.shape[0], device=self.device)
        return buffer[perm]

    def __iter__(self) -> Iterator[ActivationBatch]:
        dataset = load_dataset(self.dataset_name, split=self.split, streaming=True)
        data_iter = iter(dataset)

        while True:
            try:
                buffer = self._fill_buffer(data_iter)
            except StopIteration:
                break

            for i in range(0, buffer.shape[0] - self.sae_batch_size + 1, self.sae_batch_size):
                acts = buffer[i : i + self.sae_batch_size]
                yield ActivationBatch(activations=acts)


class PileDataLoader(TextDataLoader):
    def __init__(
        self,
        device: str | torch.device,
        model_name: str = "gelu-1l",
        hook_name: str = "blocks.0.mlp.hook_post",
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            hook_name=hook_name,
            dataset_name="monology/pile-uncopyrighted",
            device=device,
            **kwargs,
        )