from collections.abc import Iterator

from huggingface_hub import hf_hub_download
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from src.hodge_sae.datasets.base import ActivationBatch, ActivationDataLoader


class ArithmeticDataLoader(ActivationDataLoader):
    REPO_ID = "callummcdougall/grokking_full_run_data"
    FILENAME = "full_run_data.pth"

    def __init__(
        self,
        device: str | torch.device,
        hook_name: str = "blocks.0.mlp.hook_post",
        batch_size: int = 256,
        token_pos: int = -1,       # -1 = Token '=' position
        checkpoint_idx: int = 400, # Checkpoint after grokking
        shuffle: bool = True,
        drop_last: bool = False,
        p: int = 113,
    ):
        self.device = torch.device(device)
        self.hook_name = hook_name
        self.batch_size = batch_size
        self.token_pos = token_pos
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.p = p

        cfg = HookedTransformerConfig(
            n_layers=1,
            d_vocab=p + 1,
            d_model=128,
            d_mlp=4 * 128,
            n_heads=4,
            d_head=128 // 4,
            n_ctx=3,
            act_fn="relu",
            normalization_type=None,
            device=str(self.device),
        )

        full_run_data_path = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=self.FILENAME,
        )
        full_run_data = torch.load(
            full_run_data_path,
            map_location=self.device,
            weights_only=False,
        )
        state_dict = full_run_data["state_dicts"][checkpoint_idx]

        self.model = HookedTransformer(cfg)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # All pairs (a, b, '=') -> shape (12769, 3)
        self.all_data = torch.tensor(
            [(i, j, p) for i in range(p) for j in range(p)],
            dtype=torch.long,
            device=self.device,
        )
        self.labels = (self.all_data[:, 0] + self.all_data[:, 1]) % p

        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                self.all_data,
                names_filter=lambda name: name == self.hook_name,
            )
            # Final shape: (12769, d_in)
            self.activations = cache[self.hook_name][:, self.token_pos, :].detach()

        self._d_in = int(self.activations.shape[-1])

    @property
    def d_in(self) -> int:
        return self._d_in

    def __iter__(self) -> Iterator[ActivationBatch]:
        acts = self.activations
        if self.shuffle:
            perm = torch.randperm(acts.shape[0], device=self.device)
            acts = acts[perm]

        n = acts.shape[0]
        end = (n // self.batch_size) * self.batch_size if self.drop_last else n

        for i in range(0, end, self.batch_size):
            yield ActivationBatch(activations=acts[i : i + self.batch_size])