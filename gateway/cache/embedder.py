"""Sentence embeddings on CPU for the semantic cache.

Model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions, 22M parameters),
run through ONNX Runtime instead of PyTorch. Same weights and the same
mean-pooling + L2-normalisation that sentence-transformers applies, but the
Docker image stays ~1.5 GB lighter and a single short prompt embeds in a few
milliseconds on one core.

    python -m gateway.cache.embedder download models/all-MiniLM-L6-v2
"""

from __future__ import annotations

import asyncio
import io
import sys
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

DIM = 384
MAX_TOKENS = 256
HF_BASE = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
# Mirror used by fastembed; same ONNX export of the same model.
GCS_TARBALL = "https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz"


class Embedder:
    def __init__(self, model_dir: str | Path, threads: int = 1):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embed")

    def embed(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        out = self.session.run(None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)})[0]
        m = mask[..., None].astype(np.float32)
        pooled = (out * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        return pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)

    async def aembed(self, text: str) -> np.ndarray:
        # ONNX Runtime releases the GIL, so a thread keeps the event loop free.
        return (await asyncio.get_running_loop().run_in_executor(self._pool, self.embed, [text]))[0]


def download(target: str | Path) -> None:
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    try:
        for name, remote in (("model.onnx", "onnx/model.onnx"), ("tokenizer.json", "tokenizer.json")):
            urllib.request.urlretrieve(f"{HF_BASE}/{remote}", target / name)
        print(f"downloaded from Hugging Face into {target}")
        return
    except OSError as exc:
        print(f"Hugging Face download failed ({exc}); trying the fastembed mirror", file=sys.stderr)
    with urllib.request.urlopen(GCS_TARBALL) as resp:
        data = resp.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if name in ("model.onnx", "tokenizer.json") and member.isfile():
                f = tar.extractfile(member)
                assert f is not None
                (target / name).write_bytes(f.read())
    print(f"downloaded from mirror into {target}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "download":
        download(sys.argv[2])
    else:
        print(__doc__)
