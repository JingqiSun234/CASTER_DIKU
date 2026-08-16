""














from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


QWEN25_EMBEDDING_PROFILE = (
    "qwen2.5_7b_instruct_hidden_last_nonpadding_fp32_l2_v1"
)
QWEN25_EMBEDDING_MANIFEST_SCHEMA = "caster_qwen25_embedding_manifest_v1"

_CUDA_DEVICE_RE = re.compile(r"^cuda(?::([0-9]+))?$")
_RUNTIME_ASSET_SUFFIXES = {
    ".bin",
    ".json",
    ".model",
    ".safetensors",
    ".txt",
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_assets(checkpoint_path: Path) -> list[Path]:
    assets = sorted(
        (
            path
            for path in checkpoint_path.rglob("*")
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in _RUNTIME_ASSET_SUFFIXES
        ),
        key=lambda path: path.relative_to(checkpoint_path).as_posix(),
    )
    if not assets:
        raise ValueError(
            f"local checkpoint contains no recognized runtime assets: "
            f"{checkpoint_path}"
        )
    return assets


def fingerprint_local_checkpoint(checkpoint_path: str | Path) -> dict[str, object]:
    ""







    root = Path(checkpoint_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Qwen embedding checkpoint must be a local directory: {root}"
        )
    records: list[dict[str, object]] = []
    for path in _runtime_assets(root):
        records.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    return {
        "fingerprint_schema": "caster_local_hf_checkpoint_content_v1",
        "resolved_local_path": str(root),
        "asset_count": len(records),
        "assets": records,
        "checkpoint_sha256": _canonical_sha256(records),
    }


def _array_sha256(vectors: np.ndarray) -> str:
    canonical = np.ascontiguousarray(vectors, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(canonical.shape), "dtype": "float32"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _version_or_unknown(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


@dataclass(frozen=True)
class Qwen25EmbeddingConfig:
    ""

    checkpoint_path: str | Path
    device: str = "cuda"
    max_length: int = 32768
    batch_size: int = 1
    padding_side: str = "left"

    def __post_init__(self) -> None:
        if not _CUDA_DEVICE_RE.fullmatch(str(self.device)):
            raise ValueError(
                "Qwen2.5 embedding requires device='cuda' or 'cuda:<index>'; "
                "CPU execution and fallback are not supported"
            )
        if int(self.max_length) <= 0:
            raise ValueError("max_length must be positive")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if self.padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be 'left' or 'right'")


@dataclass(frozen=True)
class Qwen25EmbeddingBatch:
    ""

    vectors: np.ndarray
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.vectors)
        if matrix.ndim != 2:
            raise ValueError("embedding vectors must be a two-dimensional matrix")
        if matrix.dtype != np.float32:
            raise ValueError("embedding vectors must be FP32")


class Qwen25HiddenStateEmbedder:
    ""







    def __init__(
        self,
        *,
        config: Qwen25EmbeddingConfig,
        tokenizer: Any,
        model: Any,
        torch_module: Any,
        checkpoint_identity: Mapping[str, object],
        model_metadata: Mapping[str, object],
    ) -> None:
        self.config = config
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module
        self._checkpoint_identity = dict(checkpoint_identity)
        self._model_metadata = dict(model_metadata)

    @classmethod
    def from_local_checkpoint(
        cls,
        config: Qwen25EmbeddingConfig,
        *,
        _torch_module: Any | None = None,
        _auto_tokenizer: Any | None = None,
        _auto_model: Any | None = None,
    ) -> "Qwen25HiddenStateEmbedder":
        ""

        checkpoint_path = Path(config.checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(
                "Qwen embedding checkpoint must already exist as a local "
                f"directory: {checkpoint_path}"
            )

        if _torch_module is None:
            try:
                import torch as torch_module
            except ImportError as exc:                                           
                raise RuntimeError(
                    "PyTorch is required for the Qwen2.5 embedding backend"
                ) from exc
        else:
            torch_module = _torch_module

        if not bool(torch_module.cuda.is_available()):
            raise RuntimeError(
                "Qwen2.5 embedding requires CUDA; no CPU fallback is permitted"
            )

        match = _CUDA_DEVICE_RE.fullmatch(str(config.device))
        assert match is not None                                      
        requested_index = match.group(1)
        if requested_index is not None:
            device_count = int(torch_module.cuda.device_count())
            if int(requested_index) >= device_count:
                raise RuntimeError(
                    f"requested CUDA device {config.device} is unavailable "
                    f"(device_count={device_count}); no fallback is permitted"
                )

        if _auto_tokenizer is None or _auto_model is None:
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:                                           
                raise RuntimeError(
                    "Transformers is required for the Qwen2.5 embedding backend"
                ) from exc
            auto_tokenizer = AutoTokenizer if _auto_tokenizer is None else _auto_tokenizer
            auto_model = AutoModel if _auto_model is None else _auto_model
        else:
            auto_tokenizer = _auto_tokenizer
            auto_model = _auto_model

        checkpoint_identity = fingerprint_local_checkpoint(checkpoint_path)
        tokenizer = auto_tokenizer.from_pretrained(
            str(checkpoint_path),
            local_files_only=True,
            trust_remote_code=False,
            padding_side=config.padding_side,
        )
        if getattr(tokenizer, "pad_token_id", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is None:
                raise RuntimeError(
                    "local tokenizer defines neither a pad token nor an EOS token"
                )
            tokenizer.pad_token = eos_token

        model = auto_model.from_pretrained(
            str(checkpoint_path),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch_module.bfloat16,
            low_cpu_mem_usage=True,
        )
        model = model.to(config.device)
        model.eval()
        model_config = getattr(model, "config", None)
        model_metadata = {
            "model_type": str(getattr(model_config, "model_type", "")),
            "hidden_size": int(getattr(model_config, "hidden_size", 0) or 0),
            "num_hidden_layers": int(
                getattr(model_config, "num_hidden_layers", 0) or 0
            ),
            "torch_version": str(getattr(torch_module, "__version__", "unknown")),
            "transformers_version": _version_or_unknown("transformers"),
        }
        return cls(
            config=config,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch_module,
            checkpoint_identity=checkpoint_identity,
            model_metadata=model_metadata,
        )

    def embed_texts(self, texts: Sequence[str]) -> Qwen25EmbeddingBatch:
        ""

        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not one string")
        normalized_texts = [str(text) for text in texts]
        if not normalized_texts:
            raise ValueError("texts must not be empty")
        if any(not text.strip() for text in normalized_texts):
            raise ValueError("embedding input texts must be non-empty")

        batches: list[np.ndarray] = []
        for start in range(0, len(normalized_texts), self.config.batch_size):
            batch_texts = normalized_texts[start : start + self.config.batch_size]
            encoded = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=int(self.config.max_length),
                add_special_tokens=True,
                return_tensors="pt",
            )
            if "attention_mask" not in encoded:
                raise RuntimeError("tokenizer output is missing attention_mask")
            encoded = encoded.to(self.config.device)
            with self._torch.inference_mode():
                outputs = self._model(
                    **encoded,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
            hidden_states = getattr(outputs, "hidden_states", None)
            if not hidden_states:
                raise RuntimeError("Qwen model did not return hidden states")
            last_hidden = hidden_states[-1]
            pooled = self._last_nonpadding_pool(
                last_hidden,
                encoded["attention_mask"],
            )
            pooled_fp32 = pooled.to(dtype=self._torch.float32)
            norms = self._torch.linalg.vector_norm(
                pooled_fp32,
                ord=2,
                dim=1,
                keepdim=True,
            )
            if bool((~self._torch.isfinite(norms)).any().item()):
                raise RuntimeError("non-finite norm in Qwen embeddings")
            if bool((norms <= 0).any().item()):
                raise RuntimeError("zero-norm Qwen embedding cannot be normalized")
            normalized = pooled_fp32 / norms
            array = (
                normalized.detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            batches.append(array)

        vectors = np.ascontiguousarray(np.concatenate(batches, axis=0), dtype=np.float32)
        text_hashes = [
            hashlib.sha256(text.encode("utf-8")).hexdigest()
            for text in normalized_texts
        ]
        manifest: dict[str, object] = {
            "schema": QWEN25_EMBEDDING_MANIFEST_SCHEMA,
            "backend_profile": QWEN25_EMBEDDING_PROFILE,
            "opt_in_backend": True,
            "alternate_hash_backend_modified": False,
            "checkpoint": self._checkpoint_identity,
            "model": self._model_metadata,
            "inference": {
                "device": str(self.config.device),
                "cuda_required": True,
                "cpu_fallback": False,
                "remote_checkpoint_fallback": False,
                "local_files_only": True,
                "trust_remote_code": False,
                "model_dtype": "bfloat16",
                "output_dtype": "float32",
                "pooling": "last_nonpadding_token",
                "normalization": "l2",
                "max_length": int(self.config.max_length),
                "batch_size": int(self.config.batch_size),
                "padding_side": str(self.config.padding_side),
                "truncation": True,
                "add_special_tokens": True,
                "chat_template_applied_by_backend": False,
                "input_contract": "caller_rendered_utf8_text_v1",
            },
            "request": {
                "text_count": len(normalized_texts),
                "ordered_text_sha256": text_hashes,
                "input_sha256": _canonical_sha256(text_hashes),
            },
            "output": {
                "shape": list(vectors.shape),
                "embedding_dimension": int(vectors.shape[1]),
                "dtype": "float32",
                "vectors_sha256": _array_sha256(vectors),
            },
        }
        config_contract = {
            "backend_profile": QWEN25_EMBEDDING_PROFILE,
            "checkpoint_sha256": self._checkpoint_identity["checkpoint_sha256"],
            "inference": manifest["inference"],
        }
        manifest["embedding_contract_sha256"] = _canonical_sha256(config_contract)
        manifest["manifest_sha256"] = _canonical_sha256(manifest)
        return Qwen25EmbeddingBatch(vectors=vectors, manifest=manifest)

    def _last_nonpadding_pool(self, last_hidden: Any, attention_mask: Any) -> Any:
        if int(last_hidden.ndim) != 3:
            raise RuntimeError("last hidden state must have shape [batch, tokens, hidden]")
        if int(attention_mask.ndim) != 2:
            raise RuntimeError("attention mask must have shape [batch, tokens]")
        if tuple(last_hidden.shape[:2]) != tuple(attention_mask.shape):
            raise RuntimeError("hidden state and attention mask shapes do not align")
        mask = attention_mask.to(device=last_hidden.device)
        positions = self._torch.arange(
            int(mask.shape[1]),
            device=last_hidden.device,
        ).unsqueeze(0).expand_as(mask)
        last_indices = positions.masked_fill(mask == 0, -1).max(dim=1).values
        if bool((last_indices < 0).any().item()):
            raise RuntimeError("at least one embedding input contains no tokens")
        rows = self._torch.arange(
            int(last_hidden.shape[0]),
            device=last_hidden.device,
        )
        return last_hidden[rows, last_indices]


def embed_registry_qwen25(
    registry: pd.DataFrame,
    embedder: Qwen25HiddenStateEmbedder,
) -> tuple[pd.DataFrame, dict[str, object]]:
    ""

    if "model_id" not in registry.columns:
        raise ValueError("registry is missing model_id")
    if registry["model_id"].astype(str).duplicated().any():
        raise ValueError("registry model_id values must be unique")
    texts: list[str] = []
    for _, row in registry.iterrows():
        scientific_description = str(row.get("description", "") or "").strip()
        if not scientific_description:
            raise ValueError("candidate embedding text must be non-empty")
        texts.append(scientific_description)

    batch = embedder.embed_texts(texts)
    embedding_columns = {
        f"emb_{index:04d}": batch.vectors[:, index]
        for index in range(batch.vectors.shape[1])
    }
    frame = pd.DataFrame(
        {
            "model_id": registry["model_id"].astype(str).tolist(),
            **embedding_columns,
        }
    )
    manifest = dict(batch.manifest)
    manifest["collection"] = {
        "kind": "candidate_registry",
        "candidate_count": len(frame),
        "ordered_model_ids_sha256": _canonical_sha256(
            frame["model_id"].astype(str).tolist()
        ),
        "registry_embeddings_sha256": _array_sha256(batch.vectors),
    }
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    return frame, manifest
