from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_QWEN_7B = os.environ.get(
    "CASTER_QWEN_CHECKPOINT", "QWEN_CHECKPOINT_NOT_CONFIGURED"
)
DEFAULT_QWEN_3B = os.environ.get(
    "CASTER_QWEN_3B_CHECKPOINT", "QWEN_3B_CHECKPOINT_NOT_CONFIGURED"
)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMCall:
    stage: str
    prompt: str
    response_text: str
    response_json: dict[str, Any]
    model_path: str
    fallback_used: bool
    fallback_reason: str
    runtime_seconds: float


class ReplayJSONEngine:
    ""

    def __init__(self, responses: list[dict[str, Any]] | dict[str, Any]):
        if isinstance(responses, dict):
            responses = [responses]
        self.responses = list(responses)
        self.calls: list[LLMCall] = []

    def generate_json(self, *, stage: str, system_prompt: str, user_prompt: str, **_: Any) -> LLMCall:
        if not self.responses:
            raise LLMError(f"replay engine has no response for stage={stage}")
        payload = dict(self.responses.pop(0))
        call = LLMCall(
            stage=stage,
            prompt=f"{system_prompt}\n\n{user_prompt}",
            response_text=json.dumps(payload, sort_keys=True),
            response_json=payload,
            model_path="replay",
            fallback_used=False,
            fallback_reason="",
            runtime_seconds=0.0,
        )
        self.calls.append(call)
        return call


class DeterministicNoModelComputeEngine:
    ""

    active_model_path = "deterministic_no_model_compute"
    fallback_used = False
    fallback_reason = ""
    allow_fallback = False
    primary_model_path = "deterministic_no_model_compute"
    fallback_model_path = ""

    def __init__(self) -> None:
        self.calls: list[LLMCall] = []

    def generate_json(
        self,
        *,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        valid_model_ids: list[str] | None = None,
        **_: Any,
    ) -> LLMCall:
        if stage == "PlannerAgent":
            payload = {"plan": "ok", "selection_engine": "deterministic_no_model_compute"}
        else:
            ids = [str(x) for x in (valid_model_ids or []) if str(x)]
            if not ids:
                try:
                    prompt_payload = json.loads(user_prompt)
                    ids = [str(x) for x in prompt_payload.get("valid_model_ids", []) if str(x)]
                except Exception:
                    ids = []
            if not ids:
                raise LLMError("deterministic timing engine received no valid_model_ids")
            payload = {
                "selected_model_id": ids[0],
                "reason": "deterministic timing selection",
                "selection_engine": "deterministic_no_model_compute",
            }
        call = LLMCall(
            stage=stage,
            prompt=f"{system_prompt}\n\n{user_prompt}",
            response_text=json.dumps(payload, sort_keys=True),
            response_json=payload,
            model_path=self.active_model_path,
            fallback_used=False,
            fallback_reason="",
            runtime_seconds=0.0,
        )
        self.calls.append(call)
        return call


class QwenLocalEngine:
    ""

    def __init__(
        self,
        primary_model_path: str = DEFAULT_QWEN_7B,
        fallback_model_path: str = "",
        allow_fallback: bool = False,
        runtime_budget_seconds: float = 900.0,
        max_new_tokens: int = 128,
        required_model_path: str = "",
        require_cuda: bool = False,
    ) -> None:
        self.primary_model_path = str(primary_model_path)
        self.fallback_model_path = str(fallback_model_path)
        self.allow_fallback = bool(allow_fallback)
        self.runtime_budget_seconds = float(runtime_budget_seconds)
        self.max_new_tokens = int(max_new_tokens)
        self.required_model_path = str(required_model_path)
        self.require_cuda = bool(require_cuda)
        self._model = None
        self._tokenizer = None
        self._model_path = ""
        self._fallback_used = False
        self._fallback_reason = ""
        self._started_at = time.time()
        self._load_seconds = 0.0
        self._cuda_available = False
        self._model_device = ""

    @property
    def active_model_path(self) -> str:
        return self._model_path or self.primary_model_path

    @property
    def fallback_used(self) -> bool:
        return self._fallback_used

    @property
    def fallback_reason(self) -> str:
        return self._fallback_reason

    @property
    def load_seconds(self) -> float:
        return float(self._load_seconds)

    @property
    def primary_required(self) -> bool:
        return bool(self.required_model_path)

    @property
    def cuda_available(self) -> bool:
        return bool(self._cuda_available)

    @property
    def model_device(self) -> str:
        return str(self._model_device)

    @staticmethod
    def _same_path(left: str, right: str) -> bool:
        try:
            return Path(left).expanduser().resolve(strict=False) == Path(right).expanduser().resolve(strict=False)
        except Exception:
            return str(left) == str(right)

    @staticmethod
    def _device_is_cuda(device_text: str) -> bool:
        text = str(device_text).strip().lower()
        if not text or text == "unknown":
            return False
        if any(token in text for token in ("cpu", "disk", "meta")):
            return False
        devices = [part.strip() for part in re.split(r"[,;]", text) if part.strip()]
        if not devices:
            return False
        return all(part.startswith("cuda") or part.isdigit() for part in devices)

    @staticmethod
    def _describe_model_device(model: Any) -> str:
        device_map = getattr(model, "hf_device_map", None)
        if isinstance(device_map, dict) and device_map:
            return ",".join(sorted({str(value) for value in device_map.values()}))
        device = getattr(model, "device", None)
        if device is not None:
            return str(device)
        try:
            return str(next(model.parameters()).device)
        except Exception:
            return "unknown"

    def _remaining_budget(self) -> float:
        return self.runtime_budget_seconds - (time.time() - self._started_at)

    def _select_initial_path(self) -> tuple[str, bool, str]:
        primary = Path(self.primary_model_path)
        fallback = Path(self.fallback_model_path) if self.fallback_model_path else None
        if self.required_model_path:
            if not self._same_path(self.primary_model_path, self.required_model_path):
                raise LLMError(
                    "formal Qwen run requires primary model path "
                    f"{self.required_model_path}; got {self.primary_model_path}"
                )
            if self.allow_fallback or self.fallback_model_path:
                raise LLMError("formal Qwen run requires fallback disabled and no fallback model path")
        if not primary.exists():
            if self.allow_fallback and fallback is not None and fallback.exists():
                return str(fallback), True, f"primary_missing:{primary}"
            raise LLMError(f"primary model path missing: {primary}; fallback disabled")
        try:
            import torch

            self._cuda_available = bool(torch.cuda.is_available())
            no_cuda = not self._cuda_available
        except Exception:
            self._cuda_available = False
            no_cuda = True
        if self.require_cuda and no_cuda:
            raise LLMError(f"formal Qwen run requires CUDA for {primary}")
        if "7B" in primary.name and no_cuda:
            if self.allow_fallback and fallback is not None and fallback.exists():
                return str(fallback), True, "primary_7b_no_cuda_or_runtime_budget"
            raise LLMError(f"primary 7B model requires CUDA for formal run; fallback disabled: {primary}")
        if self._remaining_budget() <= 0:
            if self.allow_fallback and fallback is not None and fallback.exists():
                return str(fallback), True, "primary_runtime_budget_exhausted_before_load"
            raise LLMError("primary Qwen runtime budget exhausted before load; fallback disabled")
        return str(primary), False, ""

    def _load(self) -> None:
        if self._model is not None:
            return
        started = time.time()
        path, fallback_used, fallback_reason = self._select_initial_path()
        try:
            self._load_path(path)
            if self.required_model_path and not self._same_path(path, self.required_model_path):
                raise LLMError(f"formal Qwen run loaded non-required model path: {path}")
            if self.require_cuda and not self._device_is_cuda(self._model_device):
                raise LLMError(f"formal Qwen run requires CUDA-loaded model; observed device={self._model_device}")
            self._model_path = path
            self._fallback_used = fallback_used
            self._fallback_reason = fallback_reason
            self._load_seconds += round(time.time() - started, 6)
        except Exception as exc:
            fallback = Path(self.fallback_model_path) if self.fallback_model_path else None
            if (
                not self.required_model_path
                and self.allow_fallback
                and fallback is not None
                and path != str(fallback)
                and fallback.exists()
            ):
                self._load_path(str(fallback))
                self._model_path = str(fallback)
                self._fallback_used = True
                self._fallback_reason = f"primary_load_failed:{type(exc).__name__}:{exc}"
                self._load_seconds += round(time.time() - started, 6)
                return
            raise LLMError(f"Qwen model load failed for {path}: {type(exc).__name__}: {exc}") from exc

    def warm_load(self) -> float:
        ""






        before = float(self._load_seconds)
        self._load()
        return round(float(self._load_seconds) - before, 6)

    def _load_path(self, model_path: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        kwargs: dict[str, Any] = {
            "local_files_only": True,
                                                                        
                                                                           
            "trust_remote_code": False,
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if not torch.cuda.is_available():
            model = model.to("cpu")
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._cuda_available = bool(torch.cuda.is_available())
        self._model_device = self._describe_model_device(model)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise
            payload = json.loads(text[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("LLM JSON response must be an object")
        return payload

    @staticmethod
    def _recover_selector_json(
        text: str,
        *,
        valid_model_ids: list[str] | None = None,
        user_prompt: str = "",
    ) -> dict[str, Any] | None:
        ""








        ids = list(valid_model_ids or [])
        if not ids and user_prompt:
            try:
                prompt_payload = json.loads(user_prompt)
                prompt_ids = prompt_payload.get("valid_model_ids", [])
                if isinstance(prompt_ids, list):
                    ids = [str(model_id) for model_id in prompt_ids]
            except Exception:
                ids = []
        if not ids:
            return None
        matches = []
        for model_id in ids:
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(model_id)}(?![A-Za-z0-9_])"
            if re.search(pattern, text):
                matches.append(model_id)
        if len(matches) != 1:
            return None
        return {
            "selected_model_id": matches[0],
            "reason": "recovered_from_single_exact_model_id_mention",
            "json_recovered": True,
        }

    @staticmethod
    def _json_chat_messages(
        *,
        system_prompt: str,
        user_prompt: str,
        error: str = "",
    ) -> list[dict[str, str]]:
        ""







        instruction = "Return one compact valid JSON object only. Do not include markdown."
        if error:
            instruction += (
                " Your previous output was invalid. Regenerate a shorter JSON object that "
                f"satisfies the requested response schema. Error: {error}"
            )
        return [
            {"role": "system", "content": str(system_prompt)},
            {"role": "user", "content": f"{user_prompt}\n\n{instruction}"},
        ]

    def generate_json(
        self,
        *,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        retries: int = 2,
        valid_model_ids: list[str] | None = None,
        **_: Any,
    ) -> LLMCall:
        self._load()
        assert self._model is not None
        assert self._tokenizer is not None
        error = ""
        started = time.time()
        for attempt in range(int(retries) + 1):
            messages = self._json_chat_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                error=error,
            )
            full_prompt = "\n\n".join(
                f"[{message['role']}] {message['content']}" for message in messages
            )
            text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._tokenizer([text], return_tensors="pt")
            device = getattr(self._model, "device", None)
            if device is not None:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            generated = outputs[0][inputs["input_ids"].shape[-1] :]
            response_text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
            try:
                payload = self._extract_json(response_text)
                runtime = round(time.time() - started, 6)
                if self._remaining_budget() < 0 and not self._fallback_used:
                    raise LLMError("primary Qwen completed after runtime budget; fallback disabled")
                return LLMCall(
                    stage=stage,
                    prompt=full_prompt,
                    response_text=response_text,
                    response_json=payload,
                    model_path=self.active_model_path,
                    fallback_used=self.fallback_used,
                    fallback_reason=self.fallback_reason,
                    runtime_seconds=runtime,
                )
            except Exception as exc:
                recovered = self._recover_selector_json(
                    response_text,
                    valid_model_ids=valid_model_ids,
                    user_prompt=user_prompt,
                )
                if recovered is not None:
                    runtime = round(time.time() - started, 6)
                    return LLMCall(
                        stage=stage,
                        prompt=full_prompt,
                        response_text=response_text,
                        response_json=recovered,
                        model_path=self.active_model_path,
                        fallback_used=self.fallback_used,
                        fallback_reason=self.fallback_reason,
                        runtime_seconds=runtime,
                    )
                error = f"{type(exc).__name__}: {exc}; raw={response_text[:200]}"
                if attempt >= int(retries):
                    raise LLMError(f"invalid JSON from Qwen at stage={stage}: {error}") from exc
        raise LLMError(f"invalid JSON from Qwen at stage={stage}: {error}")
