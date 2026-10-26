"""llama-cpp-pythonとGGUFモデルによる、4つのプロンプト条件での抽出。"""

from __future__ import annotations

import gc
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import CONDITIONS
from .io_utils import load_patient_ids, read_candidates, write_json
from .normalize import (
    SemanticOutputError,
    StructuredOutputError,
    parse_strict_response,
    validate_semantic_response,
)
from .prompts import (
    OUTPUT_SCHEMA,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    build_extraction_messages,
    build_reasoning_messages,
    prompt_sha256,
    schema_sha256,
)


# 生成設定。全モデル・全条件で共通。
GENERATION_CONFIG = {"temperature": 0.0, "top_p": 1.0, "top_k": 40, "repeat_penalty": 1.05}
MAX_TOKENS = 512
REASONING_MAX_TOKENS = 512

QWEN35_ARCHITECTURE = "qwen35"
NON_THINKING_PREFILL = "<think>\n\n</think>\n\n"


@dataclass(frozen=True)
class ModelSettings:
    model_path: Path
    model_id: str
    n_ctx: int = 8192
    n_threads: int = 12
    n_gpu_layers: int = -1
    n_batch: int = 512
    n_ubatch: int = 512
    flash_attention: bool = True
    seed: int = 42


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _token_text(model: Any, token_id: int) -> str:
    if token_id < 0:
        return ""
    value = model.detokenize([token_id])
    return value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else str(value)


def _disable_native_thinking(model: Any) -> str:
    """Qwen3.5では、chat templateにenable_thinking=falseを渡してthinkingモードを無効にする。"""

    metadata = getattr(model, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    architecture = str(metadata.get("general.architecture") or "").strip().lower()
    model_name = str(metadata.get("general.name") or "").strip().lower()
    if architecture != QWEN35_ARCHITECTURE and "qwen3.5" not in model_name:
        return "not_applicable"

    template = metadata.get("tokenizer.chat_template")
    if not isinstance(template, str) or "enable_thinking" not in template:
        raise RuntimeError("chat templateでenable_thinkingを指定できないため、thinkingモードを無効にできません")

    from llama_cpp import llama_chat_format
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    eos_token_id = int(model.token_eos())
    bos_token_id = int(model.token_bos())
    base_formatter = Jinja2ChatFormatter(
        template=template,
        eos_token=_token_text(model, eos_token_id),
        bos_token=_token_text(model, bos_token_id),
        stop_token_ids=[eos_token_id] if eos_token_id >= 0 else None,
    )

    def nonthinking_formatter(*, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        kwargs.pop("enable_thinking", None)
        return base_formatter(messages=messages, enable_thinking=False, **kwargs)

    probe = nonthinking_formatter(messages=[{"role": "user", "content": "probe"}])
    if not probe.prompt.endswith(NON_THINKING_PREFILL):
        raise RuntimeError("chat templateが空のthinkブロックを出力しません")

    base_chat_handler = (
        getattr(model, "chat_handler", None)
        or getattr(model, "_chat_handlers", {}).get(getattr(model, "chat_format", None))
        or llama_chat_format.get_chat_completion_handler(getattr(model, "chat_format", None))
    )
    if base_chat_handler is None:
        raise RuntimeError("chat handlerを取得できません")

    def nonthinking_chat_handler(*args: Any, **kwargs: Any) -> Any:
        kwargs["enable_thinking"] = False
        return base_chat_handler(*args, **kwargs)

    model.chat_handler = nonthinking_chat_handler
    model._family_history_chat_formatter = nonthinking_formatter
    return "disabled_enable_thinking_false"


def load_model(settings: ModelSettings) -> Any:
    from llama_cpp import Llama

    model = Llama(
        model_path=str(settings.model_path),
        n_ctx=settings.n_ctx,
        n_threads=settings.n_threads,
        n_threads_batch=settings.n_threads,
        n_gpu_layers=settings.n_gpu_layers,
        n_batch=settings.n_batch,
        n_ubatch=settings.n_ubatch,
        flash_attn=settings.flash_attention,
        seed=settings.seed,
        verbose=False,
    )
    model._family_history_native_thinking_control = _disable_native_thinking(model)
    return model


def extract_content(completion: dict[str, Any]) -> str:
    message = completion["choices"][0]["message"]
    content = message.get("content") or message.get("reasoning_content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(item if isinstance(item, str) else item["text"] for item in content
                         if isinstance(item, str) or (isinstance(item, dict) and isinstance(item.get("text"), str)))
    return str(content or "")


def usage_of(completion: dict[str, Any]) -> dict[str, int]:
    usage = completion.get("usage") or {}
    return {key: int(usage.get(key, 0) or 0) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}


def _chat_completion(model: Any, messages: list[dict[str, str]], *, max_tokens: int,
                     structured: bool) -> tuple[str, dict[str, int]]:
    options: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens, **GENERATION_CONFIG}
    if structured:
        options["response_format"] = {"type": "json_object", "schema": OUTPUT_SCHEMA}
    completion = model.create_chat_completion(**options)
    return extract_content(completion), usage_of(completion)


def truncate_text(model: Any, text: str, token_budget: int) -> tuple[str, bool]:
    """入力の予算を超える本文は末尾を切り捨て、切り捨てたことを記録する。"""

    tokens = model.tokenize(text.encode("utf-8"))
    if len(tokens) <= token_budget:
        return text, False
    truncated = model.detokenize(tokens[:token_budget]).decode("utf-8", errors="ignore")
    return truncated + "\n（入力上限により以下省略）", True


def run_condition(model: Any, text: str, condition: str) -> tuple[str, dict[str, int]]:
    """1件のテキストを1つのプロンプト条件で処理する。CoT条件では分析段階を先に行う。"""

    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    reasoning_note: str | None = None
    if condition in {"zero_shot_cot", "few_shot_cot"}:
        reasoning_note, usage = _chat_completion(model, build_reasoning_messages(text, condition),
                                                 max_tokens=REASONING_MAX_TOKENS, structured=False)
        usage_total = {key: usage_total[key] + usage[key] for key in usage_total}
    raw_response, usage = _chat_completion(
        model, build_extraction_messages(text, condition, reasoning_note=reasoning_note),
        max_tokens=MAX_TOKENS, structured=True)
    usage_total = {key: usage_total[key] + usage[key] for key in usage_total}
    return raw_response, usage_total


def run_inference(candidate_dir: Path, patient_list: Path, output_root: Path, settings: ModelSettings,
                  *, conditions: tuple[str, ...] = CONDITIONS) -> None:
    """選択した患者の候補テキストを、指定したプロンプト条件ですべて処理する。

    出力は「<output_root>/<model_id>/<条件>/responses/<患者ID>_withResponse.json」に、
    日付ごとの状態（success、json_invalid、semantic_invalid）、出力、所要時間を保存する。
    """

    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"未対応のプロンプト条件: {', '.join(sorted(unknown))}")
    model_root = output_root / settings.model_id
    if model_root.exists():
        raise FileExistsError(model_root)
    token_budget = max(128, settings.n_ctx - max(MAX_TOKENS, REASONING_MAX_TOKENS) - 2048)
    generation_config = {**GENERATION_CONFIG, "seed": settings.seed, "max_tokens": MAX_TOKENS,
                         "reasoning_max_tokens": REASONING_MAX_TOKENS, "source_token_budget": token_budget}
    patient_ids = sorted(load_patient_ids(patient_list))
    model = load_model(settings)
    thinking = model._family_history_native_thinking_control
    started = time.monotonic()
    try:
        for index, patient_id in enumerate(patient_ids, start=1):
            outputs: dict[str, dict[str, Any]] = {condition: {} for condition in conditions}
            for date, payload in read_candidates(candidate_dir / f"{patient_id}.json").items():
                source_text = payload["fam_text"].strip()
                if not source_text:
                    continue
                inference_text, truncated = truncate_text(model, source_text, token_budget)
                for condition in conditions:
                    generation_started = time.monotonic()
                    raw_response, usage = run_condition(model, inference_text, condition)
                    meta = {
                        "model_id": settings.model_id, "condition": condition, "timestamp_utc": _timestamp(),
                        "latency_sec": round(time.monotonic() - generation_started, 3),
                        "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
                        "source_truncated": truncated, "native_thinking_control": thinking,
                        "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_sha256(),
                        "structured_output_schema_version": SCHEMA_VERSION,
                        "structured_output_schema_sha256": schema_sha256(),
                        "generation_config": generation_config, "usage": usage,
                    }
                    try:
                        prediction = parse_strict_response(raw_response)
                    except StructuredOutputError as exc:
                        outputs[condition][date] = {"status": "json_invalid", "raw_response": raw_response,
                                                    "parse_error": str(exc), "_meta": meta}
                        continue
                    try:
                        validate_semantic_response(prediction, inference_text)
                    except SemanticOutputError as exc:
                        outputs[condition][date] = {"status": "semantic_invalid", "raw_response": raw_response,
                                                    "prediction": prediction, "semantic_error": str(exc),
                                                    "_meta": meta}
                        continue
                    outputs[condition][date] = {"status": "success", "prediction": prediction, "_meta": meta}
            for condition in conditions:
                write_json(model_root / condition / "responses" / f"{patient_id}_withResponse.json",
                           outputs[condition])
            print(f"[infer] {settings.model_id} {index}/{len(patient_ids)}", flush=True)
    finally:
        del model
        gc.collect()
    write_json(model_root / "run_metadata.json", {
        "model_id": settings.model_id, "model_file": settings.model_path.name,
        "conditions": list(conditions), "native_thinking_control": thinking,
        "settings": {"n_ctx": settings.n_ctx, "n_threads": settings.n_threads, "n_gpu_layers": settings.n_gpu_layers,
                     "n_batch": settings.n_batch, "n_ubatch": settings.n_ubatch,
                     "flash_attention": settings.flash_attention, "seed": settings.seed},
        "generation_config": generation_config, "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_sha256(),
        "schema_sha256": schema_sha256(), "elapsed_seconds": round(time.monotonic() - started, 3)})
