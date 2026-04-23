from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .logging_utils import get_logger, kv_to_text
from .types import CandidateRecord

logger = get_logger(__name__)


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response.")
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        data = json.loads(text[start : end + 1])
        if isinstance(data, dict):
            return data
    raise ValueError("No valid JSON object found in model response.")


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _call_openai_compatible(
    *,
    model: str,
    base_url: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    timeout_sec: int,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    request = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urlopen(request, timeout=timeout_sec) as response:
        raw = json.loads(response.read().decode("utf-8"))
    choices = raw.get("choices") or []
    if not choices:
        raise ValueError("No choices returned by model.")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    return _extract_json_object(content)


def classify_candidate_with_ai(
    *,
    candidate: CandidateRecord,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    enabled = bool(config.get("ai_repo_filter_enabled", False))
    if not enabled:
        return None

    api_key_env = str(config.get("ai_api_key_env", "OPENAI_API_KEY"))
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        logger.debug("ai repo filter skipped: missing api key env %s", api_key_env)
        return None

    model = str(config.get("ai_model", "gpt-4o-mini"))
    base_url = str(config.get("ai_base_url", "https://api.openai.com/v1"))
    timeout_sec = int(config.get("ai_timeout_sec", 30))

    system_prompt = (
        "You are an OCR model triage assistant. "
        "Return STRICT JSON only, no markdown."
    )
    user_prompt = (
        "Decide if this GitHub repository should enter OCR evaluation queue.\n"
        f"repo_id: {candidate.repo_id}\n"
        f"repo_url: {candidate.repo_url}\n"
        f"name: {candidate.name}\n"
        f"owner: {candidate.owner}\n"
        f"summary: {candidate.summary}\n"
        f"stars: {candidate.stars}\n"
        f"license_spdx: {candidate.license_spdx}\n\n"
        "Output schema:\n"
        "{\n"
        '  "is_ocr_related": true|false,\n'
        '  "is_runnable": true|false,\n'
        '  "should_evaluate": true|false,\n'
        '  "confidence": 0.0-1.0,\n'
        '  "reason": "short reason"\n'
        "}"
    )

    try:
        data = _call_openai_compatible(
            model=model,
            base_url=base_url,
            api_key=api_key,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_sec=timeout_sec,
        )
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        logger.warning(
            "ai repo filter failed %s",
            kv_to_text(repo_id=candidate.repo_id, error=type(exc).__name__),
        )
        return None

    return {
        "is_ocr_related": _to_bool(data.get("is_ocr_related"), default=False),
        "is_runnable": _to_bool(data.get("is_runnable"), default=False),
        "should_evaluate": _to_bool(data.get("should_evaluate"), default=False),
        "confidence": _to_float(data.get("confidence"), default=0.0),
        "reason": str(data.get("reason", "")).strip()[:240],
    }


def _parse_github_owner_repo(repo_url: str) -> tuple[str, str] | None:
    try:
        parsed = urlparse(repo_url.strip())
    except ValueError:
        return None
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return None
    owner = parts[0]
    repo = parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not owner or not repo:
        return None
    return owner, repo


def _fetch_text(url: str, timeout_sec: int) -> str | None:
    request = Request(
        url=url,
        method="GET",
        headers={
            "Accept": "text/plain",
            "User-Agent": "ocr-leaderboard-agent",
        },
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            return response.read().decode("utf-8", errors="ignore")
    except (HTTPError, URLError, OSError, TimeoutError):
        return None


def _fetch_repo_readme(repo_url: str, ref: str, timeout_sec: int) -> str | None:
    parsed = _parse_github_owner_repo(repo_url)
    if not parsed:
        return None
    owner, repo = parsed
    refs = [ref, "main", "master"]
    file_names = ["README.md", "README_zh-CN.md", "README.en.md", "readme.md"]
    for current_ref in refs:
        for file_name in file_names:
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/{current_ref}/{file_name}"
            content = _fetch_text(url=url, timeout_sec=timeout_sec)
            if content:
                return content
    return None


def suggest_infer_command_with_ai(
    *,
    repo_url: str,
    ref: str,
    model_id: str,
    current_template: str,
    current_backend: str,
    input_images_dir: str,
    pred_path: str,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    enabled = bool(config.get("ai_infer_planner_enabled", False))
    if not enabled:
        return None

    api_key_env = str(config.get("ai_api_key_env", "OPENAI_API_KEY"))
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        logger.debug("ai infer planner skipped: missing api key env %s", api_key_env)
        return None

    timeout_sec = int(config.get("ai_timeout_sec", 30))
    model = str(config.get("ai_model", "gpt-4o-mini"))
    base_url = str(config.get("ai_base_url", "https://api.openai.com/v1"))
    max_chars = int(config.get("ai_infer_readme_max_chars", 12000))

    readme_text = _fetch_repo_readme(repo_url=repo_url, ref=ref, timeout_sec=timeout_sec)
    if not readme_text:
        logger.warning("ai infer planner skipped: no readme %s", kv_to_text(model_id=model_id, repo_url=repo_url))
        return None
    readme_text = readme_text[:max_chars]

    system_prompt = (
        "You are an inference command planner for OCR model evaluation. "
        "Return STRICT JSON only. Keep commands reproducible and concise."
    )
    user_prompt = (
        "Task: Suggest a better inference command template for this model based on README.\n"
        f"model_id: {model_id}\n"
        f"repo_url: {repo_url}\n"
        f"ref: {ref}\n"
        f"current_template: {current_template}\n"
        f"current_backend: {current_backend}\n"
        f"input_images_dir: {input_images_dir}\n"
        f"pred_path: {pred_path}\n\n"
        "README excerpt:\n"
        f"{readme_text}\n\n"
        "Output schema:\n"
        "{\n"
        '  "infer_command_template": "string with {input_images_dir} and {pred_path}",\n'
        '  "mineru_backend": "pipeline|hybrid|vlm|...",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "reason": "short reason"\n'
        "}"
    )

    try:
        data = _call_openai_compatible(
            model=model,
            base_url=base_url,
            api_key=api_key,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_sec=timeout_sec,
        )
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        logger.warning(
            "ai infer planner failed %s",
            kv_to_text(model_id=model_id, error=type(exc).__name__),
        )
        return None

    infer_template = str(data.get("infer_command_template", "")).strip()
    if "{input_images_dir}" not in infer_template or "{pred_path}" not in infer_template:
        logger.warning("ai infer planner returned invalid template %s", kv_to_text(model_id=model_id))
        return None

    logger.info("ai infer planner generated candidate command %s", kv_to_text(model_id=model_id))
    return {
        "infer_command_template": infer_template,
        "mineru_backend": str(data.get("mineru_backend", current_backend)).strip() or current_backend,
        "confidence": _to_float(data.get("confidence"), default=0.0),
        "reason": str(data.get("reason", "")).strip()[:240],
    }

