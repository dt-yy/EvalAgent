from __future__ import annotations

from typing import Any

from .ai_client import classify_candidate_with_ai
from .logging_utils import get_logger, kv_to_text
from .types import CandidateRecord, EvalJob

logger = get_logger(__name__)


def _is_likely_ocr_candidate(candidate: CandidateRecord) -> bool:
    text = f"{candidate.name} {candidate.summary}".lower()
    keywords = ["ocr", "document", "layout", "table", "recognition", "parsing"]
    return any(k in text for k in keywords)


def _normalize_repo_key(value: str) -> str:
    return value.strip().lower().rstrip("/")


def _is_readme_verified(candidate: CandidateRecord, verified_repo_keys: set[str]) -> bool:
    repo_id_key = _normalize_repo_key(candidate.repo_id)
    owner_name_key = _normalize_repo_key(f"{candidate.owner}/{candidate.name}")
    repo_url_key = _normalize_repo_key(candidate.repo_url)
    return (
        repo_id_key in verified_repo_keys
        or owner_name_key in verified_repo_keys
        or repo_url_key in verified_repo_keys
    )


def filter_candidates(
    candidates: list[CandidateRecord],
    config: dict[str, Any],
) -> list[EvalJob]:
    allow_licenses = set(config.get("allow_licenses", []))
    enforce_license = bool(config.get("enforce_license", False))
    min_stars = int(config.get("min_stars", 0))
    enforce_readme_gate = bool(config.get("enforce_readme_gate", False))
    ai_min_confidence = float(config.get("ai_repo_filter_min_confidence", 0.65))
    ai_hard_skip = bool(config.get("ai_repo_filter_hard_skip", False))
    verified_repo_keys = {
        _normalize_repo_key(item)
        for item in config.get("readme_verified_repos", [])
        if isinstance(item, str)
    }
    logger.info(
        "filter started %s",
        kv_to_text(
            candidates=len(candidates),
            min_stars=min_stars,
            enforce_license=enforce_license,
            enforce_readme_gate=enforce_readme_gate,
        ),
    )

    jobs: list[EvalJob] = []
    for idx, candidate in enumerate(candidates, start=1):
        status = "ready"
        notes: list[str] = []

        if candidate.stars < min_stars:
            status = "skipped"
            notes.append(f"stars<{min_stars}")

        if status == "ready" and not _is_likely_ocr_candidate(candidate):
            status = "need_manual"
            notes.append("weak_ocr_signal")

        if status == "ready" and enforce_license:
            if not candidate.license_spdx or candidate.license_spdx not in allow_licenses:
                status = "need_manual"
                notes.append("license_check_required")

        # Skill constraint: ideally read README before install/infer.
        if status == "ready" and enforce_readme_gate:
            if not _is_readme_verified(candidate=candidate, verified_repo_keys=verified_repo_keys):
                status = "need_manual"
                notes.append("readme_not_confirmed")

        ai_result = classify_candidate_with_ai(candidate=candidate, config=config)
        if ai_result is not None and status != "skipped":
            confidence = float(ai_result.get("confidence", 0.0))
            notes.append(f"ai_confidence={confidence:.2f}")
            reason = str(ai_result.get("reason", ""))
            if reason:
                notes.append(f"ai_reason={reason}")

            if confidence >= ai_min_confidence:
                if not ai_result.get("is_ocr_related", False):
                    status = "skipped" if ai_hard_skip else "need_manual"
                    notes.append("ai_not_ocr_related")
                elif not ai_result.get("is_runnable", False):
                    status = "need_manual"
                    notes.append("ai_not_runnable")
                elif ai_result.get("should_evaluate", False):
                    if status == "need_manual" and "weak_ocr_signal" in notes:
                        status = "ready"
                        notes.append("ai_promoted_to_ready")
                else:
                    status = "need_manual"
                    notes.append("ai_hold_for_manual")

        model_id = f"{candidate.owner}/{candidate.name}".strip("/")
        jobs.append(
            EvalJob(
                job_id=f"job_{idx:04d}",
                model_id=model_id,
                repo_url=candidate.repo_url,
                ref=candidate.default_branch,
                status=status,
                priority=max(1, 100 - candidate.stars),
                notes=notes,
            )
        )

    jobs.sort(key=lambda x: x.priority)
    logger.info(
        "filter finished %s",
        kv_to_text(
            jobs_total=len(jobs),
            ready=sum(1 for j in jobs if j.status == "ready"),
            need_manual=sum(1 for j in jobs if j.status == "need_manual"),
            skipped=sum(1 for j in jobs if j.status == "skipped"),
        ),
    )
    return jobs

