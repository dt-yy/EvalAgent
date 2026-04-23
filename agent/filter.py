from __future__ import annotations

from typing import Any

from .types import CandidateRecord, EvalJob


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
    verified_repo_keys = {
        _normalize_repo_key(item)
        for item in config.get("readme_verified_repos", [])
        if isinstance(item, str)
    }

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
    return jobs

