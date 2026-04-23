from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from .logging_utils import get_logger, kv_to_text
from .types import CandidateRecord

logger = get_logger(__name__)


def _github_search(query: str, token: str | None, per_page: int) -> list[dict[str, Any]]:
    url = f"https://api.github.com/search/repositories?q={quote_plus(query)}&sort=updated&order=desc&per_page={per_page}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "ocr-leaderboard-agent",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url=url, headers=headers, method="GET")

    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("items", [])


def discover_candidates(config: dict[str, Any]) -> list[CandidateRecord]:
    token = os.getenv(config.get("token_env", "GITHUB_TOKEN"))
    keywords: list[str] = config.get("keywords", ["OCR"])
    min_stars: int = int(config.get("min_stars", 0))
    per_keyword: int = int(config.get("max_per_keyword", 10))
    days_back: int = int(config.get("search_window_days", 7))
    logger.info(
        "discovery started %s",
        kv_to_text(
            keywords=len(keywords),
            min_stars=min_stars,
            per_keyword=per_keyword,
            days_back=days_back,
            has_token=bool(token),
        ),
    )

    dedup: dict[str, CandidateRecord] = {}
    pushed_after = (datetime.now(timezone.utc) - timedelta(days=days_back)).date().isoformat()
    for keyword in keywords:
        query = f"{keyword} stars:>={min_stars} pushed:>={pushed_after}"
        try:
            items = _github_search(query=query, token=token, per_page=per_keyword)
        except Exception as exc:
            logger.warning("discovery keyword failed %s", kv_to_text(keyword=keyword, error=type(exc).__name__))
            continue

        logger.info("discovery keyword result %s", kv_to_text(keyword=keyword, count=len(items)))
        for item in items:
            full_name = item.get("full_name")
            if not full_name or full_name in dedup:
                continue

            license_obj = item.get("license") or {}
            record = CandidateRecord(
                repo_id=full_name.lower(),
                repo_url=item.get("html_url", ""),
                owner=(item.get("owner") or {}).get("login", ""),
                name=item.get("name", ""),
                updated_at=item.get("updated_at", datetime.now(timezone.utc).isoformat()),
                stars=int(item.get("stargazers_count", 0)),
                default_branch=item.get("default_branch", "main"),
                license_spdx=license_obj.get("spdx_id"),
                summary=item.get("description") or "",
            )
            dedup[full_name] = record

    if dedup:
        logger.info("discovery finished via github %s", kv_to_text(candidates=len(dedup)))
        return list(dedup.values())

    # Fallback seed so the MVP pipeline can run locally.
    seeds = config.get("seed_repos", [])
    records: list[CandidateRecord] = []
    now = datetime.now(timezone.utc).isoformat()
    for seed in seeds:
        repo_url = seed.get("repo_url", "")
        model_id = seed.get("model_id", "seed/model")
        owner, _, name = model_id.partition("/")
        records.append(
            CandidateRecord(
                repo_id=model_id.lower(),
                repo_url=repo_url,
                owner=owner or "seed",
                name=name or model_id,
                updated_at=now,
                stars=int(seed.get("stars", 0)),
                default_branch=seed.get("ref", "main"),
                license_spdx=seed.get("license_spdx"),
                summary=seed.get("summary", "seed candidate"),
            )
        )
    logger.info("discovery fallback seeds used %s", kv_to_text(candidates=len(records)))
    return records

