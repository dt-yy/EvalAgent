from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import EvalResult, to_dict


def _load_existing_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    entries = payload.get("entries", [])
    return entries if isinstance(entries, list) else []


def update_leaderboard(
    results: list[EvalResult],
    config: dict[str, Any],
) -> dict[str, Any]:
    leaderboard_dir = Path(config.get("leaderboard_dir", "./leaderboard")).resolve()
    benchmark = config.get("benchmark", "OmniDocBench")
    sort_by = config.get("sort_by", "overall_score")
    json_path = leaderboard_dir / "leaderboard.json"
    md_path = leaderboard_dir / "leaderboard.md"

    leaderboard_dir.mkdir(parents=True, exist_ok=True)
    by_model: dict[str, dict[str, Any]] = {}

    for item in _load_existing_entries(json_path):
        model_id = item.get("model_id")
        if isinstance(model_id, str):
            by_model[model_id] = item

    for result in results:
        entry = {
            "model_id": result.model_id,
            "repo_url": result.repo_url,
            "benchmark": result.benchmark,
            "overall_score": result.overall_score,
            "metrics": result.metrics,
            "run_id": result.run_id,
            "evaluated_at": result.evaluated_at,
            "pred_path": result.pred_path,
            "score_diff_vs_official": result.score_diff_vs_official,
        }
        by_model[result.model_id] = entry

    entries = list(by_model.values())
    entries.sort(key=lambda x: float(x.get(sort_by, 0)), reverse=True)

    ranked_entries = []
    for idx, item in enumerate(entries, start=1):
        item["rank"] = idx
        ranked_entries.append(item)

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": benchmark,
        "entries": ranked_entries,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# OCR Leaderboard",
        "",
        f"- benchmark: `{benchmark}`",
        f"- updated_at: `{payload['updated_at']}`",
        "",
        "| Rank | Model | Score | CER | F1 | Diff vs Official | Repo |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in ranked_entries:
        metrics = item.get("metrics", {})
        cer = metrics.get("cer", "-")
        f1 = metrics.get("f1", "-")
        diff = item.get("score_diff_vs_official")
        diff_text = "-" if diff is None else f"{float(diff):.4f}"
        lines.append(
            "| {rank} | {model} | {score:.4f} | {cer} | {f1} | {diff} | {repo} |".format(
                rank=item.get("rank"),
                model=item.get("model_id"),
                score=float(item.get("overall_score", 0)),
                cer=cer,
                f1=f1,
                diff=diff_text,
                repo=item.get("repo_url", "-"),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"leaderboard_json": str(json_path), "leaderboard_md": str(md_path), "entries": len(ranked_entries)}

