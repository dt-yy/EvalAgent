from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .discovery import discover_candidates
from .evaluator import evaluate_jobs
from .filter import filter_candidates
from .runner import run_jobs
from .types import to_dict
from .updater import update_leaderboard


def _write_run_report(report_dir: str | Path, report: dict[str, Any]) -> str:
    target_dir = Path(report_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("pipeline_%Y%m%d_%H%M%S")
    target_path = target_dir / f"{run_id}.json"
    target_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(target_path)


def run_pipeline(configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    discovery_cfg = configs.get("discovery", {})
    eval_cfg = configs.get("evaluation", {})
    leaderboard_cfg = configs.get("leaderboard", {})

    candidates = discover_candidates(discovery_cfg)
    jobs = filter_candidates(candidates=candidates, config=eval_cfg)
    jobs_ready_before_run = sum(1 for j in jobs if j.status == "ready")
    jobs = run_jobs(jobs=jobs, config=eval_cfg)
    results = evaluate_jobs(jobs=jobs, config=leaderboard_cfg)
    leaderboard_output = update_leaderboard(results=results, config=leaderboard_cfg)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "candidates": len(candidates),
            "jobs_total": len(jobs),
            "jobs_ready_before_run": jobs_ready_before_run,
            "jobs_success": sum(1 for j in jobs if j.status == "success"),
            "jobs_failed": sum(1 for j in jobs if j.status == "failed"),
            "jobs_need_manual": sum(1 for j in jobs if j.status == "need_manual"),
            "results": len(results),
        },
        "candidates": to_dict(candidates),
        "jobs": to_dict(jobs),
        "results": to_dict(results),
        "leaderboard": leaderboard_output,
    }

    report_file = _write_run_report(eval_cfg.get("run_report_dir", "./results/runs"), report)
    report["run_report_file"] = report_file
    return report

