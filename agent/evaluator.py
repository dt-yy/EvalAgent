from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import EvalJob, EvalResult


def _stable_score_seed(model_id: str) -> int:
    digest = hashlib.md5(model_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _mock_metrics(model_id: str) -> dict[str, float]:
    seed = _stable_score_seed(model_id)
    cer = round(0.02 + (seed % 30) / 1000, 4)
    f1 = round(0.86 + (seed % 120) / 1000, 4)
    overall = round((1 - cer) * 0.4 + f1 * 0.6, 4)
    return {"cer": cer, "f1": f1, "overall_score": overall}


def evaluate_jobs(jobs: list[EvalJob], config: dict[str, Any]) -> list[EvalResult]:
    benchmark = config.get("benchmark", "OmniDocBench")
    official_scores = config.get("official_scores", {})
    results_root = Path(config.get("results_root", "./results")).resolve()
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")

    results: list[EvalResult] = []
    for job in jobs:
        if job.status != "success" or not job.pred_path:
            continue

        metrics = _mock_metrics(job.model_id)
        official = official_scores.get(job.model_id)
        score_diff = None if official is None else round(metrics["overall_score"] - float(official), 4)
        evaluated_at = datetime.now(timezone.utc).isoformat()
        result = EvalResult(
            model_id=job.model_id,
            run_id=run_id,
            repo_url=job.repo_url,
            benchmark=benchmark,
            metrics={"cer": metrics["cer"], "f1": metrics["f1"]},
            overall_score=metrics["overall_score"],
            evaluated_at=evaluated_at,
            pred_path=job.pred_path,
            score_diff_vs_official=score_diff,
        )
        results.append(result)

        model_dir = results_root / job.model_id.replace("/", "__")
        model_dir.mkdir(parents=True, exist_ok=True)
        target_file = model_dir / f"{run_id}.json"
        target_file.write_text(
            json.dumps(
                {
                    "model_id": result.model_id,
                    "run_id": result.run_id,
                    "repo_url": result.repo_url,
                    "benchmark": result.benchmark,
                    "metrics": result.metrics,
                    "overall_score": result.overall_score,
                    "evaluated_at": result.evaluated_at,
                    "pred_path": result.pred_path,
                    "score_diff_vs_official": result.score_diff_vs_official,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return results

