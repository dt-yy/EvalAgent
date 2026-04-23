from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import EvalJob


def _safe_model_dir(model_id: str) -> str:
    return model_id.replace("/", "__").replace("\\", "__")


def _write_mock_prediction(pred_file: Path, input_dir: str, model_id: str) -> None:
    payload = {
        "image": f"{input_dir}/example_page_001.png",
        "model_id": model_id,
        "text": "mock prediction for MVP",
    }
    pred_file.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def run_jobs(jobs: list[EvalJob], config: dict[str, Any]) -> list[EvalJob]:
    output_root = Path(config.get("output_root", "./results/predictions")).resolve()
    input_images_dir = config.get(
        "input_images_dir",
        "/mnt/shared-storage-user/mineru2-shared/quyuan/1.6/images",
    )
    max_retry = int(config.get("max_retry", 2))
    mock_mode = bool(config.get("mock_mode", True))

    output_root.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        if job.status != "ready":
            continue

        model_dir = output_root / _safe_model_dir(job.model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        pred_file = model_dir / "predictions.jsonl"
        run_meta_file = model_dir / "run_meta.json"

        attempts = 0
        job.status = "running"
        while attempts <= max_retry:
            attempts += 1
            job.retry_count = attempts - 1

            try:
                if mock_mode:
                    _write_mock_prediction(pred_file=pred_file, input_dir=input_images_dir, model_id=job.model_id)
                else:
                    raise NotImplementedError(
                        "Set mock_mode=true for MVP, or replace with real vLLM+rlaunch execution."
                    )

                if pred_file.exists() and pred_file.stat().st_size > 0:
                    job.status = "success"
                    job.pred_path = str(model_dir)
                    job.error = None
                    break
                job.error = "empty_output"
            except Exception as exc:
                job.error = str(exc)

            if attempts > max_retry:
                job.status = "failed"
                break

        run_meta = {
            "model_id": job.model_id,
            "job_id": job.job_id,
            "status": job.status,
            "retry_count": job.retry_count,
            "pred_path": job.pred_path,
            "error": job.error,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        run_meta_file.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return jobs

