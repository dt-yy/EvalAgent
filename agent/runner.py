from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import EvalJob


DEFAULT_SHARED_MOUNT_PREFIX = "/mnt/shared-storage-user/mineru2-shared"


def _safe_model_dir(model_id: str) -> str:
    return model_id.replace("/", "__").replace("\\", "__")


def _write_mock_prediction(pred_file: Path, input_dir: str, model_id: str) -> None:
    payload = {
        "image": f"{input_dir}/example_page_001.png",
        "model_id": model_id,
        "text": "mock prediction for MVP",
    }
    pred_file.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _normalize_path_text(path_value: str) -> str:
    return path_value.strip().rstrip("/")


def _enforce_skill_policy(config: dict[str, Any], mock_mode: bool) -> tuple[int, str]:
    shared_mount_prefix = _normalize_path_text(
        str(config.get("shared_mount_prefix", DEFAULT_SHARED_MOUNT_PREFIX))
    )
    input_images_dir = _normalize_path_text(
        str(config.get("input_images_dir", f"{DEFAULT_SHARED_MOUNT_PREFIX}/quyuan/1.6/images"))
    )
    hpc_output_root = _normalize_path_text(
        str(config.get("hpc_output_root", f"{DEFAULT_SHARED_MOUNT_PREFIX}/quyuan"))
    )
    enforce_shared_mount_paths = bool(config.get("enforce_shared_mount_paths", True))

    if enforce_shared_mount_paths:
        if not input_images_dir.startswith(shared_mount_prefix):
            raise ValueError(
                "Policy violation: input_images_dir must be under shared mount prefix "
                f"{shared_mount_prefix}"
            )
        if not hpc_output_root.startswith(shared_mount_prefix):
            raise ValueError(
                "Policy violation: hpc_output_root must be under shared mount prefix "
                f"{shared_mount_prefix}"
            )

    require_offline_compute_node = bool(config.get("require_offline_compute_node", True))
    allow_network_on_compute_node = bool(config.get("allow_network_on_compute_node", False))
    if require_offline_compute_node and allow_network_on_compute_node:
        raise ValueError(
            "Policy violation: compute node must be offline, but allow_network_on_compute_node=true."
        )

    requested_retry = int(config.get("max_retry", 2))
    hard_cap = int(config.get("max_retry_hard_cap", 2))
    if hard_cap < 0:
        hard_cap = 0
    max_retry = min(requested_retry, hard_cap)
    retry_note = ""
    if requested_retry > hard_cap:
        retry_note = f"max_retry_clamped_to_{hard_cap}"

    # For real execution, we require hpc_output_root to be explicitly set in config.
    if not mock_mode and not hpc_output_root:
        raise ValueError("Policy violation: hpc_output_root is required when mock_mode=false.")

    return max_retry, retry_note


def run_jobs(jobs: list[EvalJob], config: dict[str, Any]) -> list[EvalJob]:
    output_root = Path(config.get("output_root", "./results/predictions")).resolve()
    input_images_dir = config.get(
        "input_images_dir",
        "/mnt/shared-storage-user/mineru2-shared/quyuan/1.6/images",
    )
    mock_mode = bool(config.get("mock_mode", True))
    max_retry, retry_note = _enforce_skill_policy(config=config, mock_mode=mock_mode)

    output_root.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        if job.status != "ready":
            continue
        if retry_note:
            job.notes.append(retry_note)

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
            except (OSError, RuntimeError, ValueError, NotImplementedError) as exc:
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

