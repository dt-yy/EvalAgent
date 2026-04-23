from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from .ai_client import suggest_infer_command_with_ai
from .logging_utils import get_logger, kv_to_text
from .types import EvalJob

logger = get_logger(__name__)


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


def _tail_text(content: str, max_lines: int = 40) -> str:
    lines = content.strip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _escape_single_quotes_for_bash(command: str) -> str:
    return command.replace("'", "'\"'\"'")


def _run_shell_command(command: str, cwd: str | None = None, timeout: int | None = None) -> tuple[str, str]:
    logger.info("shell command start timeout=%s", timeout)
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_partial = exc.stdout or ""
        stderr_partial = exc.stderr or ""
        logger.error("shell command timed out after %ss stdout_lines=%d stderr_lines=%d",
                     timeout, len(stdout_partial.splitlines()), len(stderr_partial.splitlines()))
        if stdout_partial:
            logger.error("shell timeout stdout (last 20 lines):\n%s", _tail_text(stdout_partial, 20))
        if stderr_partial:
            logger.error("shell timeout stderr (last 20 lines):\n%s", _tail_text(stderr_partial, 20))
        raise RuntimeError(
            f"Command timed out after {timeout}s\n"
            f"command: {command}\n"
            f"stdout_tail:\n{_tail_text(stdout_partial)}\n"
            f"stderr_tail:\n{_tail_text(stderr_partial)}"
        )
    stdout_text = completed.stdout or ""
    stderr_text = completed.stderr or ""
    if stdout_text:
        logger.info("shell stdout (last 20 lines):\n%s", _tail_text(stdout_text, 20))
    if stderr_text:
        logger.info("shell stderr (last 20 lines):\n%s", _tail_text(stderr_text, 20))
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code "
            f"{completed.returncode}\n"
            f"command: {command}\n"
            f"stdout_tail:\n{_tail_text(stdout_text)}\n"
            f"stderr_tail:\n{_tail_text(stderr_text)}"
        )
    logger.info("shell command done exit_code=0")
    return stdout_text, stderr_text


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


def _resolve_prediction_dir(job: EvalJob, config: dict[str, Any], mock_mode: bool) -> str:
    model_dir_name_map = config.get("model_output_dir_map", {})
    model_dir_name = model_dir_name_map.get(job.model_id, _safe_model_dir(job.model_id))

    if mock_mode:
        output_root = Path(config.get("output_root", "./results/predictions")).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        return str(output_root / model_dir_name)

    hpc_output_root = str(config.get("hpc_output_root", DEFAULT_SHARED_MOUNT_PREFIX + "/quyuan"))
    normalized_root = hpc_output_root.strip()
    if normalized_root.startswith("/"):
        return str(PurePosixPath(normalized_root) / model_dir_name)
    return str(Path(normalized_root) / model_dir_name)


def _build_real_infer_command(
    *,
    config: dict[str, Any],
    input_images_dir: str,
    pred_path: str,
    model_id: str,
) -> str:
    template = str(
        config.get(
            "infer_command_template",
            'mineru -p "{input_images_dir}" -o "{pred_path}" -b "{mineru_backend}"',
        )
    )
    inner_command = template.format(
        input_images_dir=input_images_dir,
        pred_path=pred_path,
        model_id=model_id,
        mineru_backend=config.get("mineru_backend", "pipeline"),
    )
    use_rlaunch_wrapper = bool(config.get("use_rlaunch_wrapper", False))
    if not use_rlaunch_wrapper:
        return inner_command

    conda_env = str(config.get("infer_conda_env", "")).strip()
    conda_init = str(config.get("infer_conda_init", "source ~/.bashrc")).strip()
    if conda_env:
        wrapped_inner_command = f"{conda_init} && conda activate {conda_env} && {inner_command}"
    else:
        wrapped_inner_command = inner_command

    rlaunch_template = str(
        config.get(
            "rlaunch_command_template",
            "rlaunch --memory={rlaunch_memory} --gpu={rlaunch_gpu} --cpu={rlaunch_cpu} "
            "--charged-group={rlaunch_charged_group} --private-machine=yes "
            "--namespace={rlaunch_namespace} "
            "--mount=gpfs://{rlaunch_mount_src}:{rlaunch_mount_dst} "
            "-- bash -lc '{inner_infer_command_escaped}'",
        )
    )
    return rlaunch_template.format(
        inner_infer_command=wrapped_inner_command,
        inner_infer_command_escaped=_escape_single_quotes_for_bash(wrapped_inner_command),
        rlaunch_memory=config.get("rlaunch_memory", 64000),
        rlaunch_gpu=config.get("rlaunch_gpu", 2),
        rlaunch_cpu=config.get("rlaunch_cpu", 32),
        rlaunch_charged_group=config.get("rlaunch_charged_group", "mineruinfra_gpu"),
        rlaunch_namespace=config.get("rlaunch_namespace", "ailab-mineruinfra"),
        rlaunch_mount_src=config.get("rlaunch_mount_src", "gpfs1/mineru2-shared"),
        rlaunch_mount_dst=config.get("rlaunch_mount_dst", DEFAULT_SHARED_MOUNT_PREFIX),
    )


def _resolve_ai_infer_overrides(
    *,
    job: EvalJob,
    config: dict[str, Any],
    input_images_dir: str,
    pred_path: str,
) -> tuple[str | None, str | None, list[str]]:
    notes: list[str] = []
    ai_plan = suggest_infer_command_with_ai(
        repo_url=job.repo_url,
        ref=job.ref,
        model_id=job.model_id,
        current_template=str(config.get("infer_command_template", "")),
        current_backend=str(config.get("mineru_backend", "pipeline")),
        input_images_dir=input_images_dir,
        pred_path=pred_path,
        config=config,
    )
    if ai_plan is None:
        return None, None, notes

    confidence = float(ai_plan.get("confidence", 0.0))
    notes.append(f"ai_infer_confidence={confidence:.2f}")
    reason = str(ai_plan.get("reason", ""))
    if reason:
        notes.append(f"ai_infer_reason={reason}")

    min_conf = float(config.get("ai_infer_planner_min_confidence", 0.75))
    if confidence < min_conf:
        notes.append("ai_infer_below_threshold")
        return None, None, notes

    return (
        str(ai_plan.get("infer_command_template", "")).strip() or None,
        str(ai_plan.get("mineru_backend", "")).strip() or None,
        notes,
    )


def _has_prediction_outputs(pred_dir: Path, file_glob: str) -> bool:
    return any(pred_dir.glob(file_glob))


def run_jobs(jobs: list[EvalJob], config: dict[str, Any]) -> list[EvalJob]:
    input_images_dir = config.get(
        "input_images_dir",
        "/mnt/shared-storage-user/mineru2-shared/quyuan/1.6/images",
    )
    mock_mode = bool(config.get("mock_mode", True))
    real_infer_enabled = bool(config.get("real_infer_enabled", False))
    if real_infer_enabled:
        mock_mode = False

    max_retry, retry_note = _enforce_skill_policy(config=config, mock_mode=mock_mode)
    prediction_file_glob = str(config.get("prediction_file_glob", "**/*.md"))
    infer_workdir = config.get("mineru_workdir")
    log_full_command = bool(config.get("log_full_infer_command", True))
    timeout_min = float(config.get("timeout_min", 60))
    timeout_sec = timeout_min * 60
    logger.info(
        "runner started %s",
        kv_to_text(
            jobs_total=len(jobs),
            mock_mode=mock_mode,
            real_infer_enabled=real_infer_enabled,
            max_retry=max_retry,
            prediction_file_glob=prediction_file_glob,
        ),
    )

    for job in jobs:
        if job.status != "ready":
            continue
        if retry_note:
            job.notes.append(retry_note)

        model_dir = _resolve_prediction_dir(job=job, config=config, mock_mode=mock_mode)
        model_dir_path = Path(model_dir)
        if mock_mode:
            model_dir_path.mkdir(parents=True, exist_ok=True)
        pred_file = model_dir_path / "predictions.jsonl"
        run_meta_file = model_dir_path / "run_meta.json"
        ai_template: str | None = None
        ai_backend: str | None = None
        ai_notes: list[str] = []
        logger.info(
            "runner job started %s",
            kv_to_text(job_id=job.job_id, model_id=job.model_id, pred_path=model_dir),
        )
        if not mock_mode:
            ai_template, ai_backend, ai_notes = _resolve_ai_infer_overrides(
                job=job,
                config=config,
                input_images_dir=input_images_dir,
                pred_path=model_dir,
            )
            job.notes.extend(ai_notes)
            if ai_template:
                job.notes.append("ai_infer_template_applied")
            if ai_backend:
                job.notes.append(f"ai_backend_applied={ai_backend}")

        attempts = 0
        job.status = "running"
        while attempts <= max_retry:
            attempts += 1
            job.retry_count = attempts - 1

            try:
                if mock_mode:
                    _write_mock_prediction(pred_file=pred_file, input_dir=input_images_dir, model_id=job.model_id)
                else:
                    runtime_config = dict(config)
                    if ai_template:
                        runtime_config["infer_command_template"] = ai_template
                    if ai_backend:
                        runtime_config["mineru_backend"] = ai_backend

                    command = _build_real_infer_command(
                        config=runtime_config,
                        input_images_dir=input_images_dir,
                        pred_path=model_dir,
                        model_id=job.model_id,
                    )
                    if log_full_command:
                        logger.info(
                            "runner command %s",
                            kv_to_text(job_id=job.job_id, attempt=attempts, command=command),
                        )
                    else:
                        logger.info(
                            "runner command built %s",
                            kv_to_text(job_id=job.job_id, attempt=attempts, command_hidden=True),
                        )
                    stdout_text, stderr_text = _run_shell_command(command=command, cwd=infer_workdir, timeout=timeout_sec)
                    if stdout_text:
                        job.notes.append("infer_stdout_captured")
                    if stderr_text:
                        job.notes.append("infer_stderr_captured")

                if mock_mode and pred_file.exists() and pred_file.stat().st_size > 0:
                    job.status = "success"
                    job.pred_path = model_dir
                    job.error = None
                    break

                if not mock_mode and _has_prediction_outputs(model_dir_path, prediction_file_glob):
                    job.status = "success"
                    job.pred_path = model_dir
                    job.error = None
                    break

                job.error = "empty_output"
                logger.warning(
                    "runner empty output %s",
                    kv_to_text(job_id=job.job_id, attempt=attempts, pred_path=model_dir),
                )
            except (OSError, RuntimeError, ValueError) as exc:
                job.error = str(exc)
                logger.warning(
                    "runner attempt failed %s",
                    kv_to_text(job_id=job.job_id, attempt=attempts, error=type(exc).__name__),
                )

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
        logger.info(
            "runner job finished %s",
            kv_to_text(
                job_id=job.job_id,
                model_id=job.model_id,
                status=job.status,
                retry_count=job.retry_count,
                pred_path=job.pred_path,
            ),
        )

    logger.info(
        "runner finished %s",
        kv_to_text(
            success=sum(1 for j in jobs if j.status == "success"),
            failed=sum(1 for j in jobs if j.status == "failed"),
            running=sum(1 for j in jobs if j.status == "running"),
        ),
    )
    return jobs

