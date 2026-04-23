from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .logging_utils import get_logger, kv_to_text
from .types import EvalJob, EvalResult

logger = get_logger(__name__)


def _stable_score_seed(model_id: str) -> int:
    digest = hashlib.md5(model_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _mock_metrics(model_id: str) -> dict[str, float]:
    seed = _stable_score_seed(model_id)
    cer = round(0.02 + (seed % 30) / 1000, 4)
    f1 = round(0.86 + (seed % 120) / 1000, 4)
    overall = round((1 - cer) * 0.4 + f1 * 0.6, 4)
    return {"cer": cer, "f1": f1, "overall_score": overall}


def _tail_text(content: str, max_lines: int = 60) -> str:
    lines = content.strip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _run_shell_command(command: str, cwd: str | None = None) -> tuple[str, str]:
    completed = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_text = completed.stdout or ""
    stderr_text = completed.stderr or ""
    if completed.returncode != 0:
        raise RuntimeError(
            "Eval command failed with exit code "
            f"{completed.returncode}\n"
            f"command: {command}\n"
            f"stdout_tail:\n{_tail_text(stdout_text)}\n"
            f"stderr_tail:\n{_tail_text(stderr_text)}"
        )
    return stdout_text, stderr_text


def _safe_model_dir(model_id: str) -> str:
    return model_id.replace("/", "__").replace("\\", "__")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid yaml content: {path}")
    return payload


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _extract_path_value(payload: dict[str, Any], path_list: list[str], default: float = 0.0) -> float:
    cursor: Any = payload
    for key in path_list:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    try:
        return float(cursor)
    except (TypeError, ValueError):
        return default


def _normalize_percentage(value: float) -> float:
    # OmniDocBench stores some metrics in [0,1], convert them to [0,100].
    if value <= 1.5:
        return value * 100.0
    return value


def _build_omnidocbench_eval_config(job: EvalJob, config: dict[str, Any]) -> tuple[Path, Path, str]:
    repo_dir = Path(str(config.get("omnidocbench_repo_dir", "./OmniDocBench"))).resolve()
    template_path = Path(
        str(config.get("omnidocbench_config_template", repo_dir / "configs/end2end.yaml"))
    ).resolve()
    generated_dir = Path(
        str(config.get("omnidocbench_generated_config_dir", repo_dir / "configs/generated"))
    ).resolve()
    generated_path = generated_dir / f"end2end_{_safe_model_dir(job.model_id)}.yaml"
    gt_path = str(config.get("omnidocbench_ground_truth_path", "")).strip()
    if not gt_path:
        raise ValueError("Missing omnidocbench_ground_truth_path in config.")

    cfg = _load_yaml(template_path)
    if not cfg:
        raise ValueError("Empty OmniDocBench config template.")
    task_name = next(iter(cfg.keys()))
    task_cfg = cfg[task_name]
    dataset_cfg = task_cfg.setdefault("dataset", {})
    dataset_cfg.setdefault("ground_truth", {})
    dataset_cfg.setdefault("prediction", {})
    dataset_cfg["ground_truth"]["data_path"] = gt_path
    page_info = str(config.get("omnidocbench_ground_truth_page_info", "")).strip()
    if page_info:
        dataset_cfg["ground_truth"]["page_info"] = page_info
    dataset_cfg["prediction"]["data_path"] = str(job.pred_path)
    dataset_cfg["match_method"] = config.get("omnidocbench_match_method", "quick_match")
    _dump_yaml(generated_path, cfg)

    save_name = f"{Path(str(job.pred_path)).name}_{dataset_cfg['match_method']}"
    return repo_dir, generated_path, save_name


def _parse_omnidocbench_metric_result(metric_path: Path) -> dict[str, float]:
    payload = json.loads(metric_path.read_text(encoding="utf-8"))
    text_edit = _extract_path_value(payload, ["text_block", "all", "Edit_dist", "ALL_page_avg"], default=1.0)
    formula_cdm_raw = _extract_path_value(payload, ["display_formula", "page", "CDM", "ALL"], default=0.0)
    table_teds_raw = _extract_path_value(payload, ["table", "page", "TEDS", "ALL"], default=0.0)
    table_teds_s_raw = _extract_path_value(payload, ["table", "page", "TEDS_structure_only", "ALL"], default=0.0)
    reading_order_edit = _extract_path_value(
        payload,
        ["reading_order", "all", "Edit_dist", "ALL_page_avg"],
        default=1.0,
    )

    formula_cdm = _normalize_percentage(formula_cdm_raw)
    table_teds = _normalize_percentage(table_teds_raw)
    table_teds_s = _normalize_percentage(table_teds_s_raw)
    overall_score = round(((1.0 - text_edit) * 100.0 + formula_cdm + table_teds) / 3.0, 4)
    return {
        "text_edit_dist": round(text_edit, 6),
        "formula_cdm": round(formula_cdm, 4),
        "table_teds": round(table_teds, 4),
        "table_teds_structure_only": round(table_teds_s, 4),
        "reading_order_edit_dist": round(reading_order_edit, 6),
        "overall_score": overall_score,
    }


def _evaluate_job_with_omnidocbench(job: EvalJob, config: dict[str, Any]) -> tuple[dict[str, float], str]:
    repo_dir, generated_cfg_path, save_name = _build_omnidocbench_eval_config(job=job, config=config)
    python_bin = str(config.get("omnidocbench_python_bin", "python"))
    command_template = str(
        config.get(
            "omnidocbench_eval_command_template",
            '{python_bin} pdf_validation.py --config "{config_path}"',
        )
    )
    command = command_template.format(python_bin=python_bin, config_path=str(generated_cfg_path))
    logger.info(
        "evaluator run omnidocbench %s",
        kv_to_text(job_id=job.job_id, model_id=job.model_id, config_path=generated_cfg_path),
    )
    _run_shell_command(command=command, cwd=str(repo_dir))

    metric_result_path = Path(
        str(
            config.get(
                "omnidocbench_metric_result_path",
                repo_dir / "result" / f"{save_name}_metric_result.json",
            )
        )
    )
    if not metric_result_path.exists():
        raise FileNotFoundError(f"OmniDocBench metric result not found: {metric_result_path}")
    metrics = _parse_omnidocbench_metric_result(metric_result_path)
    logger.info(
        "evaluator metric parsed %s",
        kv_to_text(model_id=job.model_id, overall_score=metrics.get("overall_score"), metric_path=metric_result_path),
    )
    return metrics, str(metric_result_path)


def evaluate_jobs(jobs: list[EvalJob], config: dict[str, Any]) -> list[EvalResult]:
    benchmark = config.get("benchmark", "OmniDocBench")
    official_scores = config.get("official_scores", {})
    results_root = Path(config.get("results_root", "./results")).resolve()
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    real_eval_enabled = bool(config.get("real_eval_enabled", False))
    logger.info(
        "evaluator started %s",
        kv_to_text(jobs_total=len(jobs), real_eval_enabled=real_eval_enabled, run_id=run_id),
    )

    results: list[EvalResult] = []
    for job in jobs:
        if job.status != "success" or not job.pred_path:
            continue
        logger.info(
            "evaluator job started %s",
            kv_to_text(job_id=job.job_id, model_id=job.model_id, pred_path=job.pred_path),
        )

        if real_eval_enabled:
            metrics, metric_result_path = _evaluate_job_with_omnidocbench(job=job, config=config)
            overall_score = metrics["overall_score"]
            result_metrics = {
                "text_edit_dist": metrics["text_edit_dist"],
                "formula_cdm": metrics["formula_cdm"],
                "table_teds": metrics["table_teds"],
                "table_teds_structure_only": metrics["table_teds_structure_only"],
                "reading_order_edit_dist": metrics["reading_order_edit_dist"],
            }
        else:
            metrics = _mock_metrics(job.model_id)
            overall_score = metrics["overall_score"]
            result_metrics = {"cer": metrics["cer"], "f1": metrics["f1"]}
            metric_result_path = ""

        official = official_scores.get(job.model_id)
        score_diff = None if official is None else round(overall_score - float(official), 4)
        evaluated_at = datetime.now(timezone.utc).isoformat()
        result = EvalResult(
            model_id=job.model_id,
            run_id=run_id,
            repo_url=job.repo_url,
            benchmark=benchmark,
            metrics=result_metrics,
            overall_score=overall_score,
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
                    "omnidocbench_metric_result_path": metric_result_path,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info(
            "evaluator job finished %s",
            kv_to_text(
                job_id=job.job_id,
                model_id=job.model_id,
                overall_score=overall_score,
                result_file=target_file,
            ),
        )

    logger.info("evaluator finished %s", kv_to_text(results=len(results), run_id=run_id))
    return results

