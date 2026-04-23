from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config import load_all_configs
from agent.evaluator import evaluate_jobs
from agent.runner import run_jobs
from agent.types import EvalJob, to_dict
from agent.updater import update_leaderboard


def _normalize_repo_key(value: str) -> str:
    return value.strip().lower().rstrip("/")


def _run_command(command: str, cwd: str | None = None) -> None:
    completed = subprocess.run(command, shell=True, cwd=cwd, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed({completed.returncode}): {command}")


def _ensure_repo(repo_url: str, repo_dir: Path, auto_pull: bool) -> None:
    if repo_dir.exists():
        if auto_pull:
            _run_command("git pull", cwd=str(repo_dir))
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _run_command(f'git clone "{repo_url}" "{repo_dir}"')


def _run_setup(eval_cfg: dict[str, Any]) -> None:
    auto_pull = bool(eval_cfg.get("setup_auto_pull_repo", True))
    mineru_repo_url = str(eval_cfg.get("mineru_repo_url", "https://github.com/opendatalab/MinerU.git"))
    mineru_repo_dir = Path(str(eval_cfg.get("mineru_repo_dir", "./third_party/MinerU"))).resolve()
    omnidocbench_repo_url = str(
        eval_cfg.get("omnidocbench_repo_url", "https://github.com/opendatalab/OmniDocBench.git")
    )
    omnidocbench_repo_dir = Path(str(eval_cfg.get("omnidocbench_repo_dir", "./third_party/OmniDocBench"))).resolve()

    _ensure_repo(repo_url=mineru_repo_url, repo_dir=mineru_repo_dir, auto_pull=auto_pull)
    _ensure_repo(repo_url=omnidocbench_repo_url, repo_dir=omnidocbench_repo_dir, auto_pull=auto_pull)

    for command in eval_cfg.get("mineru_setup_commands", []):
        if isinstance(command, str) and command.strip():
            _run_command(command, cwd=str(mineru_repo_dir))
    for command in eval_cfg.get("omnidocbench_setup_commands", []):
        if isinstance(command, str) and command.strip():
            _run_command(command, cwd=str(omnidocbench_repo_dir))


def _assert_readme_gate(model_id: str, repo_url: str, eval_cfg: dict[str, Any]) -> None:
    if not bool(eval_cfg.get("enforce_readme_gate", False)):
        return
    verified = {
        _normalize_repo_key(item)
        for item in eval_cfg.get("readme_verified_repos", [])
        if isinstance(item, str)
    }
    keys = {_normalize_repo_key(model_id), _normalize_repo_key(repo_url)}
    if not (keys & verified):
        raise ValueError(
            "README gate blocked this model. Add model_id or repo_url to readme_verified_repos in configs/eval.yaml."
        )


def _write_run_report(report_dir: str | Path, report: dict[str, Any]) -> str:
    target_dir = Path(report_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("mineru_eval_%Y%m%d_%H%M%S")
    target_path = target_dir / f"{run_id}.json"
    target_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(target_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MinerU on OmniDocBench without discovery module.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--model-id", default="opendatalab/mineru")
    parser.add_argument("--repo-url", default="https://github.com/opendatalab/MinerU")
    parser.add_argument("--ref", default="master")
    parser.add_argument("--with-setup", action="store_true", help="Clone repos and run setup commands.")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Enable real infer + real eval (override mock settings).",
    )
    parser.add_argument(
        "--fail-on-job-failure",
        action="store_true",
        help="Exit with non-zero code when any job fails.",
    )
    args = parser.parse_args()

    configs = load_all_configs(args.config_dir)
    eval_cfg = configs.get("evaluation", {})
    leaderboard_cfg = configs.get("leaderboard", {})

    if args.real:
        eval_cfg["mock_mode"] = False
        eval_cfg["real_infer_enabled"] = True
        eval_cfg["real_eval_enabled"] = True

    _assert_readme_gate(model_id=args.model_id, repo_url=args.repo_url, eval_cfg=eval_cfg)
    if args.with_setup:
        _run_setup(eval_cfg)

    jobs = [
        EvalJob(
            job_id="job_mineru_0001",
            model_id=args.model_id,
            repo_url=args.repo_url,
            ref=args.ref,
            status="ready",
            priority=1,
        )
    ]
    jobs = run_jobs(jobs=jobs, config=eval_cfg)
    results = evaluate_jobs(jobs=jobs, config={**leaderboard_cfg, **eval_cfg})
    leaderboard_output = update_leaderboard(results=results, config=leaderboard_cfg)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "jobs_total": len(jobs),
            "jobs_success": sum(1 for j in jobs if j.status == "success"),
            "jobs_failed": sum(1 for j in jobs if j.status == "failed"),
            "results": len(results),
        },
        "jobs": to_dict(jobs),
        "results": to_dict(results),
        "leaderboard": leaderboard_output,
    }
    report_path = _write_run_report(eval_cfg.get("run_report_dir", "./results/runs"), report)

    print(json.dumps(report["counts"], indent=2, ensure_ascii=False))
    print(f"run_report_file={report_path}")
    print(f"leaderboard_json={leaderboard_output['leaderboard_json']}")
    print(f"leaderboard_md={leaderboard_output['leaderboard_md']}")

    if args.fail_on_job_failure and report["counts"]["jobs_failed"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

