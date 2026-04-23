from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config import load_all_configs
from agent.logging_utils import configure_logging, get_logger
from agent.orchestrator import run_pipeline

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run OCR leaderboard agent pipeline once.")
    parser.add_argument("--config-dir", default="configs", help="Directory containing yaml configs.")
    args = parser.parse_args()
    logger.info("run_once started config_dir=%s", args.config_dir)

    configs = load_all_configs(args.config_dir)
    report = run_pipeline(configs)
    logger.info(
        "run_once finished jobs_success=%s jobs_failed=%s results=%s",
        report["counts"]["jobs_success"],
        report["counts"]["jobs_failed"],
        report["counts"]["results"],
    )
    print(json.dumps(report["counts"], indent=2, ensure_ascii=False))
    print(f"run_report_file={report['run_report_file']}")
    print(f"leaderboard_json={report['leaderboard']['leaderboard_json']}")
    print(f"leaderboard_md={report['leaderboard']['leaderboard_md']}")


if __name__ == "__main__":
    main()

