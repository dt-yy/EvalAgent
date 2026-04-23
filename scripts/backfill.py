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
    parser = argparse.ArgumentParser(description="Run OCR leaderboard pipeline multiple times.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--times", type=int, default=3)
    args = parser.parse_args()
    logger.info("backfill started config_dir=%s times=%s", args.config_dir, args.times)

    configs = load_all_configs(args.config_dir)
    all_reports = []
    for _ in range(max(1, args.times)):
        report = run_pipeline(configs)
        all_reports.append(
            {
                "counts": report["counts"],
                "run_report_file": report["run_report_file"],
            }
        )
    logger.info("backfill finished runs=%s", len(all_reports))

    print(json.dumps(all_reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

