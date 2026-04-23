from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required. Install with: pip install pyyaml"
        ) from exc

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(content, dict):
        raise ValueError(f"Config must be a mapping object: {path}")
    return content


def load_all_configs(config_dir: str | Path) -> dict[str, dict[str, Any]]:
    config_path = Path(config_dir).resolve()
    return {
        "discovery": _load_yaml(config_path / "discovery.yaml"),
        "evaluation": _load_yaml(config_path / "eval.yaml"),
        "leaderboard": _load_yaml(config_path / "leaderboard.yaml"),
    }

