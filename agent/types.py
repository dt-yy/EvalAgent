from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

JobStatus = Literal["queued", "ready", "running", "success", "failed", "need_manual", "skipped"]


@dataclass
class CandidateRecord:
    repo_id: str
    repo_url: str
    owner: str
    name: str
    updated_at: str
    stars: int = 0
    default_branch: str = "main"
    license_spdx: str | None = None
    summary: str = ""


@dataclass
class EvalJob:
    job_id: str
    model_id: str
    repo_url: str
    ref: str
    status: JobStatus
    priority: int = 100
    retry_count: int = 0
    pred_path: str | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    model_id: str
    run_id: str
    repo_url: str
    benchmark: str
    metrics: dict[str, float]
    overall_score: float
    evaluated_at: str
    pred_path: str
    score_diff_vs_official: float | None = None


def to_dict(data: Any) -> Any:
    if hasattr(data, "__dataclass_fields__"):
        return asdict(data)
    if isinstance(data, list):
        return [to_dict(item) for item in data]
    if isinstance(data, dict):
        return {k: to_dict(v) for k, v in data.items()}
    return data

