from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt  # pylint: disable=import-error
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # pylint: disable=import-error

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def draw_box(ax, x, y, w, h, title, subtitle="", face="#EAF2FF", edge="#3B82F6"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=11, color="#111827")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=8.5, color="#374151")


def draw_arrow(ax, x1, y1, x2, y2, dashed=False, text=""):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=12,
        linewidth=1.4,
        color="#4B5563",
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(arr)
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.02, text, ha="center", va="bottom", fontsize=8, color="#4B5563")


def main() -> None:
    output_path = Path("leaderboard/ocr-leaderboard-agent-architecture.png").resolve()

    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.03, 0.96, "OCR Leaderboard Agent 架构图", fontsize=20, fontweight="bold", color="#111827")
    ax.text(0.03, 0.925, "自动发现新模型 -> 自动评测 -> 自动更新榜单", fontsize=11, color="#6B7280")

    # Core pipeline
    w, h = 0.15, 0.12
    y = 0.62
    xs = [0.03, 0.22, 0.41, 0.60, 0.79]
    titles = [
        ("GitHub 感知", "Search API / Topics"),
        ("候选过滤", "README Gate / License"),
        ("推理执行 Runner", "vLLM / rlaunch / 重试<=2"),
        ("评测 Evaluator", "OmniDocBench"),
        ("榜单更新 Updater", "leaderboard.json/md"),
    ]
    for x, (title, subtitle) in zip(xs, titles):
        draw_box(ax, x, y, w, h, title, subtitle)

    for i in range(len(xs) - 1):
        draw_arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    # Scheduler / orchestrator
    draw_box(
        ax,
        0.36,
        0.83,
        0.28,
        0.09,
        "Scheduler / Orchestrator",
        "统一调度与状态编排",
        face="#F3F4F6",
        edge="#6B7280",
    )
    for x in xs:
        draw_arrow(ax, 0.50, 0.83, x + w / 2, y + h, dashed=True)

    # Manual queue
    draw_box(
        ax,
        0.22,
        0.42,
        0.15,
        0.10,
        "need_manual 队列",
        "人工确认后再入队",
        face="#FEF3C7",
        edge="#D97706",
    )
    draw_arrow(ax, 0.295, 0.62, 0.295, 0.52, dashed=True, text="未通过 gate")

    # Artifacts
    draw_box(ax, 0.41, 0.40, 0.15, 0.10, "predictions.jsonl", "推理输出", face="#ECFDF5", edge="#059669")
    draw_arrow(ax, 0.485, 0.62, 0.485, 0.50)
    draw_box(
        ax,
        0.60,
        0.40,
        0.15,
        0.10,
        "results/<model>/<run>.json",
        "评测结果",
        face="#ECFDF5",
        edge="#059669",
    )
    draw_arrow(ax, 0.675, 0.62, 0.675, 0.50)
    draw_box(ax, 0.79, 0.40, 0.15, 0.10, "results/runs/*.json", "运行报告", face="#ECFDF5", edge="#059669")
    draw_arrow(ax, 0.865, 0.62, 0.865, 0.50)

    # Policy box
    draw_box(
        ax,
        0.03,
        0.15,
        0.91,
        0.16,
        "策略约束",
        "README 先读 | 共享挂载路径校验 | 计算节点离线保护 | 重试上限 hard cap=2",
        face="#EEF2FF",
        edge="#4F46E5",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(str(output_path))


if __name__ == "__main__":
    main()

