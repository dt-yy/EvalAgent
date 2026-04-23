# OCR Leaderboard Agent

一个用于**自动发现 GitHub 新 OCR 模型**并**自动更新评测榜单**的 Agent 项目。

## 1. 项目目标

构建一个端到端自动化系统，持续完成以下任务：

1. 定期扫描 GitHub 上新发布或活跃更新的 OCR 相关模型仓库。
2. 自动筛选可评测候选模型（可复现性、推理可用性）。
3. 在统一评测集上执行推理并计算指标。
4. 自动更新排行榜（Markdown/JSON/API/网页）。
5. 对失败任务自动重试，并给出可追踪日志。

## 2. 典型使用场景

- 你维护一个 OCR 模型榜单，希望每日自动更新。
- 你希望新模型出现后，无需人工介入即可进入评测队列。
- 你希望有统一、可复现的评测流程和版本化结果。

## 3. 模块组织（建议）

建议拆分为 6 个核心模块，并统一由 `orchestrator.py` 编排。

### 3.1 模块职责与边界

### A. Discovery（发现）

- 职责：从 GitHub 拉取“新增/更新”的 OCR 候选仓库。
- 只做“收集”，不做复杂可运行性判断。
- 输入：扫描窗口、关键词、stars 下限、上次游标。
- 输出：`candidate_records`（原始候选列表）。

### B. Candidate Filter（候选过滤）

- 职责：把 `candidate_records` 转为“可执行评测任务”。
- 规则：OCR 相关性、推理入口可用性、权重可获得性、许可证可发布性。
- 输入：`candidate_records`。
- 输出：`eval_jobs`（状态标签 `ready/need_manual/skip`）。

### C. Runner（评测执行）

- 职责：消费 `ready` 任务，完成模型推理并产出原始预测。
- 统一处理：超时、GPU 资源、重试、日志、错误分类。
- 输入：`eval_jobs` + benchmark 配置。
- 输出：`predictions` + `run_meta`。

### D. Evaluator（指标计算）

- 职责：读取预测结果并计算统一指标。
- 输入：`predictions` + GT。
- 输出：标准化 `eval_result.json`（建议写入 `results/<model_id>/<run_id>.json`）。

### E. Leaderboard Updater（榜单更新）

- 职责：把 `eval_result.json` 合并进榜单并重排。
- 输入：历史榜单 + 新结果。
- 输出：`leaderboard/leaderboard.json`、`leaderboard/leaderboard.md`、变更摘要。

### F. Scheduler + Agent Orchestrator（调度与编排）

- 职责：触发周期任务、维护状态机、通知结果。
- 触发方式：Cron / GitHub Actions / Airflow / 自建调度。
- 建议状态机：`queued -> running -> success/failed/skipped`。

### 3.2 模块调用关系

```text
Scheduler/Trigger
    -> Orchestrator
        -> Discovery
        -> Candidate Filter
        -> Runner
        -> Evaluator
        -> Leaderboard Updater
        -> Notification/Archive
```

### 3.3 模块接口约定（最小字段）

- `candidate_record`：`repo_url`, `owner`, `updated_at`, `stars`, `default_branch`
- `eval_job`：`model_id`, `repo_url`, `ref`, `status`, `priority`, `retry_count`
- `eval_result`：`model_id`, `run_id`, `metrics`, `overall_score`, `evaluated_at`

> 建议原则：模块间只传“结构化对象”，不要直接共享脚本内部状态，便于后续替换执行后端（本地、集群、容器）。

### 3.4 开发顺序（便于快速落地）

1. 先实现 `Discovery + Filter`，打通候选入队。
2. 再实现 `Runner + Evaluator`，打通跑分闭环。
3. 最后实现 `Updater + Orchestrator`，形成自动更新与通知。

## 4. 推荐目录结构

```text
.
├─ README.md
├─ configs/
│  ├─ discovery.yaml
│  ├─ eval.yaml
│  └─ leaderboard.yaml
├─ agent/
│  ├─ discovery.py
│  ├─ filter.py
│  ├─ runner.py
│  ├─ evaluator.py
│  ├─ updater.py
│  └─ orchestrator.py
├─ data/
│  ├─ benchmarks/
│  └─ cache/
├─ results/
│  └─ <model_id>/
├─ leaderboard/
│  ├─ leaderboard.json
│  └─ leaderboard.md
└─ scripts/
   ├─ run_once.py
   └─ backfill.py
```

## 5. 自动更新流程（建议）

1. `Discovery` 拉取最近 N 小时新增/更新仓库。
2. `Filter` 过滤不可评测项目并打标签（`ready`, `need_manual`, `skip`）。
3. `Runner` 对 `ready` 模型执行推理任务。
4. `Evaluator` 计算指标并写入标准结果。
5. `Updater` 刷新排行榜并生成变更摘要。
6. `Orchestrator` 发送通知并归档日志。

## 6. 配置示例（建议字段）

```yaml
github:
  token_env: GITHUB_TOKEN
  search_window_hours: 24
  keywords: ["OCR", "document parsing", "text recognition"]
  min_stars: 20

evaluation:
  benchmark: OmniDocBench
  max_retry: 2
  timeout_min: 60
  device: "cuda:0"

leaderboard:
  sort_by: "overall_score"
  min_runs_for_publish: 1
  publish_targets: ["leaderboard/leaderboard.json", "leaderboard/leaderboard.md"]
```

## 7. 榜单数据结构（JSON）

```json
{
  "updated_at": "2026-04-23T12:00:00Z",
  "benchmark": "OmniDocBench",
  "entries": [
    {
      "rank": 1,
      "model_id": "org/model-name",
      "repo": "https://github.com/org/repo",
      "score": 91.23,
      "metrics": {
        "cer": 0.041,
        "f1": 0.932
      },
      "evaluated_at": "2026-04-23T11:58:00Z",
      "run_id": "run_20260423_1158"
    }
  ]
}
```

## 8. 工程落地建议

- 使用任务状态机（`queued/running/success/failed/skipped`）避免重复执行。
- 对模型仓库做去重键（`repo + commit/tag + eval_config_hash`）。
- 将“发现”和“评测”解耦，避免 GitHub 抖动影响评测队列。
- 为每次跑分记录环境指纹（CUDA、驱动、依赖版本）。
- 为排行榜更新增加最小质量门槛（结果完整性、日志齐全、指标合法）。

## 9. 最小可用版本（MVP）

第一阶段建议只做：

1. GitHub 每日扫描 + OCR 关键词过滤。
2. 对接 1 个基准（例如 OmniDocBench）。
3. 评测完成后自动更新 `leaderboard.json` 和 `leaderboard.md`。
4. 失败重试 2 次 + 通知。

这样可以在较短周期内先跑通闭环，再逐步扩展到多数据集、多榜单、多执行后端。

## 10. 后续扩展

- 多源发现：Hugging Face、Papers with Code、ArXiv。
- 增加“可信度评分”：代码质量、文档完整度、社区活跃度。
- 自动创建 PR：每次榜单更新由 Agent 提交并附评测摘要。
- 可视化 Dashboard：历史趋势、模型分项雷达图、回归检测。

---

## 11. 当前仓库已落地模块

- `agent/discovery.py`：GitHub 候选发现（含 seed fallback）。
- `agent/filter.py`：候选过滤，输出 `ready/need_manual/skipped`。
- `agent/runner.py`：推理执行层（MVP 默认 `mock_mode=true`，含失败重试 2 次）。
- `agent/evaluator.py`：评测层（当前为可替换的最小实现）。
- `agent/updater.py`：更新 `leaderboard/leaderboard.json` 和 `leaderboard/leaderboard.md`。
- `agent/orchestrator.py`：统一编排端到端流程并输出运行报告。
- `scripts/run_once.py`：单次执行入口。
- `scripts/backfill.py`：回填/多次执行入口。
- `configs/discovery.yaml`、`configs/eval.yaml`、`configs/leaderboard.yaml`：默认配置。

## 12. 快速开始（MVP）

1. 安装依赖：

```bash
pip install pyyaml
```

2. 执行一次流程：

```bash
python scripts/run_once.py --config-dir configs
```

3. 查看产物：

- 运行报告：`results/runs/*.json`
- 榜单 JSON：`leaderboard/leaderboard.json`
- 榜单 Markdown：`leaderboard/leaderboard.md`

> 当前默认 `mock_mode=true`，用于先打通全链路。接入真实 vLLM + rlaunch 时，将 `agent/runner.py` 的执行逻辑替换为真实命令并把 `configs/eval.yaml` 中 `mock_mode` 改为 `false`。

## 13. Skill 规则已固化为 Agent 策略

`configs/eval.yaml` 已内置以下硬约束：

- `enforce_readme_gate: true`：只有 `readme_verified_repos` 中确认过 README 的仓库才会进入 `ready`。
- `enforce_shared_mount_paths: true`：强制 `input_images_dir` 和 `hpc_output_root` 落在共享挂载前缀下。
- `require_offline_compute_node: true` + `allow_network_on_compute_node: false`：默认按离线计算节点策略执行。
- `max_retry_hard_cap: 2`：即使误配更大值，也会被 runner 自动夹到 2。

如果你在某次实验想临时放宽规则，建议只改对应配置，不改代码逻辑。
