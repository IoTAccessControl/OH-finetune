# 鸿蒙应用代码隐私合规审计 — Qwen LoRA SFT

基于两阶段隐私数据（`privacy_two_stage_*`），用 LoRA 监督微调（SFT）让 Qwen 系列模型学会输出结构化、可审计的代码隐私合规报告（`collectionAndUse` / `permissions` JSON 字段）。

## 结果速览（无泄漏测试集，200 条）

| 模型 | 微调后 key_coverage | base 对照 key_coverage | 说明 |
|------|--------------------:|----------------------:|------|
| Qwen3-32B (LoRA r=32) | **0.95** | 0.00 | 97.5% 样本产出规范字段 |
| Qwen2.5-7B-IT (LoRA r=32) | **0.97** | 0.01 | 99.5% 样本产出规范字段 |

base 对照 = 关闭 LoRA adapter（`model.disable_adapter()`）后基座模型直接生成，
其合规字段覆盖率接近 0%，佐证提升来自微调学到的格式而非基座自带能力。

## 目录结构

```
.
├── README.md                       # 本文件
├── DATA_SPLIT.md                   # 训练/验证/测试划分 + 不交性校验
├── requirements.txt                # Python 依赖
├── scripts/
│   ├── train_sft_32b_rerun.py      # 32B 训练（字段 remap + 跑满 3 epoch）
│   ├── train_sft_7b.py             # 7B 训练
│   ├── evaluate_twostage_test.py   # 32B 测试集评测（adapter vs base 对照）
│   └── evaluate_twostage_test_7b.py
├── results/
│   ├── results_test_200_32b_rerun.json
│   └── results_test_200_7b.json
└── docs/
    └── REPORT.md                   # 方案说明 + 结果留底
```

> **注意**：本目录不再包含 `data/` 子目录。训练数据统一存放在仓库根目录的
> `data/two_stage_llamafactory/` 下（与本目录同级），训练/评测脚本通过相对路径引用。
> 这样做是为了避免数据重复存储，并方便 LLaMA-Factory 直接注册使用。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 可选，显著降低显存占用并提速：
pip install flash-attn --no-build-isolation
```

核心依赖：`torch>=2.1`、`transformers>=4.51`（Qwen3 需 4.51+）、
`trl>=0.12`、`peft>=0.14`、`bitsandbytes>=0.43`（4-bit QLoRA 量化）。

### 2. 准备基座模型

下载 `Qwen/Qwen3-32B` 或 `Qwen/Qwen2.5-7B-Instruct` 到本地，
通过环境变量 `BASE_MODEL` 指定路径（两个训练脚本都会读这个变量，
未设置时回退到脚本内默认相对路径，建议显式传入）：

```bash
export BASE_MODEL=/path/to/Qwen3-32B
```

### 3. 训练

#### 3.1 资源要求

| 模型 | 量化方式 | 最低显存 | 部署方式 |
|------|---------|---------|---------|
| Qwen3-32B | 4-bit QLoRA (nf4) | 约 4×24GB | `device_map="auto"` 跨多卡分片，**单卡 24GB 放不下** |
| Qwen2.5-7B-Instruct | 4-bit QLoRA (nf4) | 约 1×24GB | 脚本固定单卡（`CUDA_VISIBLE_DEVICES=0`） |

> 32B 在 4 张空闲卡上分片训练；7B 单卡即可。若机器只有 1 张卡，仅能跑 7B。

#### 3.2 训练命令

```bash
cd scripts

# Qwen3-32B（多卡分片，4-bit QLoRA）
BASE_MODEL=/path/to/Qwen3-32B CUDA_VISIBLE_DEVICES=2,3,4,6 \
  python train_sft_32b_rerun.py

# Qwen2.5-7B-Instruct（单卡，4-bit QLoRA）
BASE_MODEL=/path/to/Qwen2.5-7B-Instruct \
  python train_sft_7b.py
```

#### 3.3 训练超参

两个脚本使用一致的 LoRA 配置：

| 参数 | 取值 |
|------|------|
| `r` (LoRA rank) | 32 |
| `lora_alpha` | 64（`use_rslora=True`） |
| `lora_dropout` | 0.05 |
| 目标模块 | `q/k/v/o_proj` + `gate/up/down_proj` |
| 学习率 | 5e-5（cosine 调度，warmup 10%） |
| 训练轮数 | 3 epoch（跑满，无 max_steps 截断） |
| `max_length` | 4096 |
| 优化器 | `paged_adamw_8bit` |
| 随机种子 | 42（确定性可复现） |

#### 3.4 数据与字段对齐

训练数据采用 LLaMA-Factory 格式，每条样本含三个字段：

- `instruction`：任务指令（要求模型"严格按指定 JSON 格式输出合规报告"）；
- `input`：待审计的鸿蒙应用代码或功能描述；
- `output`：标准答案，为合规报告 JSON，含 `collectionAndUse` / `permissions` 两个字段。

脚本从 `../data/two_stage_llamafactory/privacy_two_stage_train.json` + `valid.json` 读取，
输出 adapter 到 `../output_twostage_32b_rerun`（或 `output_twostage_7b`）。
**训练脚本路径硬指向 train/valid，永远不会读取 test 文件。**
输出目录首次运行时由 `trainer.save_model()` 自动创建，无需手动建。

> 说明：本项目脚本**直接读取 JSON 文件自行构造 SFT 样本**，不依赖 LLaMA-Factory 框架运行；
> 仓库根 `data/two_stage_llamafactory/dataset_info.json` 仅为便于在 LLaMA-Factory 中可视化/注册而附带。

> 内置 `remap_output()` 将数据中历史字段名 `dataPractices` / `permissionPractices`
> 映射为评测一致的 `collectionAndUse` / `permissions`，确保训练目标与评测判据对齐。

### 4. 评测（测试集，adapter vs base 对照）

```bash
cd scripts
python evaluate_twostage_test.py \
    --adapter ../output_twostage_32b_rerun \
    --eval_file ../data/two_stage_llamafactory/privacy_two_stage_test.json \
    --max_new_tokens 2048
```

指标说明：
- `json_align`：输出为合法 JSON 的比例；
- `has_collectionUse` / `has_permissions`：输出含非空对应字段的比例；
- `key_coverage`：两个核心字段的平均覆盖率；
- `avg_length`：平均输出字符数。

结果写入 `results/results_test_200_32b_rerun.json`，逐条原文见同目录 `*_dump.jsonl`。

### 权重如何获取

模型权重（LoRA adapter）**不入库**（体积大）。请按上述步骤在本地自行生成：
基座模型为 `Qwen/Qwen3-32B`（或 `Qwen2.5-7B-Instruct`），
训练完成后 `trainer.save_model(OUTPUT_DIR)` 产出 LoRA adapter。
如有托管需求可上传至 ModelScope / HuggingFace 并在本文补充链接。

## 数据纪律

- **测试集严格隔离**：`privacy_two_stage_test.json` 仅用于评测，禁止并入训练；
- **禁止数据污染**：后续任何增广数据须先与测试集去重，不得生成与测试集高度雷同的样本；
- 鸿蒙隐私 API 类别有限且可枚举，扩数据时建议按类别穷举覆盖，而非盲目堆量。

详见 `DATA_SPLIT.md`。

## 复现说明

- 所有路径均基于仓库根目录相对推导，clone 后改 `BASE_MODEL` 指向本地基座模型即可运行；
- 测试集严格隔离，禁止并入训练（详见 `DATA_SPLIT.md` 的增广纪律）；
- 训练与评测均使用确定性配置（`do_sample=False`、`seed=42`），结果可复现。
