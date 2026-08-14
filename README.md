# OH-finetune — 鸿蒙应用代码隐私合规报告生成（Qwen LoRA 微调）

基于 Qwen 系列大模型，通过 LoRA 监督微调（SFT），实现：给定一段鸿蒙（OpenHarmony）应用代码或功能描述，自动输出结构化的**代码隐私合规报告**（JSON 格式）。

## 项目结构

```
OH-finetune/
├── README.md                              # 本文件
├── data/two_stage_llamafactory/           # 训练/验证/测试数据（LLaMA-Factory 格式）
│   ├── dataset_info.json                  # 数据集注册信息
│   ├── privacy_two_stage_train.json       # 训练集（1600 条）
│   ├── privacy_two_stage_valid.json       # 验证集（200 条）
│   └── privacy_two_stage_test.json        # 测试集（200 条，仅评测用）
└── qwen-privacy-adapter/                 # 微调代码与评测脚本
    ├── README.md                          # 详细使用说明
    ├── DATA_SPLIT.md                      # 数据划分与不交性校验
    ├── requirements.txt                   # Python 依赖
    ├── scripts/
    │   ├── train_sft_32b_rerun.py         # Qwen3-32B 训练脚本
    │   ├── train_sft_7b.py                # Qwen2.5-7B 训练脚本
    │   ├── evaluate_twostage_test.py      # 32B 评测脚本
    │   └── evaluate_twostage_test_7b.py   # 7B 评测脚本
    ├── results/                           # 评测结果
    └── docs/
        └── REPORT.md                      # 方案说明与结果留底
```

## 快速开始

### 1. 环境准备

```bash
pip install -r qwen-privacy-adapter/requirements.txt
# 可选，显著降低显存占用并提速：
pip install flash-attn --no-build-isolation
```

核心依赖：`torch>=2.1`、`transformers>=4.51`（Qwen3 需 4.51+）、`trl>=0.12`、`peft>=0.14`、`bitsandbytes>=0.43`（4-bit QLoRA 量化）。

### 2. 基座模型

- `Qwen/Qwen3-32B`（推荐，效果更好）
- `Qwen/Qwen2.5-7B-Instruct`（资源受限时可用）

需提前下载基座模型权重到本地，**训练/评测时通过环境变量 `BASE_MODEL` 显式指定路径**。脚本未设置该变量会直接报错退出（避免静默使用不存在的默认路径）。例如：

```bash
export BASE_MODEL=/path/to/Qwen3-32B
```

### 3. 训练数据是怎么来的

无论哪种方案，训练目标都是让模型"根据静态代码审计扫描结果 → 输出结构化隐私合规报告"。数据以 **LLaMA-Factory 三字段格式** 存放在 `data/two_stage_llamafactory/`：

| 字段 | 含义 |
|------|------|
| `instruction` | 任务指令（要求模型"严格按指定 JSON 格式输出合规报告"） |
| `input` | 待审计的鸿蒙应用代码 / 功能描述 / 静态扫描结果（JSON） |
| `output` | 标准答案：合规报告 JSON，含 `collectionAndUse`、`permissions` 两个字段 |

训练脚本**直接读取 JSON 文件自行构造 SFT 样本**（不依赖 LLaMA-Factory 框架运行）；仓库根下的 `dataset_info.json` 仅为方便在 LLaMA-Factory 里可视化/注册而附带。

### 4. 训练（LoRA SFT 细节）

本仓库用 **4-bit QLoRA + LoRA** 做监督微调，关键设计如下：

- **量化**：`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=bf16, bnb_4bit_use_double_quant=True)`，基座以 4-bit 加载，显存占用大幅降低；训练前调用 `model.enable_input_require_grads()`（量化模型必需，否则报 device 不匹配）。
- **LoRA 配置（两个脚本一致）**：`r=32`、`lora_alpha=64`、`use_rslora=True`、`lora_dropout=0.05`、`bias="none"`，目标模块覆盖全部线性层 `q/k/v/o_proj` + `gate/up/down_proj`。
- **训练超参**：`learning_rate=5e-5`（cosine 调度 + 10% warmup）、`num_train_epochs=3`（跑满，无 max_steps 截断）、`per_device_train_batch_size=1` + `gradient_accumulation_steps=4`（等效 batch 4）、`max_length=4096`、`optim="paged_adamw_8bit"`、`seed=42`（确定性可复现）、`gradient_checkpointing=True`。
- **样本构造**：脚本用 `tokenizer.apply_chat_template` 把 `system + user(input) + assistant(output)` 拼成训练文本，并 `enable_thinking=False`（Qwen3 会在模板里插入空 `<think>` 块、直接出答案；Qwen2.5 的 chat template 不支持该参数、会被静默忽略，但同样不含 `<think>` 内容）。
- **字段对齐**：内置 `remap_output()` 把历史字段名 `dataPractices` / `permissionPractices` 映射为评测认的 `collectionAndUse` / `permissions`，确保训练目标与评测判据一致（数据文件本身不动）。
- **数据纪律**：训练脚本路径**硬指向 `train` / `valid` 文件，永不读取 `test`**，测试集严格隔离。

#### 4.1 资源要求

| 模型 | 量化方式 | 最低显存 | 部署方式 |
|------|---------|---------|---------|
| Qwen3-32B | 4-bit QLoRA (nf4) | 约 4×24GB | `device_map="auto"` 跨多卡分片，**单卡 24GB 放不下** |
| Qwen2.5-7B-Instruct | 4-bit QLoRA (nf4) | 约 1×24GB | 脚本固定单卡（`CUDA_VISIBLE_DEVICES=0`） |

> 32B 在空闲卡上分片训练；7B 单卡即可。若机器只有 1 张卡，仅能跑 7B。

#### 4.2 训练命令

```bash
cd qwen-privacy-adapter/scripts

# Qwen3-32B（多卡分片，4-bit QLoRA）
BASE_MODEL=/path/to/Qwen3-32B CUDA_VISIBLE_DEVICES=2,3,4,6 \
  python train_sft_32b_rerun.py

# Qwen2.5-7B-Instruct（单卡，4-bit QLoRA）
BASE_MODEL=/path/to/Qwen2.5-7B-Instruct \
  python train_sft_7b.py
```

训练完成后 LoRA adapter 自动保存到 `qwen-privacy-adapter/output_twostage_32b_rerun/`（或 `output_twostage_7b/`）。

### 5. 评测（adapter vs base 对照）

```bash
cd qwen-privacy-adapter/scripts

# 32B
python evaluate_twostage_test.py \
    --adapter ../output_twostage_32b_rerun \
    --eval_file ../data/two_stage_llamafactory/privacy_two_stage_test.json \
    --max_new_tokens 2048

# 7B
python evaluate_twostage_test_7b.py \
    --adapter ../output_twostage_7b \
    --eval_file ../data/two_stage_llamafactory/privacy_two_stage_test.json \
    --max_new_tokens 2048
```

评测在同一份 200 条测试集上分别跑「微调后 adapter」和「关闭 adapter 的 base 模型」（`model.disable_adapter()` 切换回基座，共享同一份权重、不占双份显存），做对照打分。主要指标：

- `json_align`：输出为合法 JSON 的比例；
- `has_collectionUse` / `has_permissions`：输出含非空对应字段的比例；
- `key_coverage`：两个核心字段的平均覆盖率；
- `avg_length`：平均输出字符数。

结果写入 `qwen-privacy-adapter/results/`，逐条原文见同目录 `*_dump.jsonl`。

详细说明请参阅 [qwen-privacy-adapter/README.md](./qwen-privacy-adapter/README.md)。

## 初步效果

200 条无泄漏测试集（与训练/验证集不相交）：

| 模型 | `collectionAndUse` 覆盖 | `permissions` 覆盖 | 合法 JSON 率 | base 对照 |
|------|------------------------:|-------------------:|-------------:|-----------|
| Qwen3-32B (LoRA) | **97.5%** | 92.5% | 97.5% | ~0% |
| Qwen2.5-7B (LoRA) | **99.5%** | 94.5% | 99.5% | ~0% |

- base 对照的合规字段覆盖率接近 0%，说明按固定 JSON 格式输出是**微调学到的能力**；
- 微调后平均输出长度显著下降（32B：1624 → 812 字符；7B：1191 → 814 字符），冗余更少。

## 数据说明

数据采用 LLaMA-Factory 格式（`instruction` / `input` / `output`），`output` 为合规报告 JSON，包含两个字段：

- `collectionAndUse`：个人信息收集与使用说明
- `permissions`：应用所需权限说明

| 文件 | 用途 | 条数 |
|------|------|-----:|
| `privacy_two_stage_train.json` | 训练 | 1600 |
| `privacy_two_stage_valid.json` | 验证 | 200 |
| `privacy_two_stage_test.json` | 测试（隔离，禁止并入训练） | 200 |

测试集严格隔离，训练脚本永不读取 test 文件。详见 [DATA_SPLIT.md](./qwen-privacy-adapter/DATA_SPLIT.md)。

## License

MIT
