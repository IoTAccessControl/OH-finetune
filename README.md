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

### 环境准备

```bash
pip install -r qwen-privacy-adapter/requirements.txt
# 可选，显著降低显存占用并提速：
pip install flash-attn --no-build-isolation
```

### 基座模型

- `Qwen/Qwen3-32B`（推荐，效果更好）
- `Qwen/Qwen2.5-7B-Instruct`（资源受限时可用）

需提前下载基座模型权重到本地，训练时通过环境变量 `BASE_MODEL` 指定路径。

### 训练

```bash
# Qwen3-32B（多卡，4-bit QLoRA）
cd qwen-privacy-adapter/scripts
CUDA_VISIBLE_DEVICES=2,3,4,6 python train_sft_32b_rerun.py

# Qwen2.5-7B（单卡即可）
python train_sft_7b.py
```

训练完成后 LoRA adapter 保存在 `qwen-privacy-adapter/output_twostage_32b_rerun/` 或 `output_twostage_7b/`。

### 评测

```bash
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

评测在同一份 200 条测试集上分别跑「微调后 adapter」和「关闭 adapter 的 base 模型」，做对照打分。

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
