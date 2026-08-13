# 鸿蒙应用代码隐私合规审计 — Qwen LoRA SFT

基于提供的两阶段隐私数据（`privacy_two_stage_*`），用 LoRA 监督微调（SFT）让 Qwen 系列模型
学会输出结构化、可审计的代码隐私合规报告（`collectionAndUse` / `permissions` JSON 字段）。

## 结果速览（无泄漏测试集，200 条）

| 模型 | 微调后 key_coverage | base 对照 key_coverage | 说明 |
|------|--------------------:|----------------------:|------|
| Qwen3-32B (LoRA r=32) | **0.95** | 0.00 | 97.5% 样本产出规范字段 |
| Qwen2.5-7B-IT (LoRA r=32) | **0.97** | 0.01 | 99.5% 样本产出规范字段 |

base 对照 = 关闭 LoRA adapter（`model.disable_adapter()`）后基座模型直接生成，
其合规字段覆盖率接近 0%，佐证提升来自微调学到的格式而非基座自带能力。

> 注：早期版本曾报出 50% 的低指标，根因是**评测字段名与训练目标字段名错配**
> （训练用 `dataPractices`，评测认 `collectionAndUse`），属口径假阴性，非模型问题。
> 详见 `docs/REPORT_TO_ADVISOR_20260813.md`。

## 目录结构

```
github repo/
├── README.md                       # 本文件
├── DATA_SPLIT.md                   # ★ 训练/验证/测试划分 + 不交性校验（落实"不可数据污染"）
├── requirements.txt                # 依赖
├── data/
│   ├── dataset_info.json           # LLaMA-Factory 注册
│   ├── privacy_two_stage_train.json   # 训练集（1600 条）
│   ├── privacy_two_stage_valid.json   # 验证集（200 条）
│   └── privacy_two_stage_test.json    # 测试集（200 条，隔离，仅评测用，禁止并入训练）
├── scripts/
│   ├── train_sft_32b_rerun.py      # 32B 训练（字段 remap + 跑满 3 epoch）
│   ├── train_sft_7b.py             # 7B 训练
│   ├── evaluate_twostage_test.py   # 32B 测试集评测（adapter vs base 对照）
│   └── evaluate_twostage_test_7b.py
├── results/
│   ├── results_test_200_32b_rerun.json
│   └── results_test_200_7b.json
└── docs/
    └── REPORT_TO_ADVISOR_20260813.md  # 口径说明 + 结果留底
```

## 快速开始

### 1. 训练（32B 示例）

```bash
cd scripts
CUDA_VISIBLE_DEVICES=2,3,4,6 \
/mnt/data2/conda/envs/ar_env_py310/bin/python train_sft_32b_rerun.py
```

脚本从 `data/privacy_two_stage_train.json` + `valid.json` 读取，输出 adapter 到 `./output_twostage_32b_rerun`。
**训练脚本路径硬指向 train/valid，永远不会读取 test 文件。**

### 2. 评测（测试集，adapter vs base 对照）

```bash
cd scripts
/mnt/data2/conda/envs/ar_env_py310/bin/python evaluate_twostage_test.py \
    --adapter ./output_twostage_32b_rerun \
    --eval_file /mnt/data2/project/github\ repo/data/privacy_two_stage_test.json \
    --max_new_tokens 2048
```

> 注意：`evaluate_twostage_test.py` 默认 `--adapter` 指向上级目录的 `output_twostage_v2`，
> 重跑结果在 `output_twostage_32b_rerun`，评测时请显式传 `--adapter`。

### 权重如何生成

模型权重（adapter）**不入库**（体积大，GitHub 不适合）。请按上述训练脚本在本地自行生成：
基座模型为 `Qwen/Qwen3-32B`（4-bit QLoRA，r=32, alpha=64, lr=5e-5, 3 epoch），
训练完成后 `trainer.save_model(OUTPUT_DIR)` 产出 LoRA adapter。如有托管需求可上传至
ModelScope / HuggingFace 并在 README 补充链接。

## 数据纪律

- **测试集严格隔离**：`data/privacy_two_stage_test.json` 仅用于评测，禁止并入训练；
- **禁止数据污染**：后续任何增广数据须先与测试集去重，不得生成与测试集高度雷同的样本；
- 鸿蒙隐私 API 类别有限且可枚举，扩数据时建议按类别穷举覆盖，而非盲目堆量。

详见 `DATA_SPLIT.md`。
