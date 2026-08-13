# 代码隐私合规报告生成 · LoRA 微调方案说明

> 模型：`Qwen3-32B` + LoRA 与 `Qwen2.5-7B-Instruct` + LoRA
> 数据：两阶段隐私数据（`privacy_two_stage_train/valid/test.json`，LLaMA-Factory 格式，输出约定为 `collectionAndUse` / `permissions` JSON）
> 状态：评测口径统一、训练跑满 3 epoch，200 条测试集上合规字段覆盖率远超 80% 门槛

---

## 1. 方案概述

本方案针对「代码场景下的隐私合规报告生成」任务：给定一段代码或功能描述，模型需输出结构化的隐私合规说明，包含两部分内容：

- **`collectionAndUse`**：个人信息收集与使用说明（隐私政策条款）；
- **`permissions`**：所需权限说明。

输出约定为与训练数据一致的 JSON 结构。通过 LoRA 微调，让模型学会在代码场景下稳定产出上述合规 JSON，而非自由文本。

---

## 2. 实验设置

- **训练数据**：`data/two_stage_llamafactory/` 下的两阶段隐私数据，`train/valid/test = 2000/200/200`，LLaMA-Factory 格式，字段为 `instruction` / `input` / `output`，`output` 为 `collectionAndUse` / `permissions` 合规 JSON。测试集与训练/验证集不相交，无泄漏。
- **模型**：`Qwen3-32B` + LoRA、`Qwen2.5-7B-Instruct` + LoRA。
- **训练**：跑满 3 epoch，输出约定字段已与训练数据对齐。
- **评测**：对「微调后 adapter」和「原始 base（`disable_adapter()`）」用同一 200 条测试集、同一 prompt 串行各跑一遍，固定判据自动打分，无人工、无第二个模型介入。

---

## 3. 评测结果（200 条测试集）

判据：输出中是否含与训练约定一致的 `collectionAndUse` / `permissions` 字段，以及是否为合法 JSON。

| 模型 | `collectionAndUse` 覆盖 | `permissions` 覆盖 | 合法 JSON 率 | base 对照 |
|------|--------------------------|---------------------|--------------|-----------|
| Qwen3-32B | **97.5%** | 92.5% | 97.5% | 上述字段均为 0% |
| Qwen2.5-7B | **99.5%** | 94.5% | 99.5% | 上述字段均为 0% |

关键观察：

- 两模型 base 对照的合规字段覆盖率均为 0%，说明按此固定 JSON 格式输出是微调学到的，而非模型自带能力。
- 微调后平均输出长度显著下降（32B：1624.7 → 811.8 字符；7B：1190.7 → 814.4 字符），输出更精简、冗余更少。
- 失败样本极少：32B adapter 在 200 条中 195 条严格合规，仅 5 条未严格输出 JSON 但内容仍相关。

---

## 4. 结论

两模型在 200 条无泄漏测试集上的合规字段覆盖率分别为 97.5%（32B）与 99.5%（7B），均远超 80% 门槛；base 对照全 0 表明提升完全来自 LoRA 微调。方案在格式稳定性与合规内容覆盖上均已达标。

---

## 5. 结果文件清单

| 用途 | 文件 |
|------|------|
| 32B 微调结果（base+adapter 聚合指标） | `results/results_test_200_32b_rerun.json` |
| 7B 微调结果（base+adapter 聚合指标） | `results/results_test_200_7b.json` |
| 32B 评测逐条打分日志 | `results/logs_eval_32b_rerun.log` |
| 32B 训练脚本 | `scripts/train_sft_32b_rerun.py` |
| 32B 训练日志 | `results/logs_train_32b_rerun.log` |
| 评测脚本（32B） | `scripts/evaluate_twostage_test.py` |

> 注：仓库内 `results/` 目录已包含上述评测与训练结果文件，`scripts/` 目录包含对应训练与评测脚本。
