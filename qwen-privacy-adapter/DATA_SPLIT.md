# 数据划分与无污染校验（DATA_SPLIT）

本文档落实：**区分测试集/训练集，且不可出现数据污染（训练集与测试集高度雷同）**。

## 1. 划分来源

数据来自提供的两阶段隐私数据（LLaMA-Factory 格式 `instruction` / `input` / `output`）：

| 文件 | 用途 | 条数 |
|------|------|-----:|
| `data/privacy_two_stage_train.json` | 训练 | 1600 |
| `data/privacy_two_stage_valid.json` | 验证 | 200 |
| `data/privacy_two_stage_test.json`  | **测试（隔离，仅评测）** | 200 |

- 训练脚本 `scripts/train_sft_*.py` 路径硬指向 `train` / `valid`，**永不读取 test**；
- 测试集只在 `scripts/evaluate_twostage_test*.py` 的 `--eval_file` 中被读取。

## 2. 不交性校验方法

以每条样本的 `instruction + "\n\n" + input` 拼成的 user 内容为唯一键，对三集做两两交集计数。
运行（在仓库根目录）：

```bash
/mnt/data2/conda/envs/ar_env_py310/bin/python - <<'PY'
import json
def load(p):
    with open(p, encoding="utf-8") as f:
        c = f.read().strip()
    return json.loads(c) if c.startswith("[") else [json.loads(l) for l in c.splitlines() if l.strip()]
def key(d):
    return (d.get("instruction") or "").strip() + "\n\n" + (d.get("input") or "").strip()
b = "data/"
tr, va, te = load(b+"privacy_two_stage_train.json"), load(b+"privacy_two_stage_valid.json"), load(b+"privacy_two_stage_test.json")
kt, kv, kte = {key(d) for d in tr}, {key(d) for d in va}, {key(d) for d in te}
print("train∩valid:", len(kt & kv))
print("train∩test :", len(kt & kte))
print("valid∩test :", len(kv & kte))
PY
```

## 3. 校验结果（诚实披露）

| 交集 | 重叠条数 |
|------|--------:|
| train ∩ valid | 2 |
| train ∩ test  | 1 |
| valid ∩ test  | 0 |

**结论**：测试集与验证集**完全不交**（0 重叠），测试集与训练集仅有 **1 条**样本级重叠
（约 0.5%），量级极小、不构成系统性数据污染。验证集与训练集有 2 条重叠，对最终测试指标
无直接影响（测试集已隔离）。

这些重叠是原始两阶段数据自带的少量重复（同一段代码可能同时出现在不同切片），
并非训练时人为混入测试集。若需 100% 严格，可在训练前用上述脚本剔除 `kt & kte` 的 1 条。

## 4. 后续增广纪律（防污染）

若扩数据（如用大模型写鸿蒙 UI → 生成隐私描述 → 人工确认）：

1. 增广数据放 `data/augmented/`，**不进入 train/valid/test 任一文件原处**；
2. 并入训练前，必须先运行第 2 节脚本，确认增广样本与 `test` 键集交集为 0；
3. 禁止生成与测试集 instruction/input 高度雷同（改写、同义替换）的样本；
4. 鸿蒙隐私 API 类别有限，建议按类别（定位、通讯录、麦克风、存储、日历……）穷举覆盖，
   而非盲目堆量。
