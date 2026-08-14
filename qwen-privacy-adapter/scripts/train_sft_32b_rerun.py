"""
Qwen3-32B LoRA SFT 重跑脚本（基于 train_sft.py，标准版 + 两处修正）。

★ 相对 train_sft.py 的关键修正（目的：与 7B 链路做公平 head-to-head）：
  1. 加 remap_output()，把训练数据里的 dataPractices/permissionPractices
     映射为评测判据认的 collectionAndUse/permissions，消除字段名漂移假阴性；
  2. 去掉 max_steps=300 截断，跑满 NUM_EPOCHS=3（原版只跑了 ~0.75 epoch）。

其余超参（r=32/alpha=64/lr=5e-5/系统提示/数据）与 train_sft.py 完全一致，
保证只改"字段对齐"与"训练充分度"两个变量。

输出到 <仓库根>/output_twostage_32b_rerun（LoRA adapter 目录）。
启动: BASE_MODEL=/path/to/Qwen3-32B python train_sft_32b_rerun.py
"""

# ★ 32B-4bit 单卡(24GB)放不下，需 device_map="auto" 跨多张空闲卡分片
# （与原版 train_sft.py 一致，原版已成功跑过）。不设 CUDA_VISIBLE_DEVICES，
# 让 auto 自动避开被占的 GPU0/1/5，选用空闲卡(2/3/4/6)。
import os
import json
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ══════════════════════════════════════════════════════════════
# ★ 配置区
# ══════════════════════════════════════════════════════════════

# 仓库根目录（scripts/ 的上一级），其余路径基于此推导，clone 后可直接运行
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 基座模型：必须通过环境变量 BASE_MODEL 显式指定本地路径，例如：
#   export BASE_MODEL=/path/to/Qwen3-32B
# 未设置则给出清晰提示并退出，避免静默使用不存在的默认路径。
if "BASE_MODEL" not in os.environ:
    raise SystemExit(
        "错误：请先设置环境变量 BASE_MODEL 指向本地基座模型路径，例如：\n"
        "  export BASE_MODEL=/path/to/Qwen3-32B\n"
        "或运行：BASE_MODEL=/path/to/Qwen3-32B python train_sft_32b_rerun.py"
    )
MODEL_PATH  = os.environ["BASE_MODEL"]
# 数据统一放在仓库根下的 data/two_stage_llamafactory/（与本目录同级），避免重复存储
DATA_DIR    = os.path.join(REPO_ROOT, "..", "data", "two_stage_llamafactory")
TRAIN_FILE  = os.path.join(DATA_DIR, "privacy_two_stage_train.json")
VAL_FILE    = os.path.join(DATA_DIR, "privacy_two_stage_valid.json")
# adapter 输出到仓库根下的固定目录，基于脚本位置推导，不依赖当前工作目录
OUTPUT_DIR  = os.path.join(REPO_ROOT, "output_twostage_32b_rerun")

SYSTEM_PROMPT = (
    "你是一名专业的代码安全与隐私合规审计专家。"
    "根据静态代码分析工具提供的扫描结果，"
    "先进行业务场景分析，再生成结构化的代码隐私合规分析报告，"
    "报告包含业务场景描述、涉及数据类型、处理方式、权限依赖，"
    "以及完整的隐私政策说明。"
)

LORA_R            = 32
LORA_ALPHA        = 64
NUM_EPOCHS        = 3
BATCH_SIZE        = 1
GRADIENT_ACCUM    = 4
LEARNING_RATE     = 5e-5
MAX_LENGTH        = 4096

# ══════════════════════════════════════════════════════════════


def load_jsonl_or_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    else:
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def remap_output(output: str) -> str:
    """把训练数据里的 dataPractices/permissionPractices 字段名映射为评测认的
    collectionAndUse/permissions，使训练目标与评测判据一致。数据文件不动。"""
    try:
        obj = json.loads(output)
    except Exception:
        return output
    if not isinstance(obj, dict):
        return output
    changed = False
    if "dataPractices" in obj and "collectionAndUse" not in obj:
        obj["collectionAndUse"] = obj.pop("dataPractices")
        changed = True
    if "permissionPractices" in obj and "permissions" not in obj:
        obj["permissions"] = obj.pop("permissionPractices")
        changed = True
    return json.dumps(obj, ensure_ascii=False) if changed else output


def format_sample(example: dict) -> dict:
    if "messages" in example:
        msgs = example["messages"]
        if msgs[0]["role"] != "system":
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
    else:
        instr = (example.get("instruction") or "").strip()
        qry = (example.get("input") or "").strip()
        user_content = f"{instr}\n\n{qry}" if instr else qry
        output = remap_output(example["output"])
        msgs = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": output},
        ]
    text = _tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    return {"text": text}


# ──────────────────────────────────────────────────────────────
# 1. 分词器
# ──────────────────────────────────────────────────────────────
print(">>> [1/4] 加载分词器...")
_tok = AutoTokenizer.from_pretrained(
    MODEL_PATH, trust_remote_code=True, padding_side="right",
)
tokenizer = _tok
if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


# ──────────────────────────────────────────────────────────────
# 2. 数据集
# ──────────────────────────────────────────────────────────────
print(">>> [2/4] 加载并格式化数据集...")

train_raw = load_jsonl_or_json(TRAIN_FILE)
train_dataset = Dataset.from_list(train_raw).map(
    format_sample, remove_columns=Dataset.from_list(train_raw[:1]).column_names
)

eval_dataset = None
if os.path.exists(VAL_FILE):
    val_raw = load_jsonl_or_json(VAL_FILE)
    eval_dataset = Dataset.from_list(val_raw).map(
        format_sample, remove_columns=Dataset.from_list(val_raw[:1]).column_names
    )

print(f"    训练集: {len(train_dataset)} 条")
if eval_dataset:
    print(f"    验证集: {len(eval_dataset)} 条")

sample_text = train_dataset[0]["text"]
print("\n    样本预览（前300字）:")
print(sample_text[:300], "...")
remapped = remap_output(train_raw[0]["output"]) != train_raw[0]["output"]
print(f"    首条 output 已做字段映射(dataPractices->collectionAndUse): {remapped}")


lengths = [len(tokenizer.encode(s["text"])) for s in train_dataset]
print(f"Token 长度统计: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")
print(f"超过 4096 的样本数: {sum(1 for l in lengths if l > 4096)} / {len(lengths)}")


# ──────────────────────────────────────────────────────────────
# 3. 模型
# ──────────────────────────────────────────────────────────────
print("\n>>> [3/4] 加载模型（4-bit QLoRA, 单卡固定）...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ★ 32B-4bit 需跨多卡分片，device_map="auto" 与原版 train_sft.py 一致。
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa",
)
model.enable_input_require_grads()

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_rslora=True,
)


# ──────────────────────────────────────────────────────────────
# 4. 训练
# ──────────────────────────────────────────────────────────────
print(">>> [4/4] 配置 SFTTrainer 并开始训练（跑满 3 epoch）...")
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUM,

    optim="paged_adamw_8bit",
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,

    bf16=True,
    tf32=True,

    max_length=MAX_LENGTH,
    dataset_text_field="text",
    packing=False,

    eval_strategy="steps" if eval_dataset else "no",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    load_best_model_at_end=(eval_dataset is not None),

    logging_steps=1,
    logging_first_step=True,
    report_to="tensorboard",

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    seed=42,
    dataloader_num_workers=4,
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainer.model.print_trainable_parameters()

trainer.train()

print(f"\n>>> 保存模型到 {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(">>> 训练完成。")
