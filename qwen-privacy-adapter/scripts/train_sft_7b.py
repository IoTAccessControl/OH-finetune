"""
Qwen2.5-7B-Instruct LoRA SFT 训练脚本（7B 专用，基于 train_sft.py 改）
复用路径三两阶段数据（privacy_two_stage_train/valid.json）。

★ 相比 32B 版本的关键改动：
  1. BASE_MODEL 指向本地 Qwen2.5-7B-Instruct；
  2. 去掉 enable_thinking（Qwen2.5 chat template 不支持该参数，会被静默忽略）；
  3. 训练 output 字段做映射 dataPractices->collectionAndUse / permissionPractices->permissions，
     使训练目标字段与评测判据（evaluate_twostage_test.py 认 collectionAndUse/permissions）一致。
     数据文件本身不改动；
  4. 去掉 max_steps 截断，跑满 NUM_EPOCHS（7B 小，1600 条 3 epoch 很快）。

启动: python train_sft_7b.py
"""

import os
import json
import re
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

# 基座模型：clone 后请改为本地 Qwen2.5-7B-Instruct 路径（或设环境变量 BASE_MODEL）
MODEL_PATH  = os.environ.get(
    "BASE_MODEL",
    os.path.join(REPO_ROOT, "..", "qwen2.5-7b-instruct"),
)
TRAIN_FILE  = os.path.join(REPO_ROOT, "data", "privacy_two_stage_train.json")
VAL_FILE    = os.path.join(REPO_ROOT, "data", "privacy_two_stage_valid.json")
OUTPUT_DIR  = "./output_twostage_7b"

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
    # Qwen2.5 不支持 enable_thinking，直接拼 chat template（不含 <think> 块）
    text = _tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False,
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
print("\n>>> [3/4] 加载模型（4-bit QLoRA, device_map=auto）...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ★ 单卡固定：7B-4bit 单卡可容纳；device_map="auto" 在多卡下会把层分散到
# 不同 GPU，与 SFTTrainer 的 DataParallel 期望（全在 device_ids[0]）冲突，
# 导致 "found one of them on device: cuda:6" 报错。故固定单卡。
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
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
print(">>> [4/4] 配置 SFTTrainer 并开始训练...")
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
