"""
evaluate_twostage_test.py —— 测试集正式评测 + 失败样本归因（P1 + P2 合一）。

★ 背景：
  导师提供的两阶段数据（privacy_two_stage_train/valid/test.json，LLaMA-Factory 格式，
  instruction 要求"严格按指定 JSON 格式输出"，output 为
  {"collectionAndUse":[...], "permissions":[...]}）。
  模型 output_twostage_v2 已用 train(1600)+valid(200) 微调完成。

★ 本脚本做的事：
  1. 在 200 条测试集（privacy_two_stage_test.json）上评测 adapter 与 base 对照；
  2. 指标：json_align / has_collectionUse / has_permissions / key_coverage / 平均条目数 / 长度；
  3. 逐条保存原文到 results_test_200_7b_dump.jsonl，并对"未产出规范 collectionAndUse"的样本
     做失败类型分类（长样本 / 字段名漂移 / 自由文本 / 空），支撑 P2 归因。

★ 判据对齐训练数据格式（collectionAndUse JSON），非 Markdown。

用法（在仓库 scripts/ 目录下）：
  CUDA_VISIBLE_DEVICES=1,2,3,4,6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  nohup /mnt/data2/conda/envs/ar_env_py310/bin/python \\
      evaluate_twostage_test_7b.py \\
      --adapter ./output_twostage_7b \\
      --eval_file ../data/privacy_two_stage_test.json \\
      --max_new_tokens 2048 > logs_eval_twostage_test.log 2>&1 &
"""
import argparse
import json
import os
import re
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

# 仓库根目录（scripts/ 的上一级）
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 基座模型：clone 后请改为本地 Qwen2.5-7B-Instruct 路径（或设环境变量 BASE_MODEL）
BASE_MODEL = os.environ.get(
    "BASE_MODEL",
    os.path.join(REPO_ROOT, "..", "qwen2.5-7b-instruct"),
)
EVAL_FILE_DEFAULT = os.path.join(REPO_ROOT, "data", "privacy_two_stage_test.json")

SYSTEM_PROMPT = (
    "你是一名专业的代码安全与隐私合规审计专家。"
    "根据静态代码分析工具提供的扫描结果，"
    "先进行业务场景分析，再生成结构化的代码隐私合规分析报告，"
    "报告包含业务场景描述、涉及数据类型、处理方式、权限依赖，"
    "以及完整的隐私政策说明。"
)

CORE_KEYS = ["collectionAndUse", "permissions"]


def safe_json_load(text: str):
    t = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", t, re.DOTALL | re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t), True
    except Exception:
        s, e = t.find("{"), t.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(t[s:e + 1]), True
            except Exception:
                return None, False
        return None, False


def is_nonempty_list(v):
    return isinstance(v, list) and len(v) > 0


def classify_failure(text: str, has_cu: int) -> str:
    """对未产出规范 collectionAndUse 的样本做失败类型分类。"""
    if has_cu == 1:
        return "ok"
    obj, ok = safe_json_load(text)
    if ok and isinstance(obj, dict):
        # 是 JSON 但 collectionAndUse 缺失/为空 → 字段漂移
        if "collectionAndUse" in obj or "permissions" in obj:
            return "field_drift"   # 有 JSON 外壳但字段名不对/为空
        return "json_other"        # 是别的 JSON 结构
    # 不是合法 JSON
    if len(text.strip()) < 30:
        return "empty_or_short"
    if len(text) > 1200:
        return "long_degenerate"   # 长样本退化成自由文本
    return "free_text"             # 中等长度自由文本


def score(text: str):
    obj, ok = safe_json_load(text)
    if ok and isinstance(obj, dict):
        has_cu = 1 if is_nonempty_list(obj.get("collectionAndUse")) else 0
        has_perm = 1 if is_nonempty_list(obj.get("permissions")) else 0
        cov = sum(1 for k in CORE_KEYS if k in obj and is_nonempty_list(obj.get(k))) / len(CORE_KEYS)
        cu_items = len(obj["collectionAndUse"]) if is_nonempty_list(obj.get("collectionAndUse")) else 0
        perm_items = len(obj["permissions"]) if is_nonempty_list(obj.get("permissions")) else 0
    else:
        has_cu = 0
        has_perm = 0
        cov = 0.0
        cu_items = 0
        perm_items = 0
    fail_type = classify_failure(text, has_cu)
    return ok, has_cu, has_perm, cov, cu_items, perm_items, fail_type, len(text)


def load_samples(path: str):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        data = json.loads(content)
    else:
        data = [json.loads(l) for l in content.splitlines() if l.strip()]
    # 拼成 user 内容（与 train_sft.py 一致：instruction + input）
    out = []
    for d in data:
        instr = (d.get("instruction") or "").strip()
        qry = (d.get("input") or "").strip()
        user_content = f"{instr}\n\n{qry}" if instr else qry
        out.append({"user": user_content, "output_ref": d.get("output", "")})
    return out


def build_model(adapter_path: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f">>> 加载 base 模型 (4bit) from {BASE_MODEL}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True, attn_implementation="sdpa",
    )
    if adapter_path:
        print(f">>> 加载 adapter from {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=False, temperature=1.0, top_p=1.0,
    )
    return tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def eval_model(model, tokenizer, samples, tag, max_new_tokens, use_adapter: bool):
    rows = []
    aligns, cu, perm, covs, cu_items, perm_items, lens = [], [], [], [], [], [], []
    fail_counter = {}
    for i, s in enumerate(samples):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": s["user"]},
        ]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        if use_adapter:
            text = generate(model, tokenizer, prompt, max_new_tokens)
        else:
            with model.disable_adapter():
                text = generate(model, tokenizer, prompt, max_new_tokens)
        a, hc, hp, c, ci, pi, ft, l = score(text)
        aligns.append(a); cu.append(hc); perm.append(hp); covs.append(c)
        cu_items.append(ci); perm_items.append(pi); lens.append(l)
        fail_counter[ft] = fail_counter.get(ft, 0) + 1
        rows.append({
            "idx": i, "tag": tag, "json_align": a,
            "has_collectionUse": hc, "has_permissions": hp,
            "key_coverage": c, "cu_items": ci, "perm_items": pi,
            "fail_type": ft, "len": l, "output": text,
        })
        print(f"  [{tag}] 样本{i}: json_align={a} cu={hc} perm={hp} "
              f"key_cov={c:.2f} fail={ft} len={l}", flush=True)
    return rows, aligns, cu, perm, covs, cu_items, perm_items, lens, fail_counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter",
                    default="/mnt/data2/project/qwen微调 - 副本/qwen微调/output_twostage_7b")
    ap.add_argument("--eval_file", default=EVAL_FILE_DEFAULT)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = load_samples(args.eval_file)
    print(f">>> 评估输入: {len(samples)} 条 (来自 {args.eval_file})", flush=True)

    model = build_model(args.adapter)
    n = len(samples)

    # adapter
    print("\n>>> 评估 adapter ...", flush=True)
    rows_a, a, cu, perm, c, ci, pi, l, fa = eval_model(
        model, tokenizer, samples, "adapter", args.max_new_tokens, use_adapter=True)
    # base 对照
    print("\n>>> 评估 base-only ...", flush=True)
    rows_b, ba, bcu, bperm, bc, bci, bpi, bl, fb = eval_model(
        model, tokenizer, samples, "base", args.max_new_tokens, use_adapter=False)

    def agg(arr): return sum(arr) / len(arr)
    results = {
        "adapter": {
            "json_align": agg(a), "has_collectionUse": agg(cu),
            "has_permissions": agg(perm), "key_coverage": agg(c),
            "avg_cu_items": agg(ci), "avg_perm_items": agg(pi),
            "avg_length": agg(l), "n": n,
            "fail_type_counts": fa,
        },
        "base": {
            "json_align": agg(ba), "has_collectionUse": agg(bcu),
            "has_permissions": agg(bperm), "key_coverage": agg(bc),
            "avg_cu_items": agg(bci), "avg_perm_items": agg(bpi),
            "avg_length": agg(bl), "n": n,
            "fail_type_counts": fb,
        },
    }

    print("\n" + "=" * 60)
    print(f"=== 两阶段测试集评测（{n} 条）===")
    cols = ["json_align", "has_collectionUse", "has_permissions",
            "key_coverage", "avg_cu_items", "avg_perm_items", "avg_length"]
    print(f"{'指标':<20}" + "".join(f"{k:>18}" for k in cols))
    print(f"{'base':<20}" + "".join(f"{results['base'][k]:>18.3f}" for k in cols[:-1]) +
          f"{results['base']['avg_length']:>18.1f}")
    print(f"{'adapter':<20}" + "".join(f"{results['adapter'][k]:>18.3f}" for k in cols[:-1]) +
          f"{results['adapter']['avg_length']:>18.1f}")
    print("--- 失败类型分布 ---")
    print("adapter:", fa)
    print("base   :", fb)
    print("=" * 60)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, "results_test_200_7b.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    # 逐条原文 dump（合并 adapter + base）
    dump = []
    for ra, rb in zip(rows_a, rows_b):
        dump.append({"idx": ra["idx"], "adapter": ra, "base": rb})
    with open(os.path.join(out_dir, "results_test_200_7b_dump.jsonl"), "w", encoding="utf-8") as f:
        for d in dump:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f">>> 结果已写 results_test_200_7b.json + results_test_200_7b_dump.jsonl")


if __name__ == "__main__":
    main()
