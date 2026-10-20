#!/usr/bin/env python3
"""
Ollama 推理性能 benchmark（只用标准库，不需要 pip install）

对一组模型 × 一组上下文长度，测：
  - prefill 速度 (prompt tok/s)
  - decode 速度 (generation tok/s)
  - 首 token 延迟 TTFT ≈ load + prompt_eval
  - 冷启动加载时间 (第一次请求)
  - 模型占用的 RAM / VRAM (来自 /api/ps)

用法:
  python ollama_bench.py --label gpu                  # 默认模型矩阵
  python ollama_bench.py --label cpu --models qwen3.5:2b-q4_K_M qwen3.5:4b-q4_K_M
  python ollama_bench.py --label gpu --think          # 开 thinking 模式再跑一遍
  python ollama_bench.py --label gpu --ctx 64 2048 8192 --reps 3

结果写到 results/<label>_<时间戳>.csv 和 .md
"""
import argparse, json, statistics, time, urllib.request, urllib.error, csv, datetime, os, sys, uuid

DEFAULT_MODELS = [
    "qwen3.5:2b-q4_K_M",
    "qwen3.5:2b-q8_0",
    "qwen3.5:4b-q4_K_M",
    "qwen3.5:4b-q8_0",
    "qwen3.5:9b-q4_K_M",
]
DEFAULT_CTX = [64, 2048, 8192]   # 目标 prompt token 数：短 query / 单篇论文 / QA 层多篇
NUM_PREDICT = 128                # 固定生成长度，decode 速度才可比

# 一段论文风格的填充文本，重复到目标长度。1 token ≈ 4 个英文字符，实际 token 数以返回值为准
FILLER = (
    "Retrieval-augmented generation combines a parametric language model with a "
    "non-parametric memory of documents retrieved at inference time. In this work we "
    "study hybrid retrieval that fuses dense embeddings with sparse lexical matching, "
    "followed by a cross-encoder reranking stage, and evaluate citation faithfulness "
    "on a corpus of recent machine learning papers. "
)

def post(host, path, payload, timeout=1800):
    req = urllib.request.Request(host + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def get(host, path):
    with urllib.request.urlopen(host + path, timeout=30) as r:
        return json.loads(r.read())

def build_prompt(target_tokens):
    approx_chars = target_tokens * 4
    body = (FILLER * (approx_chars // len(FILLER) + 1))[:approx_chars]
    return body + "\n\nSummarize the above in three sentences."

def ns_to_s(ns):
    return (ns or 0) / 1e9

def one_run(host, model, prompt, think, num_ctx):
    # 随机前缀：否则 Ollama 会命中上一次的 KV cache，prefill 速度会虚高几十倍
    prompt = f"[run {uuid.uuid4().hex[:8]}]\n" + prompt
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": think,
        "options": {"num_predict": NUM_PREDICT, "num_ctx": num_ctx, "temperature": 0},
    }
    t0 = time.time()
    r = post(host, "/api/generate", payload)
    wall = time.time() - t0
    pe, pd = r.get("prompt_eval_count", 0), ns_to_s(r.get("prompt_eval_duration"))
    ec, ed = r.get("eval_count", 0), ns_to_s(r.get("eval_duration"))
    return {
        "wall_s": wall,
        "load_s": ns_to_s(r.get("load_duration")),
        "prompt_tokens": pe,
        "prefill_tok_s": pe / pd if pd > 0 else 0,
        "gen_tokens": ec,
        "decode_tok_s": ec / ed if ed > 0 else 0,
        "ttft_s": ns_to_s(r.get("load_duration")) + pd,
    }

def wait_ready(host, max_wait=90):
    """服务崩溃后 docker 会自动拉起，这里轮询直到 /api/tags 能响应"""
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            get(host, "/api/tags"); return True
        except Exception:
            time.sleep(3)
    return False

def safe_run(host, model, prompt, think, num_ctx):
    """一次请求；连接断了说明服务崩了，记录下来并等待恢复"""
    try:
        return one_run(host, model, prompt, think, num_ctx), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode(errors='ignore')[:200]}"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"  !! 服务可能崩溃 ({err})，等待恢复...", flush=True)
        wait_ready(host)
        return None, err

def mem_info(host, model):
    try:
        for m in get(host, "/api/ps").get("models", []):
            if m.get("name") == model or m.get("model") == model:
                return m.get("size", 0) / 2**30, m.get("size_vram", 0) / 2**30
    except Exception:
        pass
    return 0.0, 0.0

def unload(host, model):
    try:
        post(host, "/api/generate", {"model": model, "keep_alive": 0}, timeout=60)
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--label", required=True, help="cpu / gpu / native 等，写进结果文件名")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--ctx", nargs="+", type=int, default=DEFAULT_CTX)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "results"))
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    tag = f"{a.label}{'_think' if a.think else ''}_{stamp}"
    rows = []

    for model in a.models:
        print(f"\n=== {model} ===", flush=True)
        unload(a.host, model)
        # 冷启动：测一次加载时间
        cold, err = safe_run(a.host, model, "hi", a.think, 2048)
        if cold is None:
            print(f"  跳过 {model}: {err}")
            rows.append({"label": a.label, "think": a.think, "model": model, "target_ctx": -1,
                         "prompt_tokens": 0, "prefill_tok_s": 0, "decode_tok_s": 0, "ttft_s": 0,
                         "cold_load_s": 0, "mem_total_gb": 0, "mem_vram_gb": 0, "reps": 0,
                         "error": err[:120]})
            continue
        print(f"  cold load {cold['load_s']:.1f}s")
        for ctx in a.ctx:
            prompt = build_prompt(ctx)
            num_ctx = max(2048, ctx + NUM_PREDICT + 256)
            runs = []
            for i in range(a.reps):
                r, err = safe_run(a.host, model, prompt, a.think, num_ctx)
                if r is None:
                    print(f"  ctx={ctx} rep{i} 失败: {err}")
                    break
                runs.append(r)
                print(f"  ctx~{ctx} rep{i}: prompt={r['prompt_tokens']} "
                      f"prefill={r['prefill_tok_s']:.0f} tok/s decode={r['decode_tok_s']:.1f} tok/s "
                      f"ttft={r['ttft_s']:.2f}s", flush=True)
            if not runs:
                continue
            ram, vram = mem_info(a.host, model)
            med = lambda k: statistics.median(x[k] for x in runs)
            rows.append({
                "label": a.label, "think": a.think, "model": model,
                "target_ctx": ctx, "prompt_tokens": int(med("prompt_tokens")),
                "prefill_tok_s": round(med("prefill_tok_s"), 1),
                "decode_tok_s": round(med("decode_tok_s"), 2),
                "ttft_s": round(med("ttft_s"), 3),
                "cold_load_s": round(cold["load_s"], 2),
                "mem_total_gb": round(ram, 2), "mem_vram_gb": round(vram, 2),
                "reps": len(runs), "error": "",
            })
        unload(a.host, model)

    if not rows:
        print("没有任何结果"); sys.exit(1)
    csv_path = os.path.join(a.outdir, f"{tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    md_path = os.path.join(a.outdir, f"{tag}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Ollama benchmark — {a.label}{' (thinking)' if a.think else ''} — {stamp}\n\n")
        f.write("| model | prompt tok | prefill tok/s | decode tok/s | TTFT s | cold load s | mem GB | VRAM GB |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            if r.get("error"):
                f.write(f"| {r['model']} | — | — | — | — | — | — | 崩溃: {r['error']} |\n"); continue
            f.write(f"| {r['model']} | {r['prompt_tokens']} | {r['prefill_tok_s']} | {r['decode_tok_s']} "
                    f"| {r['ttft_s']} | {r['cold_load_s']} | {r['mem_total_gb']} | {r['mem_vram_gb']} |\n")
    print(f"\n写入 {csv_path}\n写入 {md_path}")

if __name__ == "__main__":
    main()
