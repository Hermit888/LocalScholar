#!/usr/bin/env python3
"""
thinking 模式的真实代价：同一个 QA 风格 prompt，think 开/关各跑 N 次，不限生成长度。
测总耗时、思考 token 数、回答 token 数。ollama_bench.py 固定生成 128 token，测不出这个。

用法: python think_cost.py --models qwen3.5:4b-q4_K_M qwen3.5:2b-q4_K_M --reps 3
"""
import argparse, json, statistics, time, urllib.request, uuid, os, datetime

ABSTRACTS = [
    ("Paper A", "We propose a hybrid retrieval pipeline that fuses BM25 lexical scores with dense "
     "SPECTER2 embeddings via reciprocal rank fusion, followed by a cross-encoder reranker. On a "
     "corpus of 40k arXiv cs.AI abstracts, hybrid retrieval improves recall@10 by 12 points over "
     "dense-only retrieval, with the largest gains on queries containing rare technical terms."),
    ("Paper B", "This work studies citation faithfulness in retrieval-augmented generation. We attach "
     "a lightweight NLI model that checks whether each generated sentence is entailed by its cited "
     "passage, and filter unsupported sentences. Human evaluation shows hallucinated claims drop from "
     "18% to 6% at the cost of 9% shorter answers."),
    ("Paper C", "We benchmark small quantized language models (2B-9B parameters) for local RAG on "
     "consumer hardware. Q4 quantization retains 97% of answer quality while halving memory; hybrid "
     "linear-attention architectures keep decode speed flat up to 8k context, unlike pure attention "
     "models whose KV cache dominates memory at long context."),
]
QUESTION = ("Compare the three papers: what problem does each address, and how could their methods be "
            "combined into a single local paper-QA system? Cite papers as [A], [B], [C].")

def build_prompt():
    parts = [f"[run {uuid.uuid4().hex[:8]}]", "You are answering questions about research papers. "
             "Use only the provided abstracts and cite them.\n"]
    for name, text in ABSTRACTS:
        parts.append(f"### {name}\n{text}\n")
    parts.append(f"### Question\n{QUESTION}")
    return "\n".join(parts)

def run(host, model, think, num_predict):
    payload = {"model": model, "prompt": build_prompt(), "stream": False, "think": think,
               "options": {"num_predict": num_predict, "num_ctx": 8192, "temperature": 0}}
    req = urllib.request.Request(host + "/api/generate", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    thinking = d.get("thinking", "") or ""
    answer = d.get("response", "") or ""
    return {
        "wall_s": round(time.time() - t0, 2),
        "prompt_tokens": d.get("prompt_eval_count", 0),
        "gen_tokens": d.get("eval_count", 0),
        "thinking_chars": len(thinking),
        "answer_chars": len(answer),
        "decode_tok_s": round(d.get("eval_count", 0) / (d.get("eval_duration", 1) / 1e9), 1),
        "truncated": d.get("done_reason") == "length",
        "answer_preview": answer.strip().replace("\n", " ")[:160],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--models", nargs="+", default=["qwen3.5:4b-q4_K_M"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--num-predict", type=int, default=4096)
    ap.add_argument("--label", default="gpu")
    a = ap.parse_args()

    out = []
    for model in a.models:
        for think in (False, True):
            runs = []
            for i in range(a.reps):
                r = run(a.host, model, think, a.num_predict)
                runs.append(r)
                print(f"{model} think={think} rep{i}: wall={r['wall_s']}s gen={r['gen_tokens']} tok "
                      f"decode={r['decode_tok_s']} tok/s thinking_chars={r['thinking_chars']}"
                      f"{' TRUNCATED' if r['truncated'] else ''}", flush=True)
            med = lambda k: statistics.median(x[k] for x in runs)
            out.append({"model": model, "think": think, "wall_s": med("wall_s"),
                        "gen_tokens": int(med("gen_tokens")), "thinking_chars": int(med("thinking_chars")),
                        "answer_chars": int(med("answer_chars")), "decode_tok_s": med("decode_tok_s"),
                        "any_truncated": any(x["truncated"] for x in runs),
                        "answer_preview": runs[-1]["answer_preview"]})

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(os.path.dirname(__file__), "results", f"think_{a.label}_{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# thinking cost - {a.label} - {stamp}\n\n")
        f.write("| model | think | total s | gen tok | thinking chars | answer chars | decode tok/s | truncated |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---|\n")
        for r in out:
            f.write(f"| {r['model']} | {r['think']} | {r['wall_s']} | {r['gen_tokens']} | {r['thinking_chars']} "
                    f"| {r['answer_chars']} | {r['decode_tok_s']} | {r['any_truncated']} |\n")
        f.write("\n## 回答预览\n")
        for r in out:
            f.write(f"- **{r['model']} think={r['think']}**: {r['answer_preview']}…\n")
    print(f"\n写入 {path}")

if __name__ == "__main__":
    main()
