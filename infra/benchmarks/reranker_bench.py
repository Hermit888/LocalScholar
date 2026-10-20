#!/usr/bin/env python3
"""
Reranker 延迟 benchmark：MiniLM vs bge-reranker-base，fp32 vs int8，CPU vs GPU。

回答的问题：bge-reranker-base（278M）在 CPU 上给 50 个候选打分要多久？
量化成 int8 能快多少？有 GPU 时还是不是问题？

用法:
  pip install optimum[onnxruntime]        # int8 量化需要，没装会跳过 int8
  python infra/benchmarks/reranker_bench.py
  python infra/benchmarks/reranker_bench.py --n-candidates 50 --reps 3

结果写到 results/reranker_<时间戳>.md
"""
import argparse, time, statistics, os, datetime, random, sys

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",   # 22M，当前使用
    "BAAI/bge-reranker-base",                   # 278M，候选
]
QUERY = "medical image recognition with neural networks"
BATCH = 16
MAX_LEN = 512

# 造 50 篇长度不一的"摘要"（100-300 词），模拟真实候选
_WORDS = ("neural network convolutional segmentation medical imaging lesion detection "
          "transformer attention dataset annotation radiology classification accuracy "
          "retrieval augmented generation benchmark evaluation diagnosis clinical model "
          "training inference latency quantization embedding sparse dense hybrid").split()

def make_docs(n, seed=0):
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        length = rng.randint(100, 300)
        title = f"Paper {i}: " + " ".join(rng.choice(_WORDS) for _ in range(6))
        body = " ".join(rng.choice(_WORDS) for _ in range(length))
        docs.append(f"{title}. {body}")
    return docs

def score_torch(model, tok, docs, device):
    out = []
    for s in range(0, len(docs), BATCH):
        batch = docs[s:s+BATCH]
        feats = tok([QUERY]*len(batch), batch, padding=True, truncation=True,
                    max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.no_grad():
            out.extend(model(**feats).logits.squeeze(-1).tolist())
    return out

def bench(fn, reps):
    fn()  # warmup
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-candidates", type=int, default=50)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "results"))
    a = ap.parse_args()

    docs = make_docs(a.n_candidates)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    print(f"devices: {devices}  candidates: {len(docs)}  torch {torch.__version__}")
    rows = []

    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        have_onnx = True
    except ImportError:
        have_onnx = False
        print("optimum[onnxruntime] 未安装，跳过 int8；pip install optimum[onnxruntime] 可补")

    for name in MODELS:
        print(f"\n=== {name}")
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        params = sum(p.numel() for p in model.parameters()) / 1e6

        for dev in devices:
            m = model.to(dev)
            t = bench(lambda: score_torch(m, tok, docs, dev), a.reps)
            vram = torch.cuda.max_memory_allocated()/2**30 if dev == "cuda" else 0
            rows.append((name, "fp32", dev, params, t, vram))
            print(f"  fp32 {dev:4s}: {t:6.2f} s  ({params:.0f}M params{f', {vram:.2f} GB VRAM' if vram else ''})")
            if dev == "cuda": torch.cuda.reset_peak_memory_stats()
        model.to("cpu")

        if have_onnx:
            try:
                onnx_dir = os.path.join(a.outdir, "onnx", name.replace("/", "__"))
                ort = ORTModelForSequenceClassification.from_pretrained(name, export=True)
                ort.save_pretrained(onnx_dir)
                q = ORTQuantizer.from_pretrained(onnx_dir)
                q.quantize(save_dir=onnx_dir + "-int8",
                           quantization_config=AutoQuantizationConfig.avx2(is_static=False))
                qm = ORTModelForSequenceClassification.from_pretrained(onnx_dir + "-int8")
                def score_ort():
                    out = []
                    for s in range(0, len(docs), BATCH):
                        batch = docs[s:s+BATCH]
                        feats = tok([QUERY]*len(batch), batch, padding=True, truncation=True,
                                    max_length=MAX_LEN, return_tensors="pt")
                        out.extend(qm(**feats).logits.squeeze(-1).tolist())
                    return out
                t = bench(score_ort, a.reps)
                size_mb = sum(os.path.getsize(os.path.join(onnx_dir+"-int8", f))
                              for f in os.listdir(onnx_dir+"-int8") if f.endswith(".onnx")) / 2**20
                rows.append((name, "int8-onnx", "cpu", params, t, 0))
                print(f"  int8 cpu : {t:6.2f} s  (onnx file {size_mb:.0f} MB)")
            except Exception as e:
                print(f"  int8 失败: {type(e).__name__}: {str(e)[:150]}")

    os.makedirs(a.outdir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(a.outdir, f"reranker_{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Reranker latency - {a.n_candidates} candidates - {stamp}\n\n")
        f.write("| model | precision | device | params M | time s (median) | VRAM GB |\n|---|---|---|---:|---:|---:|\n")
        for name, prec, dev, p, t, v in rows:
            f.write(f"| {name.split('/')[-1]} | {prec} | {dev} | {p:.0f} | {t:.2f} | {v:.2f} |\n")
    print(f"\n写入 {path}")

if __name__ == "__main__":
    main()
