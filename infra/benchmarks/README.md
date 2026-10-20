# infra/benchmarks

## ollama_bench.py — 模型 × 量化 × 上下文长度 的延迟/内存矩阵

前提：compose 已启动，模型已 `pull`（见 scripts/pull_models.ps1）。只用 Python 标准库。

```powershell
# 1. GPU 模式启动，跑一遍
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
python infra\benchmarks\ollama_bench.py --label gpu

# 2. 换 CPU 模式（去掉 GPU 覆盖文件），再跑一遍
docker compose down
docker compose up -d
python infra\benchmarks\ollama_bench.py --label cpu

# 3. thinking 模式的代价（只测 QA 层会用的模型）
python infra\benchmarks\ollama_bench.py --label gpu --think --models qwen3.5:4b-q4_K_M qwen3.5:9b-q4_K_M
```

每个格子跑 3 次取中位数。结果在 `results/` 下，`.md` 可以直接贴进报告。

### 指标含义
| 列 | 意思 | 对项目的用处 |
|---|---|---|
| prefill tok/s | 读 prompt 的速度 | QA 层塞 N 篇论文要等多久 |
| decode tok/s | 生成速度 | 用户看到回答逐字出现的速度，CPU 上低于 ~8 基本不可用 |
| TTFT | 首 token 延迟 | 用户点下去到开始出字 |
| cold load | 第一次加载模型 | 决定 keep_alive 设多长 |
| mem / VRAM | /api/ps 报告的占用 | 9B Q4 在 8GB 卡上能不能全进显存，是关键数据点 |

### 预期会看到什么（先写下来，跑完对照）
- GPU 上 2B/4B 全进显存，decode 几十 tok/s；9B Q4 + 8K 上下文可能部分溢出到内存，decode 速度断崖下跌——如果出现，`mem_vram_gb < mem_total_gb` 就是证据
- CPU 上 decode 速度大致和模型大小成反比，9B 可能个位数 tok/s
- prefill 在 CPU 上比 decode 更吃亏，8K 上下文的 TTFT 可能是十几秒到几十秒 → 这是 "CPU-first 下 QA 层直塞全文不现实" 的证据
- thinking 模式：decode 速度不变，但总 token 数翻几倍，总时间随之翻几倍

### 后续要加的实验（按优先级）
1. 长上下文曲线：固定模型，ctx 从 1K 到 32K，看延迟和显存怎么涨 → 决定 QA 层"直塞 vs 检索"的阈值
2. 推理引擎对比：同一个 GGUF 在 Ollama / llama.cpp server / vLLM 下的吞吐和显存
3. 容器 vs 原生：Windows 原生 Ollama vs Docker Desktop 里的 Ollama
