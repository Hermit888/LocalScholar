# Infra Benchmark Report

> 边做边填。每一节先写"问题 → 实验设置 → 数据 → 结论"，数据从 infra/benchmarks/results 里贴。

## 0. 测试环境
- 机器：Lenovo 笔记本（Windows 11）
- CPU / 内存：（待填 CPU 型号）/ 16 GB（Ollama 看到 15.3 GiB，空闲约 7.7 GiB）
- GPU / 显存：RTX 5070 Laptop, 8 GB（8151 MiB；Windows 桌面常驻占用约 0.5 GB；Ollama 报告可用 6.7 GiB，预留后实际 6.3 GiB）
- 驱动 582.05，CUDA 13.0
- Docker Desktop（WSL2 后端），GPU 透传通过 `--gpus all` 验证
- Ollama 0.40.5（容器 `ollama/ollama:latest`），`OLLAMA_NUM_PARALLEL=1`，`OLLAMA_MAX_LOADED_MODELS=1`，flash attention 关
- 模型：Qwen3.5 系列（0.8b/2b/4b/9b…，**没有 8B**），GGUF 量化 tag 见 `services/ollama/README.md`。该系列为多模态模型，权重附带约 330M 参数的视觉 projector（mmproj 预估最坏占用约 0.9 GB），文本请求时不一定加载

**测量方法**：`infra/benchmarks/ollama_bench.py`，每格 3 次取中位数，固定生成 128 token，温度 0。
每次请求加随机前缀以击穿 Ollama 的 prompt cache（否则重复 prompt 的 prefill 速度会虚高 30-40 倍，第一版脚本踩过）。
prefill/decode 速度来自 Ollama 返回的 `prompt_eval_duration` / `eval_duration`；内存占用来自 `/api/ps` 的 `size` / `size_vram`。

## 1. 模型规格 × 量化 × CPU/GPU 延迟矩阵
**问题**：QA 层选几 B 的模型？CPU 保底能跑到哪一档？

### 1.1 GPU（RTX 5070 Laptop 8 GB）

| model | prompt tok | prefill tok/s | decode tok/s | TTFT s | mem GB | VRAM GB |
|---|---:|---:|---:|---:|---:|---:|
| 2B Q4 | 81 | 1876 | 117.1 | 0.04 | 1.49 | 1.49 |
| 2B Q4 | 1525 | 6201 | 112.7 | 0.25 | 1.50 | 1.50 |
| 2B Q4 | 5995 | 6810 | 108.7 | 0.88 | 1.63 | 1.63 |
| 2B Q8 | 83 | 1883 | 91.1 | 0.05 | 2.18 | 2.18 |
| 2B Q8 | 1524 | 6701 | 89.3 | 0.23 | 2.18 | 2.18 |
| 2B Q8 | 5994 | 7493 | 87.0 | 0.80 | 2.32 | 2.32 |
| 4B Q4 | 83 | 1074 | 63.2 | 0.08 | 2.86 | 2.86 |
| 4B Q4 | 1525 | 2922 | 60.9 | 0.52 | 2.88 | 2.88 |
| 4B Q4 | 5993 | 2729 | 59.3 | 2.20 | 3.15 | 3.15 |
| 4B Q8 | 81 | 983 | 47.7 | 0.08 | 4.57 | 4.57 |
| 4B Q8 | 1522 | 3099 | 49.6 | 0.49 | 4.58 | 4.58 |
| 4B Q8 | 5993 | 3707 | 55.9 | 1.62 | 4.78 | 4.78 |
| 9B Q4 | 81 | 734 | 47.1 | 0.11 | 5.75 | **5.02** |
| 9B Q4 | 1523 | 1764 | 45.7 | 0.86 | 5.77 | **5.04** |
| 9B Q4 | 5994 | 1755 | **35.8** | 3.42 | 5.98 | **4.97** |

（原始数据：`infra/benchmarks/results/gpu_20261017_0213.csv`；冷启动加载 3.6-7.0 s，容器首次 CUDA 初始化另加约 30 s，已排除）

**观察**：
1. decode 速度随参数量干净递减：2B Q4 117 → 2B Q8 91 → 4B Q4 63 → 4B Q8 ~50 → 9B Q4 47 tok/s。Q8 比同规格 Q4 慢 20-25%，显存多 45-60%。
2. **9B Q4 溢出显存**：mem 5.98 GB vs VRAM 4.97 GB，约 1 GB 权重被放到主机内存。后果是它成为唯一一个随上下文明显掉速的模型（47 → 35.8 tok/s @ 6K），并且显存被占满，embedding / reranker / NLI 三个小模型只能走 CPU。
3. **上下文对显存的影响极小**：2B 从 80 到 6000 token 只多用 0.14 GB，4B 多 0.3 GB。Qwen3.5 是混合架构（部分层为线性注意力，日志中可见 `llama_memory_recurrent`），KV cache 远小于同规模纯 attention 模型，decode 速度几乎不随上下文下降。这对"直塞多篇论文"的 QA 模式是利好。
4. 6K token 上下文的首 token 延迟：2B 0.9 s，4B 2.2 s，9B 3.4 s。GPU 上 QA 层塞 3-4 篇论文摘要+章节是可接受的。
5. 稳定性：4B Q8 第一轮在前一个模型未完全卸载时加载，导致 Ollama 进程崩溃（连接直接断开）；显式 `keep_alive=0` 卸载后第二轮正常。接近显存上限时需要串行加载并显式卸载。

### 1.2 CPU（同一台机器，GPU profile 关闭，16 GB 内存）

| model | prompt tok | prefill tok/s | decode tok/s | TTFT s | mem GB |
|---|---:|---:|---:|---:|---:|
| 2B Q4 | 82 | 490 | 32.8 | 0.17 | 1.49 |
| 2B Q4 | 1524 | 663 | 32.6 | 2.34 | 1.50 |
| 2B Q4 | 5993 | 581 | 30.3 | **10.3** | 1.64 |
| 2B Q8 | 82 | 300 | 23.2 | 0.28 | 2.18 |
| 2B Q8 | 1523 | 358 | 22.6 | 4.63 | 2.19 |
| 2B Q8 | 5995 | 329 | 22.0 | 18.2 | 2.33 |
| 4B Q4 | 81 | 188 | 15.9 | 0.43 | 2.87 |
| 4B Q4 | 1524 | 259 | 16.1 | 5.89 | 2.88 |
| 4B Q4 | 5995 | 239 | 15.1 | **25.4** | 3.16 |
| 4B Q8 | 83 | 119 | 10.6 | 0.70 | 4.57 |
| 4B Q8 | 1524 | 134 | 10.6 | 11.5 | 4.59 |
| 4B Q8 | 5994 | 122 | 10.2 | 49.3 | 4.87 |
| 9B Q4 | 82 | 119 | 9.7 | 0.69 | 5.70 |
| 9B Q4 | 1524 | 151 | 9.4 | 10.6 | 5.71 |
| 9B Q4 | 5993 | 142 | 9.0 | **43.8** | 6.03 |

（原始数据：`infra/benchmarks/results/cpu_20261018_1737.csv`）

**观察**：
1. GPU 对 CPU 的加速比：decode 3.5-5×，prefill 10-15×。**CPU 上 prefill 是主要瓶颈**：GPU 的 prefill/decode 速度比约 50:1，CPU 只有 15:1，所以上下文越长 CPU 越吃亏。
2. 2B Q4 是 CPU 保底档：33 tok/s 的生成速度可读，但 6K 上下文首 token 要等 10 s。
3. 4B Q4 在 CPU 上处于可用边缘：16 tok/s 勉强，6K 上下文首 token 25 s，QA 层塞三篇论文用户要等半分钟。
4. 9B 在 CPU 上不可用（9 tok/s，首 token 44 s），4B Q8 同样（10 tok/s）。
5. 内存：9B Q4 占 6 GB，在 16 GB 机器上可以加载，但与 Postgres/Qdrant/辅助模型共存时余量不足。

**结论**：
- **推荐档位：4B Q4**。63 tok/s、首 token 2 s 内、3.15 GB 显存，剩余约 3 GB 可容纳 SPECTER2 + cross-encoder + NLI 三个辅助模型同时驻留 GPU。
- **9B Q4 定位为"可演示、不推荐"**：能跑但无余量，辅助模型被挤到 CPU，整体流水线反而更慢。
- **2B Q4 用于 query 理解**：0.04 s 首 token、1.5 GB，structured output 的延迟用户无感。可与 4B 同时驻留（合计 < 5 GB），避免来回切换模型的加载开销。
- Q8 的质量收益需要在评测集上验证；若与 Q4 差距不显著，不采用。
- **CPU 保底档：2B Q4**（query 理解 + QA 生成都用它）；**CPU 模式下 QA 层必须走 chunk 检索压缩上下文**，目标把 prompt 控制在 2K token 内（首 token 约 2-3 s），直塞全文（6K+）首 token 10 s 起步，不可接受。
- 硬件分档草案（供 `hardware_sizing_guide.md`）：
  - 无独显 / <8 GB 可用内存：2B Q4，上下文上限 2K
  - 无独显 / 16 GB 内存：2B Q4 主力，4B Q4 可选（接受 15 tok/s）
  - 6-8 GB 显存：4B Q4 主力 + 2B Q4 做 query 理解，辅助模型驻留 GPU
  - 12 GB+ 显存：9B Q4 全进显存（未测，推测）

## 2. thinking 模式的延迟代价
**问题**：QA 层开 thinking 值不值？

**方法**：`infra/benchmarks/think_cost.py`。真实 QA 风格 prompt（3 段论文摘要 + "对比三篇并引用"的问题，约 600 token），
think 开/关各 3 次取中位数，`num_predict=4096` 让模型把回答写完。GPU，温度 0。
（注：`ollama_bench.py --think` 固定生成 128 token，只能测每 token 速度，测不出思考 token 的总量，故另写此脚本。）

| model | think | 总耗时 s | 生成 tok | 思考字符 | 回答字符 | decode tok/s | 截断 |
|---|---|---:|---:|---:|---:|---:|---|
| 4B Q4 | off | 9.9 | 658 | 0 | 3082 | 69.9 | 否 |
| 4B Q4 | on | **17.4** | 1299 | 4461 | 1248 | 76.1 | 否 |
| 2B Q4 | off | 4.8 | 639 | 0 | 3111 | 137.1 | 否 |
| 2B Q4 | on | **26.7** | 3576 | 12438 | 2007 | 134.9 | **是（撞 4096 上限）** |

（原始数据：`results/think_gpu_20261018_1800.md`，含各配置回答预览）

**观察**：
1. thinking 不改变每 token 速度（甚至略高），代价全在 token 数量：4B 翻倍，2B 翻 5.6 倍。
2. **4B 开 thinking 总耗时 +75%**，但最终回答反而更短（3082 → 1248 字符）——思考消耗了大部分预算。
3. **2B 开 thinking 失控**：12k 字符的思考仍未收敛，撞到生成上限被截断。小模型的思考模式倾向于绕圈。
4. 2B 关 thinking 4.8 s 写完 3000 字符的三篇对比回答，是 Discovery 层摘要和轻量 QA 的合理选择。

**结论**：
- **QA 层默认 `think=false`。** 2B 任何场景不开；4B 仅在用户显式请求"深度分析"时开，且必须设 `num_predict` 上限（建议 2048）防止失控。
- thinking 是否提升回答质量需在评测集上验证（回答变短是负面信号）；在质量数据出来之前，按"不开"设计 API 默认值。
- 对 CPU 模式：按 CPU decode 速度换算，4B thinking 一次回答约 80 s，2B thinking 超过 100 s，均不可用。

## 3. 长上下文 vs chunk 检索
**问题**：直塞全文的 token 上限设多少？

## 4. 推理引擎对比（Ollama / llama.cpp / vLLM）

## 5. 容器 vs 原生（Windows）

## 6. 推荐配置
（提炼成 hardware_sizing_guide.md）
