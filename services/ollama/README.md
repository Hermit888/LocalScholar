# services/ollama

当前直接用官方镜像 `ollama/ollama:latest`，CPU/GPU 的差别只在 compose 的 GPU 覆盖文件里，
所以暂时不需要 Dockerfile.cpu / Dockerfile.gpu 两份。

后面需要自定义的时候再加：
- `Modelfile`：固定 system prompt、num_ctx、temperature 等参数，`ollama create` 成一个项目专用模型名
- 硬件探测脚本：启动时读显存大小，决定拉 2b / 4b / 9b 哪一档

## 模型 tag 约定
Ollama 上 `qwen3.5:2b` 这种默认 tag 是 **Q8_0**，不是 Q4。做对比实验必须写全名：

| 模型 | Q4 | Q8 |
|---|---|---|
| 2B | qwen3.5:2b-q4_K_M | qwen3.5:2b-q8_0 |
| 4B | qwen3.5:4b-q4_K_M | qwen3.5:4b-q8_0 |
| 9B | qwen3.5:9b-q4_K_M | qwen3.5:9b-q8_0 |

注意：Qwen3.5 系列没有 8B，是 9B。模型是多模态的，权重里带一个约 330M 参数的视觉 projector，
所以同样参数量比纯文本模型略大。
