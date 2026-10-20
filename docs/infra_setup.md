# LocalScholar — infra 部分

本地部署的学术论文检索与问答系统。这个目录目前只有 infra 这一层（部署 + 性能实验），
应用层代码（services/api）由队友维护，之后合并到同一个 repo。

## 一、Windows 上第一次跑起来

1. **装 Docker Desktop**（官网下载，安装时保持默认的 WSL2 后端）。装完重启，开 Docker Desktop，等左下角变绿。
2. **确认 NVIDIA 驱动是新的**（GeForce Experience 或 NVIDIA App 更新一下）。Docker Desktop 的 WSL2 后端自带 GPU 支持，不需要另装 NVIDIA Container Toolkit。
3. **验证 GPU 透传**，PowerShell 里跑：
   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
   ```
   看到 RTX 5070 的表格就是通了。
4. **准备环境变量**：复制 `.env.example` 为 `.env`。
5. **启动（GPU）**：
   ```powershell
   cd $HOME\Documents\localscholar
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
   docker compose ps          # 三个容器都应该是 running
   ```
   只用 CPU 的话去掉 `-f docker-compose.gpu.yml`。
6. **拉模型**（第一次要下几个 GB，之后在数据卷里不用重拉）：
   ```powershell
   .\scripts\pull_models.ps1
   ```
7. **冒烟测试**：
   ```powershell
   curl http://localhost:11434/api/generate -d '{"model":"qwen3.5:2b-q4_K_M","prompt":"Say hi","stream":false}'
   curl http://localhost:6333/collections
   ```

## 二、跑 benchmark
见 `infra/benchmarks/README.md`。

## 三、目录
```
docker-compose.yml        CPU 默认编排：ollama + qdrant + postgres
docker-compose.gpu.yml    GPU 覆盖
.env.example              环境变量模板
services/ollama/          模型 tag 约定；以后放 Modelfile 和硬件探测脚本
infra/benchmarks/         性能实验脚本和结果
docs/                     infra_benchmark_report.md（边做边填）
scripts/                  pull_models.ps1
```

## 四、git
```powershell
git init
git add .
git commit -m "infra skeleton: compose, ollama config, benchmark script"
```
队友建好 GitHub repo 后：`git remote add origin <url>`，`git push -u origin main`。
