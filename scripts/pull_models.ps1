# 在 PowerShell 里运行: .\scripts\pull_models.ps1
# 把 benchmark 要用的模型拉进 ollama 容器的数据卷里（只需拉一次）
$models = @(
  "qwen3.5:2b-q4_K_M",
  "qwen3.5:2b-q8_0",
  "qwen3.5:4b-q4_K_M",
  "qwen3.5:4b-q8_0",
  "qwen3.5:9b-q4_K_M"
)
foreach ($m in $models) {
  Write-Host "==> pulling $m"
  docker compose exec ollama ollama pull $m
}
docker compose exec ollama ollama list
