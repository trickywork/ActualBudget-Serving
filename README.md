# ActualBudget Serving OneFolder

## 目录

```text
app/                    FastAPI 服务与 backend 代码
tools/                  prepare / benchmark / package 的 Python 脚本
docker/                 单镜像 Dockerfile
models/
  source/               训练团队模型源文件（已放入 repo）
  optimized/            生成的 ONNX 与 quantized ONNX
artifacts/
  examples/             input/output sample
  gradescope/           自动整理出的提交材料
results/
  raw/                  原始 benchmark 结果
  summary/              汇总结果
legacy_notebooks/       repo 里的 notebook 备份
tests/                  轻量 smoke tests
run.py                  唯一推荐入口
docker-compose.yml      多容器定义（serve / tooling）
```

---

## 最短上手路径

### 1) 本地整理后推 GitHub

```bash
git init
git add .
git commit -m "one-folder serving bundle"
git remote add origin <your-github-repo>
git push -u origin main
```

### 2) Chameleon 上运行

```bash
git clone <your-github-repo>
cd ActualBudget-Serving-OneFolder
python3 run.py doctor
python3 run.py prepare
python3 run.py up
python3 run.py smoke
python3 run.py bench-all
python3 run.py package
```

---

## 你平时只需要记住这些命令

### 准备模型与镜像

```bash
python3 run.py prepare
```

它会做这些事：

- build 单一 Docker image
- 检查 `models/source/` 下的 sklearn 模型
- 导出 ONNX
- 导出 ONNX dynamic quantization
- 生成 `models/manifest.json`
- 生成示例 output json

### 启动服务

默认启动最推荐配置：

- variant: `onnx_dynamic_quant`
- workers: `2`
- cpus: `2.0`
- mem: `3g`

```bash
python3 run.py up
```

### 切换 baseline / ONNX / quantized ONNX

```bash
python3 run.py up --variant baseline
python3 run.py up --variant onnx
python3 run.py up --variant onnx_dynamic_quant
```

### 改 system-level 参数

```bash
python3 run.py up --variant onnx_dynamic_quant --workers 1
python3 run.py up --variant onnx_dynamic_quant --workers 2
python3 run.py up --variant onnx_dynamic_quant --workers 4
```

### 改 infra-level 参数

```bash
python3 run.py up --variant onnx_dynamic_quant --workers 2 --cpus 1.0 --mem 1g
python3 run.py up --variant onnx_dynamic_quant --workers 2 --cpus 2.0 --mem 2g
python3 run.py up --variant onnx_dynamic_quant --workers 2 --cpus 2.0 --mem 3g
```

### 跑 benchmark

```bash
python3 run.py bench-model
python3 run.py bench-system
python3 run.py bench-infra
python3 run.py bench-all
```

### 生成 Gradescope 友好材料

```bash
python3 run.py package
```

---

## 对应课程要求

### model-level

已支持：

- `baseline`：sklearn pipeline
- `onnx`：导出的 ONNX pipeline
- `onnx_dynamic_quant`：ONNX dynamic quantized pipeline

### system-level

已支持：

- worker count sweep
- concurrency sweep
- `/predict_batch` batch size sweep
- constant / poisson arrival pattern

### infrastructure-level

已支持：

- cold start / readiness timing
- container CPU / memory usage
- right-sizing 对比（通过 `--cpus` / `--mem` profile 跑）

---

## API

### `GET /healthz`

基础健康检查

### `GET /readyz`

模型是否已加载完成

### `GET /versionz`

返回：

- backend kind
- model path
- model version
- code version / git sha
- providers
- hardware summary

### `POST /predict`

请求示例见：

- `artifacts/examples/input_sample.json`

### `POST /predict_batch`

请求示例见：

- `artifacts/examples/batch_input_sample.json`

---

## 默认推荐提交配置

在 `m1.medium` 上，

- `variant = onnx_dynamic_quant`
- `workers = 2`
- `cpus = 2.0`
- `mem = 3g`

然后用 `bench-model`、`bench-system`、`bench-infra` 产出结果，再从 `artifacts/gradescope/serving_options_table.csv` 里挑最终表格。

---

## 说明

1. 这个工程把**运行入口收敛到 `run.py`**
2. build / prepare / benchmark 仍然都在容器里完成；`run.py` 只负责 orchestration。
3. `legacy_notebooks/` 只是保留材料，不再是默认运行路径。
4. 结果目录默认不提交大结果文件；需要时可以挑 `artifacts/gradescope/` 提交。
