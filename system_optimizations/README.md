# system_optimizations

简化后的 system-level 结构，只保留你项目最需要的三部分：

1. **worker × concurrency 区间**：找在线请求的稳定区间
2. **arrival pattern 区间**：比较 constant vs poisson/burst
3. **batch endpoint 区间**：找导入场景更合适的 batch size

## 目录

- `1_worker_concurrency.ipynb`
- `2_arrival_patterns.ipynb`
- `3_batch_endpoint.ipynb`
- `src/bench.py`：唯一的公共 helper
- `config/system_config.example.yaml`：唯一配置文件
- `example_data/`：参考你的 model input/output JSON

## 为什么这样更合理

你这个项目是 **CPU 轻量分类服务**，system optimization 的核心不是复现 Triton/GPU lab，而是回答：

- 在线请求时，几个 worker 最稳？
- burst 到来时，p95 latency 从哪里开始变坏？
- 导入场景到底要不要 batch endpoint？

## 你应该替换的地方

只需要优先检查这三项：

- `base_url`
- `predict_path` / `predict_batch_path`
- `model_name` / `model_version`

如果 training team 后续换模型，只要 API 输入输出契约不变，这套 benchmark 基本不用改；如果字段变了，再改 `example_data/request_single.json` 即可。
