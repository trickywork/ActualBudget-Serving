# CPU-only model optimization notebook

This folder adds a **lab-style model-level experiment** for the serving role.

## Files
- `notebooks/model_optimization_cpu.ipynb`: single notebook containing all model variants
- `Dockerfile.modelopt`: CPU-only Jupyter container for running the notebook on Chameleon
- `requirements-modelopt.txt`: extra dependencies for the notebook container

## Variants compared in the notebook
- `torch_eager_fp32`
- `torch_compiled_fp32`
- `onnxruntime_cpu_fp32`
- `onnxruntime_cpu_dynamic_int8`

## Why one notebook?
Putting all model variants in one notebook keeps the experiment fair:
- same hardware
- same synthetic representative inputs
- same preprocessing freeze
- same benchmark functions
- same output summary files

That matches the serving notes idea of comparing model artifacts in isolation before system-level changes.

## How to run on a Chameleon CPU VM
From the repository root:

```bash
sudo docker build -f Dockerfile.modelopt -t tx-modelopt-cpu .
sudo docker run --rm -it -p 8888:8888   -v $(pwd):/workspace   --name tx-modelopt tx-modelopt-cpu
```

Then open Jupyter in the browser and run:
- `notebooks/model_optimization_cpu.ipynb`

## Headless execution option
If you want a fully executed notebook artifact without clicking cells manually:

```bash
sudo docker run --rm -it   -v $(pwd):/workspace   --name tx-modelopt-headless tx-modelopt-cpu   jupyter nbconvert --to notebook --execute   --inplace /workspace/notebooks/model_optimization_cpu.ipynb
```

## Output files
After execution, results are saved to:
- `results/model_optimization_summary.csv`
- `results/model_optimization_summary.json`
- exported model artifacts under `models/`

## Practical interpretation
For your actual project, this notebook should be treated as a **model-level shortlist step**.
It does **not** force your final production service to use the PyTorch surrogate. Instead, it gives you a clean, reproducible CPU-only comparison in the style of the lab, which you can cite when deciding whether a more optimized artifact is worth integrating.
