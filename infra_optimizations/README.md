# infra_optimizations

Minimal infra-level optimization template for the ActualBudget serving project.

## Goals
1. Compare right-sizing candidates after fixing the best model and best system configuration.
2. Measure readiness / startup behavior and capacity headroom under representative load.

## Folder structure
- `1_rightsizing.ipynb`: compare VM/resource profiles
- `2_readiness_and_capacity.ipynb`: measure startup, readiness, and burst capacity
- `config/infra_config.example.yaml`: minimal config
- `src/infra_bench.py`: shared helpers

## Suggested workflow
1. Finalize the best model candidate from `model_optimizations`.
2. Finalize the best worker / batching setup from `system_optimizations`.
3. Start the service with that fixed configuration.
4. Run `1_rightsizing.ipynb` on each resource tier.
5. Run `2_readiness_and_capacity.ipynb` for startup and burst tests.
