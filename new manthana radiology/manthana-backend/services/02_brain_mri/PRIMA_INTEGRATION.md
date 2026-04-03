# Prima integration (MLNeurosurg/Prima)

- Repository: https://github.com/MLNeurosurg/Prima
- License: MIT
- Weights: volume-mount only (e.g. `primafullmodel107.pt`, VQ-VAE tokenizer) — not baked into Docker images.
- Entrypoint (upstream): `python end-to-end_inference_pipeline/pipeline.py --config <yaml>`
- Fake-data runs validate **code paths** only; clinical logits require real checkpoints.
- Set `PRIMA_REPO_DIR` (default `/opt/Prima`) and `PRIMA_CONFIG_YAML` to a config whose paths point at mounted weights before enabling `run_prima_study` subprocess wiring in `shared/prima_pipeline.py`.
