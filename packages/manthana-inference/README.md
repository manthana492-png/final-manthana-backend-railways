# manthana-inference

Loads [`config/cloud_inference.yaml`](../../config/cloud_inference.yaml) and builds OpenAI-compatible clients pointed at OpenRouter.

```python
from pathlib import Path
from manthana_inference import (
    load_cloud_inference_config,
    resolve_role,
    build_openrouter_async_client,
    chat_complete_async,
)

cfg = load_cloud_inference_config(Path(os.environ["CLOUD_INFERENCE_CONFIG_PATH"]))
role = resolve_role(cfg, "oracle_chat")
client = build_openrouter_async_client(api_key=os.environ["OPENROUTER_API_KEY"], config=cfg)
text = await chat_complete_async(client, role, messages=[{"role": "user", "content": "Hi"}])
```

Install (from repo root):

```bash
pip install -e packages/manthana-inference
```
