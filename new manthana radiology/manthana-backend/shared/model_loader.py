"""
Manthana — Lazy Model Loader
Auto-downloads models from HuggingFace on first inference.
Caches forever in /models/ volume.
"""

import os
import logging
import threading
from pathlib import Path

logger = logging.getLogger("manthana.model_loader")

MODEL_DIR = os.environ.get(
    "MANTHANA_MODEL_CACHE",
    os.path.join(os.path.expanduser("~"), ".manthana", "models"),
)
os.makedirs(MODEL_DIR, exist_ok=True)
DEVICE = os.getenv("DEVICE", "cuda")
def _hf_token() -> str | None:
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HF_API_TOKEN")
    )


def get_hf_token() -> str | None:
    """Public helper for gated Hugging Face downloads outside LazyModel."""
    return _hf_token()


class LazyModel:
    """Thread-safe lazy model loader.
    
    Downloads model from HuggingFace on first .get() call,
    caches in MODEL_DIR forever after.
    
    Usage:
        model = LazyModel("paige-ai/Virchow", "virchow")
        m = model.get()  # Downloads on first call, instant after
    """

    def __init__(self, model_id: str, cache_name: str, model_class=None, 
                 device: str = None, extra_kwargs: dict = None):
        """
        Args:
            model_id: HuggingFace model ID (e.g., "paige-ai/Virchow")
            cache_name: Subdirectory in MODEL_DIR (e.g., "virchow")
            model_class: The class to use for loading (e.g., AutoModel). 
                         If None, uses transformers.AutoModel.
            device: Override device ("cuda", "cpu", or None for env default)
            extra_kwargs: Extra kwargs passed to from_pretrained()
        """
        self.model_id = model_id
        self.cache_name = cache_name
        self.cache_dir = os.path.join(MODEL_DIR, cache_name)
        self.device = device or DEVICE
        self.model_class = model_class
        self.extra_kwargs = extra_kwargs or {}
        self._model = None
        self._lock = threading.Lock()
        self._offloaded_from_cuda = False

    def get(self):
        """Get the loaded model. Downloads on first call."""
        if self._model is None:
            with self._lock:
                if self._model is None:  # Double-check after lock
                    self._load()
        elif self._offloaded_from_cuda and self.device == "cuda":
            import torch
            if torch.cuda.is_available() and hasattr(self._model, "to"):
                self._model = self._model.to("cuda")
                self._offloaded_from_cuda = False
        return self._model

    def _load(self):
        """Download and load the model."""
        logger.info(f"Loading model: {self.model_id} → {self.cache_dir}")
        
        os.makedirs(self.cache_dir, exist_ok=True)

        # Import here to avoid startup cost
        if self.model_class is None:
            from transformers import AutoModel
            self.model_class = AutoModel

        kwargs = {
            "cache_dir": self.cache_dir,
            "trust_remote_code": True,
            **self.extra_kwargs,
        }
        
        tok = _hf_token()
        if tok:
            kwargs["token"] = tok

        self._model = self.model_class.from_pretrained(
            self.model_id, **kwargs
        )

        # Move to device if applicable
        if hasattr(self._model, "to") and self.device:
            import torch
            device = self.device
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"
            self._model = self._model.to(device)
        self._offloaded_from_cuda = False

        logger.info(f"Model loaded: {self.model_id} on {self.device}")

    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory."""
        return self._model is not None

    def offload_to_cpu(self):
        """Move model to CPU RAM (faster to reload than from disk)."""
        if self._model is None:
            return
        import torch
        if hasattr(self._model, "cpu"):
            self._model = self._model.cpu()
            self._offloaded_from_cuda = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Model offloaded to CPU: {self.model_id}")

    def unload(self):
        """Free model from memory (weights stay cached on disk)."""
        if self._model is not None:
            del self._model
            self._model = None
            self._offloaded_from_cuda = False
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Model unloaded: {self.model_id}")


class LazyPipeline:
    """Lazy loader for HuggingFace pipelines (e.g., image-classification)."""

    def __init__(self, task: str, model_id: str, cache_name: str,
                 device: str = None, extra_kwargs: dict = None):
        self.task = task
        self.model_id = model_id
        self.cache_dir = os.path.join(MODEL_DIR, cache_name)
        self.device = device or DEVICE
        self.extra_kwargs = extra_kwargs or {}
        self._pipeline = None
        self._lock = threading.Lock()

    def get(self):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    self._load()
        return self._pipeline

    def _load(self):
        from transformers import pipeline as hf_pipeline
        import torch

        os.makedirs(self.cache_dir, exist_ok=True)

        device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        kwargs = {
            "model": self.model_id,
            "device": device,
            "model_kwargs": {"cache_dir": self.cache_dir},
            **self.extra_kwargs,
        }

        tok = _hf_token()
        if tok:
            kwargs["token"] = tok

        self._pipeline = hf_pipeline(self.task, **kwargs)
        logger.info(f"Pipeline loaded: {self.task}/{self.model_id}")

    def is_loaded(self) -> bool:
        return self._pipeline is not None


class ManagedModel(LazyModel):
    """LazyModel with VRAM tracking for the Memory Manager (Phase 4)."""

    def __init__(
        self,
        model_id: str,
        cache_name: str,
        device="cuda",
        model_class=None,
        extra_kwargs=None,
        vram_gb: float = 0,
        priority: int = 5,
    ):
        super().__init__(model_id, cache_name, model_class, device, extra_kwargs)
        self.vram_gb = float(vram_gb)
        self.priority = int(priority)
        self.last_used = None
        self.call_count = 0

    def get(self):
        import time

        self.last_used = time.time()
        self.call_count += 1
        ModelMemoryManager.get_instance().acquire(self)
        return super().get()

    def is_on_gpu(self) -> bool:
        if self._model is None or getattr(self, "_offloaded_from_cuda", False):
            return False
        import torch
        try:
            return next(self._model.parameters()).device.type == "cuda"
        except (StopIteration, AttributeError):
            return self.device == "cuda" and not getattr(self, "_offloaded_from_cuda", False)


class ModelMemoryManager:
    """GPU VRAM manager with LRU-style eviction via CPU offload."""

    _instance = None
    _singleton_lock = threading.Lock()

    def __init__(self, gpu_budget_gb=None):
        if gpu_budget_gb is None:
            import torch
            if torch.cuda.is_available():
                gpu_budget_gb = (
                    torch.cuda.get_device_properties(0).total_memory / 1e9 * 0.85
                )
            else:
                gpu_budget_gb = 512.0
        self.gpu_budget = float(gpu_budget_gb)
        self.loaded: dict[str, ManagedModel] = {}
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def acquire(self, model: ManagedModel):
        with self._lock:
            self.loaded[model.cache_name] = model
            # Reserve VRAM for this model once it will sit on GPU
            while (
                self._used_vram() + model.vram_gb > self.gpu_budget
                and len(self.loaded) > 0
            ):
                if not self._evict_coldest(exclude_cache_name=model.cache_name):
                    break

    def _evict_coldest(self, exclude_cache_name: str | None = None) -> bool:
        candidates = [
            m
            for cn, m in self.loaded.items()
            if cn != exclude_cache_name and m.is_loaded() and m.is_on_gpu()
        ]
        if not candidates:
            return False
        coldest = min(candidates, key=lambda m: (m.last_used is None, m.last_used or 0.0))
        coldest.offload_to_cpu()
        return True

    def _used_vram(self) -> float:
        return sum(
            m.vram_gb
            for m in self.loaded.values()
            if m.is_loaded() and m.is_on_gpu()
        )


def download_weights(url: str, dest_path: str, filename: str = None) -> str:
    """Download model weights from a direct URL (non-HuggingFace).
    
    Used for models like MedSAM2, Comp2Comp, etc. that
    distribute weights via direct download links.
    
    Returns the full path to the downloaded file.
    """
    import urllib.request

    os.makedirs(dest_path, exist_ok=True)
    
    if filename is None:
        filename = url.split("/")[-1]
    
    filepath = os.path.join(dest_path, filename)
    
    if os.path.exists(filepath):
        logger.info(f"Weights already cached: {filepath}")
        return filepath

    logger.info(f"Downloading weights: {url} → {filepath}")
    urllib.request.urlretrieve(url, filepath)
    logger.info(f"Download complete: {filepath}")
    
    return filepath
