"""
base_model.py — Abstract base class for all model wrappers.

Every model wrapper (GPT-2, LLaMA, FLAN-T5) inherits from BaseModel
and implements the _load_model() and generate() methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time, os, yaml, torch
from pathlib import Path


# ── Load global config ──────────────────────────────────────────────────────
def load_config() -> dict:
    """Load config.yaml from the project root."""
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()


@dataclass
class GenerationConfig:
    """Consistent sampling parameters across all models."""
    temperature: float = CONFIG["generation"]["temperature"]
    top_p: float       = CONFIG["generation"]["top_p"]
    max_new_tokens: int = CONFIG["generation"]["max_new_tokens"]
    do_sample: bool     = CONFIG["generation"]["do_sample"]


class BaseModel(ABC):
    """
    Abstract wrapper around a Hugging Face model.

    Subclasses must implement:
      - _load_model()    → sets self.model and self.tokenizer
      - generate(prompt) → returns generated text string
    """

    def __init__(self, model_key: str):
        info = CONFIG["models"][model_key]
        self.model_key   = model_key
        self.hf_id       = info["hf_id"]
        self.model_type  = info["type"]
        self.description = info["description"]
        self.gen_config  = GenerationConfig()

        # Device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Hugging Face token for gated models
        self.hf_token = os.environ.get("HF_TOKEN", None)

        # Placeholders — set by subclass
        self.model     = None
        self.tokenizer = None

        # Load
        print(f"  Loading {self.description} on {self.device} …")
        self._load_model()
        self.model.eval()
        print(f"  ✓ {model_key} ready")

    # ── Abstract interface ──────────────────────────────────────────────────

    @abstractmethod
    def _load_model(self):
        """Load self.model and self.tokenizer from Hugging Face."""
        ...

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the model's text response for a given prompt."""
        ...

    # ── Utility ─────────────────────────────────────────────────────────────

    def generate_with_timing(self, prompt: str) -> dict:
        """Generate a response and measure wall-clock inference time."""
        start = time.perf_counter()
        response = self.generate(prompt)
        elapsed = time.perf_counter() - start
        return {"response": response, "inference_time": elapsed}

    def __repr__(self):
        return f"<{self.__class__.__name__}  model={self.hf_id}  device={self.device}>"
