"""
models – Shared model wrappers for LLM inference.

Usage:
    from models import load_model
    model = load_model("gpt2")
    response = model.generate("What is the capital of France?")
"""

from models.base_model import BaseModel, GenerationConfig, load_config

def load_model(model_key: str) -> BaseModel:
    """Factory function: returns the correct wrapper for a model key."""
    if model_key == "gpt2":
        from models.gpt2_wrapper import GPT2Wrapper
        return GPT2Wrapper()
    elif model_key == "llama3":
        from models.llama_wrapper import LLaMAWrapper
        return LLaMAWrapper()
    elif model_key == "flan_t5":
        from models.flan_t5_wrapper import FlanT5Wrapper
        return FlanT5Wrapper()
    else:
        raise ValueError(f"Unknown model key: {model_key}. Choose from: gpt2, llama3, flan_t5")
