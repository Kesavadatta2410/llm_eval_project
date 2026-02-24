"""
llama_wrapper.py — LLaMA-3 8B Instruct causal language model wrapper.

Requires:
  - HF_TOKEN environment variable set (gated model)
  - ~10 GB VRAM for 8-bit quantization, or CPU fallback
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.base_model import BaseModel, CONFIG


class LLaMAWrapper(BaseModel):
    """Wrapper for Meta LLaMA-3-8B-Instruct with optional 8-bit quantization."""

    def __init__(self):
        super().__init__("llama3")

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id, token=self.hf_token, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {"trust_remote_code": True, "attn_implementation": "eager"}

        # 8-bit quantization on CUDA (requires bitsandbytes)
        quant_cfg = CONFIG.get("quantization", {})
        if quant_cfg.get("use_8bit", False) and "cuda" in str(self.device):
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # verify it is actually installed
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                load_kwargs["device_map"] = "auto"
            except ImportError:
                print("  ⚠ bitsandbytes not available — loading in float16")
                load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.float16

        # Free any cached GPU memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(self.hf_id, **load_kwargs)

        # Move to device only if device_map was not used
        if "device_map" not in load_kwargs:
            self.model = self.model.to(self.device)

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        # Clear cache before each generation to reduce OOM risk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)

        try:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.gen_config.max_new_tokens,
                do_sample=False,          # greedy — avoids logits_process CUDA crash
                pad_token_id=self.tokenizer.pad_token_id,
            )
            generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"  ⚠ CUDA error on this record, skipping: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "[CUDA_ERROR]"
