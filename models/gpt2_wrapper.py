"""
gpt2_wrapper.py — GPT-2 (124M) causal language model wrapper.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.base_model import BaseModel


class GPT2Wrapper(BaseModel):
    """Wrapper for OpenAI GPT-2 (124M parameters)."""

    def __init__(self):
        super().__init__("gpt2")

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # attn_implementation="eager" avoids the SDPA padding-mask CUDA
        # assertion error (srcIndex < srcSelectDimSize) seen in newer
        # transformers versions when input_ids are padded.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, attn_implementation="eager"
        )
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        # max_length=512 keeps well within GPT-2's 1024-token positional limit
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.gen_config.max_new_tokens,
            temperature=self.gen_config.temperature,
            top_p=self.gen_config.top_p,
            do_sample=self.gen_config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Strip the input tokens from the output
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
