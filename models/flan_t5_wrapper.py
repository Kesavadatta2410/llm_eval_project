"""
flan_t5_wrapper.py — FLAN-T5-Large (780M) encoder-decoder wrapper.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from models.base_model import BaseModel


class FlanT5Wrapper(BaseModel):
    """Wrapper for Google FLAN-T5-Large (seq2seq, instruction-tuned)."""

    def __init__(self):
        super().__init__("flan_t5")

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hf_id)
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.gen_config.max_new_tokens,
            temperature=self.gen_config.temperature,
            top_p=self.gen_config.top_p,
            do_sample=self.gen_config.do_sample,
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
