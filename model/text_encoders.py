# Copyright (c) Meta Platforms, Inc. and affiliates.
"""T5 text encoding for motion generation models."""

import torch
import torch.nn as nn
from typing import List


class T5TextEncoder(nn.Module):
    """Frozen MoLingo-style T5 encoder that suppresses expected decoder-weight warnings."""
    def __init__(self, model_name: str = "google/flan-t5-large", device: str = "cuda", max_length: int = 77):
        super().__init__()
        from transformers import T5EncoderModel, T5Tokenizer

        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        if "xxl" in model_name:
            self.feature_dim = 4096
        elif "xl" in model_name:
            self.feature_dim = 2048
        elif "large" in model_name:
            self.feature_dim = 1024
        elif "base" in model_name:
            self.feature_dim = 768
        elif "small" in model_name:
            self.feature_dim = 512
        else:
            self.feature_dim = 1024

        print(f"Loading T5 encoder: {model_name}")
        from transformers.utils import logging as hf_logging
        previous_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5EncoderModel.from_pretrained(model_name)
        finally:
            hf_logging.set_verbosity(previous_verbosity)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Return [B, seq_len, feature_dim] tokens with FP16 autocast disabled to avoid NaNs."""
        device = next(self.model.parameters()).device

        inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=False):
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state

        return features.float()

    def encode_text_with_mask(self, texts: List[str]) -> tuple:
        """Return tokens and a 1-valid mask with FP16 autocast disabled to avoid NaNs."""
        device = next(self.model.parameters()).device

        inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=False):
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state

        mask = inputs.attention_mask.float()

        return features.float(), mask

    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self
