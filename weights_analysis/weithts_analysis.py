import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import random, time, argparse, os
from tqdm import tqdm
from typing import Optional


torch.set_grad_enabled(False)
torch.manual_seed(0)
random.seed(0)


def load_model(model_str: str):
    """Load a pre-trained model from Hugging Face."""
    model = AutoModelForCausalLM.from_pretrained(
        model_str, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_str, use_fast=False, trust_remote_code=True
    )
    return model, tokenizer


def weights_analysis(
    model: nn.Module, group_method: str = "layerwise", save_path: Optional[str] = None
):
    """Analyze the weights of the model."""
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if group_method == "layerwise":
                pass
            elif group_method == "channelwise":
                pass
            elif group_method == "groupwise":
                pass
            else:
                raise ValueError(f"Unknown group method: {group_method}")

        


if __name__ == "__main__":
    model_name = "facebook/opt-125m"
    group_method = "layerwise"  # 按层划分, 按通道划分 和 按组划分
