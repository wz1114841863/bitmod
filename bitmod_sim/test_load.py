import os
import time
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


def touch_model(model_name):
    """
    负责加载和校验
    """
    print(f"[Loading] {model_name}")
    try:
        config = AutoConfig.from_pretrained(model_name, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            local_files_only=True,
            config=config,
            device_map="auto",
        )
        print(f"[OK] 成功加载: {model_name}")

        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[Error] 加载失败 {model_name}: {e}")


if __name__ == "__main__":
    print(f"Current HF_HOME: {os.getenv('HF_HOME')}")
    MODELS = [
        "facebook/opt-125m",
        "facebook/opt-1.3b",
        "facebook/opt-6.7b",
        "facebook/opt-13b",
        "huggyllama/llama-7b",
        "huggyllama/llama-13b",
        "Qwen/Qwen3-8B",
    ]
    for m in MODELS:
        touch_model(m)
