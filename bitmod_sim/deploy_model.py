import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from torchinfo import summary


def get_model_structure(model_name: str, depth: int = 6, is_print: bool = True):
    """获取模型结构并打印"""
    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="float16",
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 构造符合模型输入要求的dummy input
    seq_length = getattr(model.config, "max_position_embeddings", 128)
    vocab_size = getattr(model.config, "vocab_size", 50272)
    dummy_input_ids = torch.randint(0, vocab_size, (1, seq_length), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, seq_length), dtype=torch.long)

    # 使用torchinfo打印模型结构
    if is_print:
        print("Model structure:")
        summary(model, input_data=(dummy_input_ids, dummy_attention_mask), depth=depth)
    else:
        info = summary(
            model, input_data=(dummy_input_ids, dummy_attention_mask), depth=depth
        )
        model_info = model_name.split("/")
        if len(model_info) > 1:
            model_str = f"{model_info[0]}__{model_info[1]}"
        else:
            model_str = f"{model_info[0]}__"
        os.makedirs("model_struct", exist_ok=True)
        with open(f"model_struct/{model_str}_struct.txt", "w") as f:
            f.write(str(info))
        print(f"Model structure saved to model_struct/{model_name}_struct.txt")


if __name__ == "__main__":
    model_name = "facebook/opt-125m"
    get_model_structure(model_name, depth=6, is_print=False)
