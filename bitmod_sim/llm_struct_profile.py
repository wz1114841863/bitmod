import os
import torch
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


model_name_dict = {
    "facebook/opt-125m": "opt_125m",
    "facebook/opt-1.3b": "opt_1_point_3",
    "facebook/opt-6.7b": "opt_6_point_7",
    "microsoft/phi-2": "phi_2",
    "01-ai/Yi-6B": "yi_6",
    "meta-llama/Llama-2-7b-hf": "llama_2_7",
    "meta-llama/Llama-2-13b-hf": "llama_2_13",
    "meta-llama/Meta-Llama-3-8B": "llama_3_8",
}


def get_model_structure(model_name: str):
    torch.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto"
    )
    if hasattr(model.config, "n_positions"):
        seq_length = model.config.n_positions
    elif hasattr(model.config, "max_position_embeddings"):
        seq_length = model.config.max_position_embeddings
    else:
        raise ValueError("Cannot determine sequence length from model config.")
    vocab_size = getattr(model.config, "vocab_size", 50272)  # 50272 可替换为默认

    dummy_input = torch.randint(0, vocab_size, (1, seq_length), dtype=torch.long)

    device = next(model.parameters()).device  # 获取模型所在设备
    dummy_input = dummy_input.to(device)
    torch.onnx.export(
        model, dummy_input, f"{model_name}_structure.onnx", export_params=False
    )

    # model_config = AutoConfig.from_pretrained(model_name).to_dict()

    # layer_config = {}
    # for n, m in model.named_modules():
    #     if isinstance(m, torch.nn.Linear):
    #         layer_config[n] = list(m.weight.shape)
    #         print(f"Module name:  {n}")
    #         print(f"Module shape: {m.weight.shape}")
    #         print()
    # print("\n\n")

    # file_path = f"./model_shape_config/{model_name_dict[model_name]}.pickle"
    # os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # with open(file_path, "wb") as f:
    #     pickle.dump((model_config, layer_config), f)




if __name__ == "__main__":
    model_name = "facebook/opt-125m"
    get_model_structure(model_name)
