import os
import torch
import pickle
import gc
from transformers import AutoModelForCausalLM, AutoConfig

model_name_dict = {
    "facebook/opt-125m": "opt_125m",
    # "facebook/opt-1.3b": "opt_1_point_3",
    # "facebook/opt-6.7b": "opt_6_point_7",
    # "facebook/opt-13b": "opt_13b",
    # "huggyllama/llama-7b": "llama_7b",
    # "huggyllama/llama-13b": "llama_13b",
    # "Qwen/Qwen3-8B": "Qwen3_8B",
    # "deepseek-ai/deepseek-llm-7b-chat": "deepseek_llm_7b_chat",
}

# 全局关闭梯度计算，节省显存
torch.set_grad_enabled(False)

# 遍历字典中的每一个模型
for model_str, file_name in model_name_dict.items():
    print(f"========== 正在处理模型: {model_str} ==========")

    try:
        # 加载模型和配置
        model = AutoModelForCausalLM.from_pretrained(
            model_str,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        model_config = AutoConfig.from_pretrained(model_str).to_dict()

        # 记录 Linear 层的参数 shape
        layer_config = {}
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                layer_config[n] = list(m.weight.shape)
                # print(f"Module name:  {n}")            # 如果不需要每次都打印具体的层信息，可以注释掉这部分
                # print(f"Module shape: {m.weight.shape}\n")

        # 保存为 pickle 文件
        file_path = f"./model_shape_config/{file_name}.pickle"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump((model_config, layer_config), f)

        print(f"成功: {model_str} 的参数已保存至 {file_path}\n")

    except Exception as e:
        print(f"错误: 处理模型 {model_str} 时发生异常: {e}\n")

    finally:
        # 释放显存和内存，防止加载下一个大模型时 OOM
        if "model" in locals():
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print("所有模型处理完毕！")
