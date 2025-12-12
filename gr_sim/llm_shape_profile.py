import argparse
import os
import torch
import pickle
import transformers

from transformers import AutoModelForCausalLM, AutoConfig
from model_list import model_name_dict

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default="facebook/opt-125m",
    help="Name of model",
)
args = parser.parse_args()
model_str = args.model

torch.set_grad_enabled(False)

model = AutoModelForCausalLM.from_pretrained(
    model_str,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)
model_config = AutoConfig.from_pretrained(model_str).to_dict()

layer_config = {}
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        layer_config[n] = list(m.weight.shape)
        print(f"Module name:  {n}")
        print(f"Module shape: {m.weight.shape}")
        print()
print("\n\n")


file_path = f"./model_shape_config/{model_name_dict[model_str]}.pickle"
os.makedirs(os.path.dirname(file_path), exist_ok=True)
with open(file_path, "wb") as f:
    pickle.dump((model_config, layer_config), f)
