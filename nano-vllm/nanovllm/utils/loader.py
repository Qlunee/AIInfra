# 负责 把 Hugging Face safetensors 权重加载到 nano-vLLM 自己定义的模型结构里
import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

# 默认权重加载：模型参数 param <- safetensors 里的 loaded_weight
def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 判断是不是 packed 权重
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 把 HF 参数名改成 nano-vLLM 参数名
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
