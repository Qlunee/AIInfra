#打印模型 safetensors 中前 20 个 key。
#打印 nano-vLLM model.named_parameters() 前 20 个 key。
import os
from glob import glob

import torch
import torch.distributed as dist
from safetensors import safe_open
from transformers import AutoConfig

from nanovllm.models.qwen3 import Qwen3ForCausalLM


def init_dist():
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:23456",
            rank=0,
            world_size=1,
        )


def print_safetensors_keys(model_path: str, limit: int = 20):
    print("\n=== safetensors keys ===")

    files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    assert files, f"No safetensors files found in {model_path}"

    count = 0
    for file in files:
        print(f"\nfile: {os.path.basename(file)}")
        with safe_open(file, framework="pt", device="cpu") as f:
            for key in f.keys():
                print(key)
                count += 1
                if count >= limit:
                    return


def print_nanovllm_parameter_keys(model_path: str, limit: int = 20):
    print("\n=== nano-vLLM model.named_parameters keys ===")

    init_dist()

    config = AutoConfig.from_pretrained(model_path)

    # 避免真实分配大权重时初始化无所谓，这里只是构造结构打印参数名。
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)

    for i, (name, param) in enumerate(model.named_parameters()):
        if i >= limit:
            break
        print(name, tuple(param.shape))


def main():
    model_path = "/home/lq/huggingface/Qwen3-0.6B"

    print_safetensors_keys(model_path, limit=20)
    print_nanovllm_parameter_keys(model_path, limit=20)


if __name__ == "__main__":
    main()