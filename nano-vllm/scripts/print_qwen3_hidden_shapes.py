# 给一个小 batch 的假输入，打印每层 hidden states 形状。

import os
import sys
import tempfile
import types

import torch
import torch.distributed as dist
from transformers import Qwen3Config


def install_optional_attention_stubs():
    if "triton" not in sys.modules:
        triton = types.ModuleType("triton")
        triton.jit = lambda fn=None, **_: fn if fn is not None else (lambda f: f)
        sys.modules["triton"] = triton

    if "triton.language" not in sys.modules:
        tl = types.ModuleType("triton.language")
        tl.constexpr = int
        sys.modules["triton.language"] = tl

    if "flash_attn" not in sys.modules:
        flash_attn = types.ModuleType("flash_attn")

        def _unused_flash_attn(*args, **kwargs):
            raise RuntimeError("flash_attn is stubbed in this CPU shape-debug script")

        flash_attn.flash_attn_varlen_func = _unused_flash_attn
        flash_attn.flash_attn_with_kvcache = _unused_flash_attn
        sys.modules["flash_attn"] = flash_attn


def init_single_process_dist():
    if dist.is_initialized():
        return
    init_file = tempfile.NamedTemporaryFile(delete=False)
    init_file.close()
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file.name}",
        rank=0,
        world_size=1,
    )
    os.unlink(init_file.name)


def patch_attention_for_cpu_shape_debug():
    from nanovllm.layers.attention import Attention

    def fake_attention_forward(self, q, k, v):
        print(f"    attention q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}")
        return torch.zeros_like(q)

    Attention.forward = fake_attention_forward


def register_shape_hooks(model):
    hooks = []

    def hook(name):
        def _hook(module, inputs, output):
            if isinstance(output, tuple):
                shapes = [
                    tuple(x.shape) if torch.is_tensor(x) else type(x).__name__
                    for x in output
                ]
            else:
                shapes = tuple(output.shape)
            print(f"{name:<42} -> {shapes}")
        return _hook

    hooks.append(model.model.embed_tokens.register_forward_hook(hook("embed_tokens")))

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.input_layernorm.register_forward_hook(hook(f"layers.{i}.input_layernorm")))
        hooks.append(layer.self_attn.qkv_proj.register_forward_hook(hook(f"layers.{i}.self_attn.qkv_proj")))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(hook(f"layers.{i}.self_attn.o_proj")))
        hooks.append(layer.self_attn.register_forward_hook(hook(f"layers.{i}.self_attn")))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(hook(f"layers.{i}.post_attention_layernorm")))
        hooks.append(layer.mlp.gate_up_proj.register_forward_hook(hook(f"layers.{i}.mlp.gate_up_proj")))
        hooks.append(layer.mlp.down_proj.register_forward_hook(hook(f"layers.{i}.mlp.down_proj")))
        hooks.append(layer.mlp.register_forward_hook(hook(f"layers.{i}.mlp")))
        hooks.append(layer.register_forward_hook(hook(f"layers.{i}")))

    hooks.append(model.model.norm.register_forward_hook(hook("final_norm")))
    hooks.append(model.lm_head.register_forward_hook(hook("lm_head")))
    return hooks


def main():
    init_single_process_dist()
    install_optional_attention_stubs()

    from nanovllm.models.qwen3 import Qwen3ForCausalLM

    patch_attention_for_cpu_shape_debug()

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        tie_word_embeddings=False,
    )

    model = Qwen3ForCausalLM(config).eval()
    hooks = register_shape_hooks(model)

    # 假设这是 prefill 后拼接成的一个小 batch：2 条请求，长度分别为 3 和 5。
    # nano-vLLM 的模型实际接收的是 flatten 后的一维 token 序列。
    input_ids = torch.tensor([10, 11, 12, 20, 21, 22, 23, 24], dtype=torch.long)
    positions = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4], dtype=torch.long)

    print(f"input_ids  -> {tuple(input_ids.shape)}")
    print(f"positions  -> {tuple(positions.shape)}")
    print()

    with torch.inference_mode():
        hidden_states = model(input_ids, positions)
        logits = model.compute_logits(hidden_states)

    print()
    print(f"final hidden_states -> {tuple(hidden_states.shape)}")
    print(f"final logits        -> {tuple(logits.shape)}")

    for h in hooks:
        h.remove()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
