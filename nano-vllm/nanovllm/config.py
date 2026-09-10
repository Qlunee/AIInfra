import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str # 本地模型目录路径，例如 ~/huggingface/Qwen3-0.6B/
    max_num_batched_tokens: int = 16384  # 一次调度中允许处理的最大 token 数，用来限制 batch 的总体计算量
    max_num_seqs: int = 512   # 一次 batch 中最多同时处理多少条序列/请求
    max_model_len: int = 4096     # 模型单条序列的最大上下文长度，会再和 HuggingFace 配置里的最大位置长度取较小值
    gpu_memory_utilization: float = 0.9    # KV cache 最多使用多少比例的 GPU 显存，0.9 表示最多使用 90%
    tensor_parallel_size: int = 1    # 张量并行大小，也就是把模型切到多少张 GPU 上运行；1 表示单卡
    enforce_eager: bool = False    # 是否强制使用 PyTorch eager 模式；True 更方便调试，False 可能启用 CUDA Graph 等优化
    hf_config: AutoConfig | None = None    # HuggingFace 的模型配置对象，初始化后由 AutoConfig.from_pretrained 自动加载
    eos: int = -1    # 结束符 token id，-1 表示先占位，后续通常会从 tokenizer 或模型配置中设置
    kvcache_block_size: int = 256    # KV cache 的 block 大小；这里要求是 256 的整数倍
    num_kvcache_blocks: int = -1    # KV cache block 数量，-1 表示初始化时还未计算，后续会根据显存容量确定

    def __post_init__(self):
        # 确保传入的是存在的本地模型目录
        assert os.path.isdir(self.model)
        # KV cache block 大小需要满足后端内存管理/对齐要求
        assert self.kvcache_block_size % 256 == 0
        # 这个简化实现只支持 1 到 8 路张量并行
        assert 1 <= self.tensor_parallel_size <= 8
        # 读取模型目录中的 config.json，得到 hidden_size、层数、最大位置长度等模型元信息
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 实际最大上下文长度不能超过模型本身支持的最大位置长度
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
