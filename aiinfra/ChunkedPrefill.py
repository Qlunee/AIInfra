import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class CausalChunkedPrefill(nn.Module):
    """
    流式 + 因果的Chunked Prefill实现
    专为自回归LLM（如GPT、LLaMA）的推理优化
    """
    def __init__(self, d_model: int, n_heads: int, chunk_size: int = 512):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.chunk_size = chunk_size
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """将张量分割成多头"""
        batch_size, seq_len, _ = x.shape
        ## 交换head、seq_len维度是因为希望每个head独立计算
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)    

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """将多头合并"""
        batch_size, n_heads, seq_len, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    
    
    def prefill_standard(self, x: torch.Tensor) -> torch.Tensor:
        """
        标准注意力（不分块）- 用于验证正确性
        因果注意力：每个位置只能看到之前的位置
        """
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        mask = mask.view(1, 1, seq_len, seq_len)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)
        
        output = self._merge_heads(attn_output)
        output = self.out_proj(output)
        
        return output
    

    def prefill_chunked(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
            """
            分块预填充（流式 + 因果）

            Args:
                x: 输入序列 [batch, seq_len, d_model]

            Returns:
                output: 注意力输出 [batch, seq_len, d_model]
                kv_cache: KV缓存 (K列表, V列表)
            """
            batch_size, seq_len, _ = x.shape
            
            n_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
            
            k_cache = []
            v_cache = []
            
            outputs = []
            
            print(f"分块预填充: 序列长度={seq_len}, 分块大小={self.chunk_size}, 分块数={n_chunks}")
            
            for chunk_idx in range(n_chunks):
                # 计算当前分块的起始和结束索引
                start = chunk_idx * self.chunk_size
                end = min((chunk_idx + 1) * self.chunk_size, seq_len)
                chunk_len = end - start
                
                # 提取当前分块
                chunk = x[:, start:end, :]  # [batch, chunk_len, d_model]
                
                q = self.q_proj(chunk)
                k = self.k_proj(chunk)
                v = self.v_proj(chunk)
                
                # 分割多头
                q = self._split_heads(q)  # [batch, n_heads, chunk_len, head_dim]
                k = self._split_heads(k)  # [batch, n_heads, chunk_len, head_dim]
                v = self._split_heads(v)  # [batch, n_heads, chunk_len, head_dim]
                
                # 将当前分块的K、V缓存起来
                k_cache.append(k)
                v_cache.append(v)
                
                # 当前累计的KV总长度
                total_kv_len = sum(k.shape[2] for k in k_cache)
                
                # 将所有缓存的K、V拼接起来
                k_all= torch.cat(k_cache, dim=2)  # [batch, n_heads, total_kv_len, head_dim]
                v_all= torch.cat(v_cache, dim=2)  # [batch, n_heads, total_kv_len, head_dim]
                
                # 计算注意力分数
                scores = torch.matmul(q, k_all.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch, n_heads, chunk_len, total_kv_len]            
                
                # 创建因果掩码
                # 注意：我们需要确保当前chunk内的Q也不能看到同一chunk内未来的K
                # 所以需要构建一个 [chunk_len, total_kv_len] 的掩码

                # 方法1：构建完整的掩码矩阵          
                q_positions = torch.arange(chunk_len, device=x.device).unsqueeze(1) + start # [chunk_len, 1]
                kv_positions = []
                for i, k_chunk in enumerate(k_cache):
                    kv_start = i * self.chunk_size
                    kv_len = k_chunk.shape[2]
                    kv_positions.extend(range(kv_start, kv_start + kv_len))
                kv_positions = torch.tensor(kv_positions, device=x.device).unsqueeze(0)
                
                # q只能看到kv_positions中小于等于自己的位置
                mask = q_positions >= kv_positions # [chunk_len, total_kv_len]
                mask = mask.view(1, 1, chunk_len, total_kv_len)  # [1, 1, chunk_len, total_kv_len]
                
                scores = scores.masked_fill(~mask, float('-inf'))
                
                attn_weights = F.softmax(scores, dim=-1)  # [batch, n_heads, chunk_len, total_kv_len]
                
                attn_output = torch.matmul(attn_weights, v_all)  # [batch, n_heads, chunk_len, head_dim]
                
                output_chunk = self._merge_heads(attn_output)  # [batch, chunk_len, d_model]
                output_chunk = self.out_proj(output_chunk)  # [batch, chunk_len, d_model]
                
                outputs.append(output_chunk)

                print(f"  处理chunk {chunk_idx+1}/{n_chunks}: "
                    f"位置 {start}:{end}, "
                    f"KV缓存长度={total_kv_len}")
                
            # 将所有分块的输出拼接起来
            output = torch.cat(outputs, dim=1)  # [batch, seq_len, d_model]
            
            return output, (k_cache, v_cache)
        
    def decode_step(self,
                    x: torch.Tensor,
                    kv_cache: Tuple[List[torch.Tensor], List[torch.Tensor]]
                    ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
            """
            解码步骤：处理单个token（使用KV缓存）

            Args:
                x: 当前token [batch, 1, d_model]
                kv_cache: KV缓存 (K列表, V列表)

            Returns:
                output: 当前token的输出 [batch, 1, d_model]
                updated_kv_cache: 更新后的KV缓存
            """           
            k_cache, v_cache = kv_cache
            
            q = self.q_proj(x)  # [batch, 1, d_model]
            k = self.k_proj(x)  # [batch, 1, d_model]
            v = self.v_proj(x)  # [batch, 1, d_model]
            
            q = self._split_heads(q)  # [batch, n_heads, 1, head_dim]
            k = self._split_heads(k)  # [batch, n_heads, 1, head_dim]
            v = self._split_heads(v)  # [batch, n_heads, 1, head_dim]
            
            k_cache.append(k)
            v_cache.append(v)
            
            k_all = torch.cat(k_cache, dim=2)  # [batch, n_heads, total_kv_len, head_dim]
            v_all = torch.cat(v_cache, dim=2)  # [batch, n_heads, total_kv_len, head_dim]
            
            scores = torch.matmul(q, k_all.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch, n_heads, 1, total_kv_len]
            
            attn_weights = F.softmax(scores, dim=-1)  # [batch, n_heads, 1, total_kv_len]
            
            attn_output = torch.matmul(attn_weights, v_all)  # [batch, n_heads, 1, head_dim]    
            
            output = self._merge_heads(attn_output)  # [batch, 1, d_model]
            output = self.out_proj(output)  # [batch, 1, d_model]
            
            return output, (k_cache, v_cache)
    
class StreamingLLMAttention:
    """
    流式LLM注意力层
    """
  
    def __init__(self, d_model: int, n_heads: int, chunk_size: int = 512):
        self.attn = CausalChunkedPrefill(d_model, n_heads, chunk_size)
        self.chunk_size = chunk_size
        self.kv_cache = None

    def prefill(self, prompt: torch.Tensor) -> torch.Tensor:
        """
        预填充阶段：处理用户输入的prompt

        Args:
            prompt: 用户输入的prompt [batch, prompt_len, d_model]

        Returns:
            注意力输出
        """  
        output, self.kv_cache = self.attn.prefill_chunked(prompt)
        return output
    
    def generate_token(self, token_emb: torch.Tensor) -> torch.Tensor:
        """
        生成一个token

        Args:
            token_emb: 当前token的embedding [batch, 1, d_model]

        Returns:
            当前token的输出
        """
        if self.kv_cache is None:
            raise ValueError("请先调用prefill初始化KV缓存")

        output, self.kv_cache = self.attn.decode_step(token_emb, self.kv_cache)
        return output

    def reset_cache(self):
        """重置KV缓存"""
        self.kv_cache = None
        
        
        
def test_causal_chunked_prefill():
    """测试流式 + 因果的分块预填充"""

    torch.manual_seed(42)

    print("=" * 70)
    print("流式 + 因果分块预填充测试")
    print("=" * 70)

    # 测试配置
    batch_size = 2
    seq_len = 9  # 测试用短序列
    d_model = 64
    n_heads = 4
    chunk_size = 3

    print(f"配置:")
    print(f"  batch_size={batch_size}, seq_len={seq_len}")
    print(f"  d_model={d_model}, n_heads={n_heads}")
    print(f"  chunk_size={chunk_size}")

    # 创建模型
    model = CausalChunkedPrefill(
        d_model=d_model,
        n_heads=n_heads,
        chunk_size=chunk_size
    )

    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\n输入形状: {x.shape}")

    print("\n1. 计算标准注意力（不分块）...")
    with torch.no_grad():
        output_standard = model.prefill_standard(x)
    print(f"   标准注意力输出形状: {output_standard.shape}")

    print("\n2. 计算分块预填充...")
    with torch.no_grad():
        output_chunked, kv_cache = model.prefill_chunked(x)
    print(f"   分块预填充输出形状: {output_chunked.shape}")
    print("\n3. 比较两种方法的输出...")
    diff = torch.abs(output_standard - output_chunked)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"   最大差异: {max_diff:.10f}")
    print(f"   平均差异: {mean_diff:.10f}")

    tolerance = 1e-5
    if max_diff < tolerance:
        print(f"   ✓ 测试通过！差异在容忍范围内 (< {tolerance})")
    else:
        print(f"   ✗ 测试失败！差异超出容忍范围")
        print(f"\n   调试信息（第一个样本的前3个位置）:")
        for pos in range(min(3, seq_len)):
            print(f"   位置 {pos}:")
            print(f"     标准: {output_standard[0, pos, :5].detach().numpy().round(4)}")
            print(f"     分块: {output_chunked[0, pos, :5].detach().numpy().round(4)}")
            print(f"     差异: {diff[0, pos, :5].detach().numpy().round(8)}")

    print("\n4. 测试解码步骤（生成后续token）...")

    # 生成一个测试token
    test_token = torch.randn(batch_size, 1, d_model)

    # 使用KV缓存解码
    output_decode, updated_kv_cache = model.decode_step(test_token, kv_cache)

    print(f"   解码输出形状: {output_decode.shape}")
    print(f"   更新后KV缓存长度: {sum(k.shape[2] for k in updated_kv_cache[0])}")

    # 验证解码的正确性
    x_with_new = torch.cat([x, test_token], dim=1)

    # 用标准注意力计算完整结果
    output_full_new = model.prefill_standard(x_with_new)

    # 取最后一个位置的输出（对应新token）
    output_full_last = output_full_new[:, -1:, :]

    # 比较
    diff_decode = torch.abs(output_decode - output_full_last).max().item()
    print(f"   解码vs标准差异: {diff_decode:.10f}")

    if diff_decode < tolerance:
        print(f"   ✓ 解码步骤正确")
    else:
        print(f"   ✗ 解码步骤有误")

    return max_diff < tolerance and diff_decode < tolerance


def test_streaming_api():
    """测试流式API"""
    torch.manual_seed(42)

    print("\n" + "=" * 70)
    print("流式API测试")
    print("=" * 70)

    # 创建流式LLM注意力
    stream_attn = StreamingLLMAttention(
        d_model=64,
        n_heads=4,
        chunk_size=3
    )

    # 模拟一个prompt
    prompt_len = 9
    prompt = torch.randn(1, prompt_len, 64)

    print(f"1. 预填充阶段: 处理{prompt_len}个token的prompt")
    output = stream_attn.prefill(prompt)
    print(f"   输出形状: {output.shape}")

    print(f"\n2. 生成阶段: 生成3个token")
    for i in range(3):
        # 模拟一个token的embedding（实际中从embedding层获取）
        token_emb = torch.randn(1, 1, 64)

        output_token = stream_attn.generate_token(token_emb)
        print(f"   生成token {i+1}: 输出形状 {output_token.shape}")

    print(f"\n3. 重置缓存")
    stream_attn.reset_cache()
    print(f"   KV缓存已重置")


def benchmark_performance():
    """性能基准测试"""

    import time

    torch.manual_seed(42)

    print("\n" + "=" * 70)
    print("性能基准测试")
    print("=" * 70)

    # 测试长序列
    d_model = 1024
    n_heads = 16
    chunk_size = 512

    # 创建模型
    model = CausalChunkedPrefill(
        d_model=d_model,
        n_heads=n_heads,
        chunk_size=chunk_size
    )

    # 测试不同序列长度
    test_cases = [
        {"seq_len": 512, "desc": "短序列（一个chunk）"},
        {"seq_len": 2048, "desc": "中等序列（4个chunk）"},
        {"seq_len": 8192, "desc": "长序列（16个chunk）"},
    ]
    
    for test_case in test_cases:
        seq_len = test_case["seq_len"]

        print(f"\n测试: {test_case['desc']} (seq_len={seq_len})")

        x = torch.randn(1, seq_len, d_model)

        # 标准注意力（可能OOM）
        try:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            start = time.time()
            with torch.no_grad():
                output_std = model.prefill_standard(x)
            time_std = time.time() - start

            mem_std = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

            print(f"  标准注意力: {time_std:.3f}s, "
                  f"内存: {mem_std/1024**2:.1f}MB" if torch.cuda.is_available() else f"{time_std:.3f}s")
        except RuntimeError as e:
            print(f"  标准注意力 OOM: {e}")
            time_std = float('inf')
            mem_std = float('inf')

        # 分块注意力
        try:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            start = time.time()
            with torch.no_grad():
                output_chunk, _ = model.prefill_chunked(x)
            time_chunk = time.time() - start

            mem_chunk = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

            print(f"  分块注意力: {time_chunk:.3f}s, "
                  f"内存: {mem_chunk/1024**2:.1f}MB" if torch.cuda.is_available() else f"{time_chunk:.3f}s")

            if time_std != float('inf'):
                speedup = time_std / time_chunk if time_chunk > 0 else 0
                mem_reduction = (mem_std - mem_chunk) / mem_std * 100 if mem_std > 0 else 0
                print(f"  加速: {speedup:.1f}x, 内存减少: {mem_reduction:.1f}%")
        except RuntimeError as e:
            print(f"  分块注意力 OOM: {e}")


print("流式 + 因果分块预填充实现")

# 运行基本测试
test_passed = test_causal_chunked_prefill()

if test_passed:
    # 测试流式API
    test_streaming_api()

    # 性能测试
    if torch.cuda.is_available():
        benchmark_performance()
    else:
        print("\n注意: CUDA不可用，跳过性能测试")
        
        
            
            