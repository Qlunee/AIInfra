"""
推理引擎入口层
1. 初始化配置、tokenizer、scheduler、model runner
2. 接收用户 prompt，把它包装成 Sequence
3. 每次调用 step() 执行一轮 prefill 或 decode
4. 在 generate() 中循环调度，直到所有请求完成
"""


import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        # TP初始化
        ctx = mp.get_context("spawn")
        ## 启动多个子进程
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        ## 主进程
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        # 创建调度器，并注册程序退出时的清理函数。
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    # 添加请求
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        print(
        f"\033[33m[add_request] \033[0m"
        f"seq_id={seq.seq_id}, "
        f"prompt_len={seq.num_prompt_tokens}, "
        f"max_tokens={seq.max_tokens}"
    )
        self.scheduler.add(seq)

    # 单步执行
    def step(self):
        ## 调度器先决定这一轮要执行哪些请求，以及这一轮是 prefill 还是 decode
        seqs, is_prefill = self.scheduler.schedule()
        
        phase = "prefill" if is_prefill else "decode"

        if phase == "prefill":
            print(
            f"\033[33m[step:schedule] \033[0m"
            f"phase={phase}, "
            f"num_seqs={len(seqs)}, "
            f"seq_ids={[seq.seq_id for seq in seqs]}, "
            f"scheduled_tokens={[seq.num_scheduled_tokens for seq in seqs]}, "
            f"prompt_lens={[seq.num_prompt_tokens for seq in seqs]}, "
            f"cached_tokens={[seq.num_cached_tokens for seq in seqs]}"
            )
    
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # print(
        # f"\033[33m[step:model] \033[0m"
        # f"phase={phase}, "
        # f"output_token_ids={token_ids}"
        # )

        ## 把新token写回并判断请求是否结束
        self.scheduler.postprocess(seqs, token_ids, is_prefill)

        # print(
        # f"\033[33m[step:postprocess] \033[0m"
        # f"seq_status={[(seq.seq_id, seq.status.name, seq.num_completion_tokens) for seq in seqs]}"
        # )
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        # 本质是问 scheduler：waiting 队列是否为空；running 队列是否为空
        return self.scheduler.is_finished()

    # 批量生成
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        # 创建进度条
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # 把所有请求加入 scheduler
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        round_id = 0

        while not self.is_finished():
            round_id += 1
            # print(f"\n[generate] round={round_id}")

            # 高精度计时器
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                # prefill_throughput = 本轮处理的 prompt token 数 / 本轮耗时
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                # 每完成一个请求，就放进字典
                # seq_id 是请求的唯一编号；token_ids 是这个请求生成出来的 token id 列表。
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
