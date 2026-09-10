from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto() # 等待调度，还没进入运行队列
    RUNNING = auto() # 已经完成 prefill，正在 decode
    FINISHED = auto() #


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids) # 当前这个请求的完整 token 列表
        self.last_token = token_ids[-1]  # 当前最后一个 token
        self.num_tokens = len(self.token_ids) # 当前总 token 数
        self.num_prompt_tokens = len(token_ids) # 原始 prompt 的 token 数
        self.num_cached_tokens = 0 # 已经写入 KV cache 的 token 数
        self.num_scheduled_tokens = 0 # 本轮被调度处理的 token 数
        self.is_prefill = True 
        self.block_table = [] # 这个请求占用的 KV cache block 列表
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens # 限制的是“生成的新 token 数”，不是 prompt + completion 的总长度
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    '''装饰器，用来把“方法”伪装成“属性”来访问'''
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        '''当前已经生成了多少个新 token'''
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        '''只取生成出来的部分'''
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        '''计算当前请求需要多少个 KV cache block'''
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        '''返回第 i 个 block 对应的 token 切片'''
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        序列化：
        为了多进程传输 Sequence 时做序列化优化
        主进程要把 Sequence 对象传给 worker 进程。进程之间不能直接共享普通 Python 对象，所以需要先把对象“打包”成可传输的数据，这个过程就会用到 __getstate__()。
        它决定：这个 Sequence 被序列化时，只传哪些字段。
        """
        # prefill 阶段：传完整 token_ids；decode 阶段：只传 last_token
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        """
        反序列化：
        收到刚才打包的数据后，怎么重新构造一个 Sequence 对象
        """
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
