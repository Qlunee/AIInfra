from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def make_seq(length: int):
    token_ids = list(range(length))
    return Sequence(token_ids, SamplingParams(max_tokens=4))


def print_round(round_id, seqs, is_prefill):
    print(f"\n[round {round_id}] is_prefill={is_prefill}")
    for seq in seqs:
        print(
            f"  seq_id={seq.seq_id}, "
            f"prompt_len={seq.num_prompt_tokens}, "
            f"cached={seq.num_cached_tokens}, "
            f"scheduled={seq.num_scheduled_tokens}, "
            f"num_tokens={seq.num_tokens}, "
            f"status={seq.status.name}, "
            f"block_table={seq.block_table}"
        )


def main():
    config = SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=10,
        eos=-1,
        kvcache_block_size=4,
        num_kvcache_blocks=100,
    )

    Sequence.block_size = config.kvcache_block_size
    scheduler = Scheduler(config)

    seqs = [
        make_seq(3),
        make_seq(15),
        make_seq(17),
    ]

    for seq in seqs:
        scheduler.add(seq)

    round_id = 0

    while scheduler.waiting:
        round_id += 1
        scheduled_seqs, is_prefill = scheduler.schedule()
        print_round(round_id, scheduled_seqs, is_prefill)

        # 这里不跑模型，所以手动模拟 postprocess 里对 cached tokens 的更新
        for seq in scheduled_seqs:
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0


if __name__ == "__main__":
    main()