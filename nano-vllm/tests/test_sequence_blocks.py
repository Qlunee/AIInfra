import unittest
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

nanovllm_pkg = types.ModuleType("nanovllm")
nanovllm_pkg.__path__ = [str(ROOT / "nanovllm")]
sys.modules.setdefault("nanovllm", nanovllm_pkg)

from nanovllm.engine.sequence import Sequence


class TestSequenceBlocks(unittest.TestCase):
    def setUp(self):
        self.original_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.original_block_size

    def make_sequence(self, length: int) -> Sequence:
        return Sequence(list(range(length)))

    def test_one_token_uses_one_block(self):
        seq = self.make_sequence(1)

        self.assertEqual(seq.num_blocks, 1)
        self.assertEqual(seq.last_block_num_tokens, 1)
        self.assertEqual(seq.block(0), [0])

    def test_exactly_one_full_block(self):
        seq = self.make_sequence(4)

        self.assertEqual(seq.num_blocks, 1)
        self.assertEqual(seq.last_block_num_tokens, 4)
        self.assertEqual(seq.block(0), [0, 1, 2, 3])

    def test_one_token_over_block_boundary(self):
        seq = self.make_sequence(5)

        self.assertEqual(seq.num_blocks, 2)
        self.assertEqual(seq.last_block_num_tokens, 1)
        self.assertEqual(seq.block(0), [0, 1, 2, 3])
        self.assertEqual(seq.block(1), [4])

    def test_multiple_blocks_with_partial_last_block(self):
        seq = self.make_sequence(10)

        self.assertEqual(seq.num_blocks, 3)
        self.assertEqual(seq.last_block_num_tokens, 2)
        self.assertEqual(seq.block(0), [0, 1, 2, 3])
        self.assertEqual(seq.block(1), [4, 5, 6, 7])
        self.assertEqual(seq.block(2), [8, 9])

    def test_multiple_full_blocks(self):
        seq = self.make_sequence(8)

        self.assertEqual(seq.num_blocks, 2)
        self.assertEqual(seq.last_block_num_tokens, 4)
        self.assertEqual(seq.block(0), [0, 1, 2, 3])
        self.assertEqual(seq.block(1), [4, 5, 6, 7])

    def test_invalid_block_index_raises(self):
        seq = self.make_sequence(4)

        with self.assertRaises(AssertionError):
            seq.block(-1)

        with self.assertRaises(AssertionError):
            seq.block(1)


if __name__ == "__main__":
    unittest.main()
