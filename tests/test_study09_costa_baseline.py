"""Tests for the Costa (macaque) block-aware author baseline."""

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "studies"
    / "09-gru-cross-species-transfer"
    / "run_costa_author_baseline.py"
)
SPEC = importlib.util.spec_from_file_location("study09_costa_baseline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TestCostaAuthorBaseline(unittest.TestCase):
    def test_real_source_blocks_are_separate_reset_units(self):
        rows = pd.DataFrame(
            {
                "ses_idx": ["adapt", "adapt", "adapt", "test"],
                "source_block": [1, 1, 2, 1],
                "trial": [1, 0, 2, 0],
                "animal_response": [0, 1, 0, 1],
                "earned_reward": [1, 0, 1, 1],
            }
        )
        blocks = MODULE._ordered_blocks(rows, {"adapt"})
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["trial"].tolist(), [0, 1])
        self.assertEqual(blocks[1]["trial"].tolist(), [2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
