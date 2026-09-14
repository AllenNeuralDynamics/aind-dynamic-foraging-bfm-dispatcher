"""Tests for the Tang (macaque) task-aware author baseline."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "studies"
    / "09-gru-cross-species-transfer"
    / "run_tang_author_baseline.py"
)
SPEC = importlib.util.spec_from_file_location("study09_tang_baseline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TestTangAuthorBaseline(unittest.TestCase):
    def test_what_block_learns_objects_and_maps_back_to_right(self):
        rows = pd.DataFrame(
            {
                "block_type": [MODULE.WHAT_BLOCK] * 4,
                "animal_response": [1, 0, 1, 0],
                "trial_object": [1, 1, 0, 0],
            }
        )
        np.testing.assert_array_equal(MODULE._domain_choice(rows), [1, 1, 0, 0])
        np.testing.assert_allclose(
            MODULE._probability_right(rows, np.array([0.8, 0.8, 0.3, 0.3])),
            [0.8, 0.2, 0.7, 0.3],
        )

    def test_where_block_learns_and_predicts_actions(self):
        rows = pd.DataFrame(
            {
                "block_type": [MODULE.WHERE_BLOCK] * 3,
                "animal_response": [1, 0, 1],
                "trial_object": [0, 0, 1],
            }
        )
        np.testing.assert_array_equal(MODULE._domain_choice(rows), [1, 0, 1])
        np.testing.assert_allclose(
            MODULE._probability_right(rows, np.array([0.7, 0.2, 0.6])),
            [0.7, 0.2, 0.6],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
