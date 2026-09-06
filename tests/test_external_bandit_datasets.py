from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import sys
import tarfile
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from external_bandit_datasets.schema import (  # noqa: E402
    interleaved_session_manifest,
    prefix_trial_manifest,
    validate_canonical_table,
    write_dataset,
)
from external_bandit_datasets.sources import (  # noqa: E402
    SOURCES,
    Source,
    download_source,
)
from external_bandit_datasets.adapters import (  # noqa: E402
    EXPECTED_AUDITS,
    _finish,
    adapt_beron,
    adapt_alsio,
    adapt_eckstein,
    adapt_costa,
    adapt_lopez_mouse,
    adapt_findling,
    adapt_kwak,
    adapt_lebedeva,
    adapt_miller,
    adapt_tang,
)
from external_bandit_datasets.cli import _names, run  # noqa: E402


def _table(*, sessions: int = 4, trials: int = 4) -> pd.DataFrame:
    rows = []
    for subject in ("a", "b"):
        for session in range(sessions):
            for trial in range(trials):
                rows.append(
                    {
                        "subject_id": subject,
                        "ses_idx": f"s{session + 1}",
                        "trial": trial,
                        "animal_response": trial % 2,
                        "rewarded": (trial + 1) % 2,
                        "earned_reward": (trial + 1) % 2,
                    }
                )
    return pd.DataFrame(rows)


class TestExternalBanditDatasets(unittest.TestCase):
    @staticmethod
    def _write_npy(archive: zipfile.ZipFile, name: str, values: list[int]) -> None:
        payload = io.BytesIO()
        np.save(payload, np.asarray([values], dtype=float))
        archive.writestr(name, payload.getvalue())

    def test_empty_adapter_output_has_a_schema_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing columns"):
            _finish([], name="grossman", excluded_trials=0, split="sessions")

    def test_cli_rejects_unknown_dataset_cleanly(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            _names("not-a-dataset")

    def test_no_download_reports_missing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "without --no-download"):
                run("grossman", cache_root=Path(directory), download=False)

    def test_sources_preserve_priority_order(self) -> None:
        self.assertEqual(list(SOURCES)[:3], ["grossman", "chen", "zid"])
        self.assertEqual(SOURCES["grossman"].digest_algorithm, "sha256")
        self.assertEqual(SOURCES["chen"].digest_algorithm, "sha256")
        self.assertEqual(SOURCES["zid"].digest_algorithm, "sha256")
        self.assertTrue(SOURCES["zid"].filename.endswith(".mat"))
        self.assertEqual(SOURCES["lebedeva"].digest_algorithm, "md5")

    def test_bad_forced_download_preserves_valid_cached_source(self) -> None:
        valid_payload = b"known-good-source"
        source = Source(
            dataset_id="test",
            species="mouse",
            title="test",
            repository="test",
            doi="test",
            version="1",
            license="CC0-1.0",
            url="https://example.invalid/source.bin",
            filename="source.bin",
            digest_algorithm="sha256",
            digest=hashlib.sha256(valid_payload).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "test" / source.filename
            destination.parent.mkdir(parents=True)
            destination.write_bytes(valid_payload)

            def write_bad_payload(_url: str, path: Path) -> None:
                path.write_bytes(b"truncated")

            with (
                mock.patch.dict(SOURCES, {"test": source}),
                mock.patch(
                    "external_bandit_datasets.sources._download",
                    side_effect=write_bad_payload,
                ),
                self.assertRaisesRegex(ValueError, "Checksum mismatch"),
            ):
                download_source("test", directory, force=True)

            self.assertEqual(destination.read_bytes(), valid_payload)
            self.assertFalse((destination.parent / ".source.bin.download").exists())

    def test_interleaved_split_then_first_k_semantics(self) -> None:
        manifest = interleaved_session_manifest(
            _table(), dataset_id="test", species="mouse"
        )
        row = manifest["subjects"][0]
        self.assertEqual(row["adapt_session_ids"], ["s1", "s3"])
        self.assertEqual(row["test_session_ids"], ["s2", "s4"])
        self.assertEqual(row["adapt_session_ids"][:1], ["s1"])

    def test_interleaved_split_is_independent_of_input_row_order(self) -> None:
        table = _table()
        expected = interleaved_session_manifest(
            table, dataset_id="test", species="mouse"
        )
        shuffled = table.sample(frac=1.0, random_state=42).reset_index(drop=True)

        actual = interleaved_session_manifest(
            shuffled, dataset_id="test", species="mouse"
        )

        self.assertEqual(actual, expected)

    def test_prefix_manifest_preserves_one_session(self) -> None:
        table = _table(sessions=1)
        manifest = prefix_trial_manifest(table, dataset_id="test", species="human")
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(
            manifest["subjects"][0],
            {
                "subject_id": "a",
                "session_id": "s1",
                "adapt_prefix_trials": 2,
                "total_trials": 4,
            },
        )
        shuffled = table.sample(frac=1.0, random_state=42).reset_index(drop=True)
        self.assertEqual(
            prefix_trial_manifest(shuffled, dataset_id="test", species="human"),
            manifest,
        )

    def test_canonical_validation_rejects_noncontiguous_trials(self) -> None:
        table = _table(sessions=1)
        table.loc[1, "trial"] = 7
        with self.assertRaisesRegex(ValueError, "contiguous zero-based"):
            validate_canonical_table(table)

    def test_canonical_validation_accepts_shuffled_rows(self) -> None:
        table = _table().sample(frac=1.0, random_state=42).reset_index(drop=True)
        validate_canonical_table(table)

    def test_write_dataset_round_trip(self) -> None:
        table = _table()
        manifest = interleaved_session_manifest(
            table, dataset_id="test", species="mouse"
        )
        with tempfile.TemporaryDirectory() as directory:
            table_path, manifest_path = write_dataset(
                table, manifest, output_root=Path(directory), stem="test"
            )
            pd.testing.assert_frame_equal(pd.read_parquet(table_path), table)
            self.assertEqual(json.loads(manifest_path.read_text()), manifest)

    def test_lebedeva_adapter_maps_signed_choices_and_binary_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "figure1.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for session in ("2026-01-01", "2026-01-02"):
                    prefix = f"figure1/mouse-a/{session}/trials."
                    self._write_npy(archive, f"{prefix}choices.npy", [-1, 1, 1, -1])
                    self._write_npy(archive, f"{prefix}feedback.npy", [0, 1, 0, 1])
                    self._write_npy(
                        archive, f"{prefix}correct_answers.npy", [-1, -1, 1, 1]
                    )
            source = replace(
                SOURCES["lebedeva"], digest=hashlib.md5(path.read_bytes()).hexdigest()
            )
            expected = {
                "num_subjects": 1,
                "num_sessions": 2,
                "num_trials": 8,
                "excluded_trials": 0,
            }
            with (
                mock.patch.dict(SOURCES, {"lebedeva": source}),
                mock.patch.dict(EXPECTED_AUDITS, {"lebedeva": expected}),
            ):
                table, manifest, audit = adapt_lebedeva(path)

            self.assertEqual(table["animal_response"].tolist()[:4], [0, 1, 1, 0])
            self.assertEqual(table["rewarded"].tolist()[:4], [0, 1, 0, 1])
            self.assertEqual(
                table["reward_probability_arm_0"].tolist()[:4],
                [0.8, 0.8, 0.2, 0.2],
            )
            self.assertEqual(
                manifest["subjects"][0]["adapt_session_ids"], ["2026-01-01"]
            )
            self.assertEqual(audit["num_trials"], 8)

    def test_beron_adapter_preserves_original_trial_and_orders_sessions(self) -> None:
        source_table = pd.DataFrame(
            {
                "Trial": [11, 12, 11, 12],
                "blockTrial": [11, 12, 11, 12],
                "Decision": [1, 0, 0, 1],
                "Switch": [0, 1, 0, 1],
                "Reward": [1, 0, 1, 0],
                "Condition": ["90-10"] * 4,
                "Target": [1, 1, 0, 0],
                "blockLength": [50] * 4,
                "Session": ["m1_2", "m1_2", "m1_10", "m1_10"],
                "Mouse": ["m1"] * 4,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bandit_data.csv"
            source_table.to_csv(path, index=False)
            source = replace(
                SOURCES["beron"], digest=hashlib.md5(path.read_bytes()).hexdigest()
            )
            expected = {
                "num_subjects": 1,
                "num_sessions": 2,
                "num_trials": 4,
                "excluded_trials": 0,
            }
            with (
                mock.patch.dict(SOURCES, {"beron": source}),
                mock.patch.dict(EXPECTED_AUDITS, {"beron": expected}),
            ):
                table, manifest, audit = adapt_beron(path)

            self.assertEqual(set(table["ses_idx"]), {"session-002", "session-010"})
            self.assertEqual(table["source_trial"].tolist(), [11, 12, 11, 12])
            self.assertEqual(
                manifest["subjects"][0]["adapt_session_ids"], ["session-002"]
            )
            self.assertEqual(audit["num_trials"], 4)

    def test_tang_adapter_preserves_sessions_and_binary_actions(self) -> None:
        from scipy.io import savemat

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tang.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for subject, date in (
                    ("V", "20160929"),
                    ("V", "20160930"),
                    ("w", "20160112"),
                    ("w", "20160113"),
                ):
                    payload = io.BytesIO()
                    savemat(
                        payload,
                        {
                            "beh": {
                                "trialDirection": np.array([0, 1]),
                                "reward": np.array([1, 0]),
                                "blockType": np.array([1, 2]),
                                "blockIndex": np.array([1, 1]),
                                "trialObject": np.array([1, 2]),
                                "trialValue": np.array([0.2, 0.8]),
                                "optimal": np.array([0, 1]),
                            }
                        },
                    )
                    archive.writestr(
                        f"Neurophysiology/{subject}{date}_neurons.mat",
                        payload.getvalue(),
                    )
            source = replace(
                SOURCES["tang"], digest=hashlib.sha256(path.read_bytes()).hexdigest()
            )
            expected = {
                "num_subjects": 2,
                "num_sessions": 4,
                "num_trials": 8,
                "excluded_trials": 0,
            }
            with (
                mock.patch.dict(SOURCES, {"tang": source}),
                mock.patch.dict(EXPECTED_AUDITS, {"tang": expected}),
            ):
                table, manifest, audit = adapt_tang(path)

            self.assertEqual(set(table["subject_id"]), {"voltaire", "waldo"})
            self.assertEqual(table["animal_response"].tolist()[:2], [0, 1])
            self.assertEqual(manifest["subjects"][0]["adapt_session_ids"], ["20160929"])
            self.assertEqual(audit["num_trials"], 8)

    def test_alsio_adapter_keeps_complete_supported_cohorts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alsio.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for task, filename, subject_column, choice_column in (
                    (
                        "skf",
                        "Alsio_2019_SKF81297_VPVD_trials.csv",
                        "Subject",
                        "StimChosenPosition",
                    ),
                    (
                        "quin",
                        "Alsio_2019_quinpirole_VPVD_trials.csv",
                        "Subject",
                        "StimChosenPosition",
                    ),
                    (
                        "tsvr",
                        "Alsio_2019_Raclo_SCH39166_TSVR_trials.csv",
                        "ID",
                        "selected_location",
                    ),
                    (
                        "raclo",
                        "Alsio_2019_raclopride_SCH39166_VPVD_trials.csv",
                        "Subject",
                        "StimChosenPosition",
                    ),
                ):
                    base = {
                        subject_column: ["rat-a", "rat-a"],
                        "session_code": [201 if task in {"skf", "tsvr"} else 501] * 2,
                        "trial_within_session": [1, 2],
                        "outcome": [1, 0],
                        choice_column: (
                            ["left", "right"]
                            if choice_column == "StimChosenPosition"
                            else [-1, 1]
                        ),
                    }
                    if choice_column == "StimChosenPosition":
                        base.update({"IsProbe": [0, 1], "StimChosen": ["A", "B"]})
                    else:
                        base.update(
                            {
                                "selected_stimulus": ["A", "B"],
                                "condition": ["vehicle", "vehicle"],
                            }
                        )
                    archive.writestr(filename, pd.DataFrame(base).to_csv(index=False))
            source = replace(
                SOURCES["alsio"], digest=hashlib.sha256(path.read_bytes()).hexdigest()
            )
            expected = {
                "num_subjects": 1,
                "num_sessions": 4,
                "num_trials": 8,
                "excluded_trials": 0,
            }
            with (
                mock.patch.dict(SOURCES, {"alsio": source}),
                mock.patch.dict(EXPECTED_AUDITS, {"alsio": expected}),
            ):
                table, manifest, audit = adapt_alsio(path)

            self.assertEqual(set(table["animal_response"]), {0, 1})
            self.assertEqual(len(manifest["subjects"][0]["adapt_session_ids"]), 2)
            self.assertIn("PRL", audit["excluded_cohort"])

    def test_eckstein_adapter_uses_one_full_session_per_subject(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eckstein.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for subject in (1, 2):
                    table = pd.DataFrame(
                        {
                            "RT": [100, 200, 300, 400],
                            "selected_box": [0, 1, 0, 1],
                            "reward": [1, 0, 1, 0],
                            "correct_box": [0, 0, 1, 1],
                            "sID": [subject] * 4,
                            "TrialID": [1, 2, 3, 4],
                            "rewardversion": [0] * 4,
                            "block": [0, 0, 1, 1],
                            "trialsinceswitch": [0, 1, 0, 1],
                        }
                    )
                    archive.writestr(f"PS_{subject}.csv", table.to_csv(index=False))
            source = replace(
                SOURCES["eckstein"],
                digest=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            expected = {
                "num_subjects": 2,
                "num_sessions": 2,
                "num_trials": 8,
                "excluded_trials": 0,
            }
            with (
                mock.patch.dict(SOURCES, {"eckstein": source}),
                mock.patch.dict(EXPECTED_AUDITS, {"eckstein": expected}),
            ):
                table, manifest, audit = adapt_eckstein(path)

            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["subjects"][0]["adapt_prefix_trials"], 2)
            self.assertEqual(audit["num_subjects"], 2)

    def test_costa_adapter_maps_the_two_stable_shapes(self) -> None:
        from scipy.io import savemat

        rows = []
        for subject in (1, 2):
            for date in (1012020, 1022020):
                for trial, (choice, reward) in enumerate(((10, 1), (20, 0)), 1):
                    row = np.zeros(16, dtype=int)
                    row[[0, 1, 2, 3, 8, 11, 12, 13, 14, 15]] = (
                        subject,
                        date,
                        trial,
                        1,
                        reward,
                        7030,
                        300,
                        choice,
                        1,
                        -1,
                    )
                    rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "costa.mat"
            savemat(path, {"stochasticRL": np.asarray(rows)})
            source = replace(
                SOURCES["costa"], digest=hashlib.md5(path.read_bytes()).hexdigest()
            )
            expected = {
                "num_subjects": 2,
                "num_sessions": 4,
                "num_trials": 8,
                "excluded_trials": 0,
            }
            with (
                mock.patch.dict(SOURCES, {"costa": source}),
                mock.patch.dict(EXPECTED_AUDITS, {"costa": expected}),
            ):
                table, manifest, audit = adapt_costa(path)

            self.assertEqual(table["animal_response"].tolist()[:2], [0, 1])
            self.assertEqual(
                manifest["subjects"][0]["adapt_session_ids"], ["2020-01-01"]
            )
            self.assertEqual(audit["num_trials"], 8)

    def test_lopez_mouse_adapter_excludes_timeout_trials(self) -> None:
        from scipy.io import savemat

        nan_state = np.array([np.nan, np.nan])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lopez.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for subject in ("mouse-a", "mouse-b"):
                    for day in ("01", "02"):
                        trials = []
                        for state_name in ("LeftReward", "NoRewardRight", "timeout"):
                            states = {
                                "LeftReward": nan_state,
                                "NoRewardLeft": nan_state,
                                "RightReward": nan_state,
                                "NoRewardRight": nan_state,
                                "timeout": nan_state,
                            }
                            states[state_name] = np.array([0.0, 1.0])
                            trials.append({"States": states})
                        payload = io.BytesIO()
                        savemat(
                            payload,
                            {"SessionData": {"RawEvents": {"Trial": trials}}},
                        )
                        archive.writestr(
                            f"01_{subject}_FreeChoiceDynamicMatching_"
                            f"Jan{day}_2020_Session1.mat",
                            payload.getvalue(),
                        )
            source = replace(
                SOURCES["lopez_mouse"],
                digest=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            expected = {
                "num_subjects": 2,
                "num_sessions": 4,
                "num_trials": 8,
                "excluded_trials": 4,
            }
            with (
                mock.patch.dict(SOURCES, {"lopez_mouse": source}),
                mock.patch.dict(EXPECTED_AUDITS, {"lopez_mouse": expected}),
            ):
                table, manifest, audit = adapt_lopez_mouse(path)

            self.assertEqual(table["animal_response"].tolist()[:2], [0, 1])
            self.assertEqual(audit["excluded_trials"], 4)
            self.assertEqual(len(manifest["subjects"][0]["adapt_session_ids"]), 1)

    def test_kwak_adapter_preserves_treatment_and_reward_probabilities(self) -> None:
        from scipy.io import savemat

        groups = []
        for group in range(4):
            sessions = np.empty((2, 2), dtype=object)
            for session in range(2):
                for subject in range(2):
                    sessions[session, subject] = np.asarray([[0, 1], [1, 0]])
            groups.append(sessions)
        conditions = []
        times = []
        for _group in range(4):
            group_conditions = np.empty((2, 2), dtype=object)
            group_times = np.empty((2, 2), dtype=object)
            for session in range(2):
                for subject in range(2):
                    group_conditions[session, subject] = np.asarray(
                        [[0.12, 0.72], [0.72, 0.12]]
                    )
                    group_times[session, subject] = np.asarray([[1, 2], [3, 4]])
            conditions.append(group_conditions)
            times.append(group_times)
        payload = io.BytesIO()
        savemat(
            payload,
            {
                "behavior": np.asarray(groups, dtype=object),
                "condition": np.asarray(conditions, dtype=object),
                "time": np.asarray(times, dtype=object),
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data_all.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("TAB_data.mat", payload.getvalue())
            source = replace(
                SOURCES["kwak"], digest=hashlib.md5(path.read_bytes()).hexdigest()
            )
            expected = {
                "num_subjects": 4,
                "num_sessions": 16,
                "num_trials": 32,
                "excluded_trials": 0,
            }
            with (
                mock.patch.dict(SOURCES, {"kwak": source}),
                mock.patch.dict(EXPECTED_AUDITS, {"kwak": expected}),
            ):
                table, manifest, _audit = adapt_kwak(path)

        self.assertEqual(set(table["treatment"]), {"cno", "dmso"})
        # Release encoding is 0=right, 1=left; canonical encoding is the reverse.
        self.assertEqual(table["animal_response"].tolist()[:2], [1, 0])
        self.assertEqual(table["source_choice"].tolist()[:2], [0, 1])
        self.assertEqual(table["reward_probability_arm_0"].tolist()[:2], [0.12, 0.72])
        self.assertEqual(
            manifest["subjects"][0]["adapt_session_ids"],
            ["session-01-cno", "session-02-cno"],
        )

    def test_miller_adapter_excludes_invalid_choices_without_merging_sessions(
        self,
    ) -> None:
        source_subject = {
            "ratname": "rat-a",
            "sides": "lvrrl",
            "rewards": [1, 0, 1, 0, 1],
            "left_prob1": [0.8] * 5,
            "right_prob1": [0.2] * 5,
            "new_sess": [1, 0, 0, 1, 0],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tab_dataset.json"
            path.write_text(json.dumps([source_subject]), encoding="utf-8")
            source = replace(
                SOURCES["miller"], digest=hashlib.md5(path.read_bytes()).hexdigest()
            )
            expected = {
                "num_subjects": 1,
                "num_sessions": 2,
                "num_trials": 4,
                "excluded_trials": 1,
            }
            with (
                mock.patch.dict(SOURCES, {"miller": source}),
                mock.patch.dict(EXPECTED_AUDITS, {"miller": expected}),
            ):
                table, manifest, audit = adapt_miller(path)

        self.assertEqual(table["animal_response"].tolist(), [0, 1, 1, 0])
        self.assertEqual(table["trial"].tolist(), [0, 1, 0, 1])
        self.assertEqual(audit["excluded_trials"], 1)
        self.assertEqual(manifest["subjects"][0]["adapt_session_ids"], ["session-001"])

    def test_findling_adapter_uses_paper_cohort_and_session_then_run_order(
        self,
    ) -> None:
        payload = {
            "A_chosen": np.asarray([0, 1]),
            "reward": np.asarray([1, 0]),
            "Z": np.asarray([0, 1]),
            "tau": np.asarray([0.05, 0.10]),
            "nb_timeout": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "volnoise.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                for subject in (1, 6):
                    for run in range(2):
                        member = tarfile.TarInfo(
                            "Volnoise-test/main/data/python/"
                            f"td_volnoise_subj_{subject}_run_{run}_session_0.pkl"
                        )
                        member_payload = pickle.dumps(payload)
                        member.size = len(member_payload)
                        archive.addfile(member, io.BytesIO(member_payload))
            source = replace(
                SOURCES["findling"],
                digest=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            expected = {
                "num_subjects": 1,
                "num_sessions": 2,
                "num_trials": 4,
                "excluded_trials": 2,
            }
            with (
                mock.patch.dict(SOURCES, {"findling": source}),
                mock.patch.dict(EXPECTED_AUDITS, {"findling": expected}),
            ):
                table, manifest, audit = adapt_findling(path)

        self.assertEqual(set(table["subject_id"]), {"human-01"})
        self.assertEqual(audit["excluded_subject_ids"], [6, 7, 12])
        self.assertEqual(
            manifest["subjects"][0]["adapt_session_ids"],
            ["session-01-run-01"],
        )
