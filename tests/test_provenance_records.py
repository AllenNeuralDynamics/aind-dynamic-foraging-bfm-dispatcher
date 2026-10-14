"""Native launcher records and labeled launch directories are valid provenance."""
import json
import types
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies" / "util"))
from launch_record import load_records
from validate_provenance import check_beaker, check_records_parse, recorded_for_group
from launch_record import experiment_ids_in


class ProvenanceRecordsTests(unittest.TestCase):
    def test_native_launcher_in_labeled_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "launch_record"
            labeled = Path(temp) / "launch_record_paired-rerun"
            labeled.mkdir()
            record = {
                "mode": "resumable-grid", "n_tasks": 3,
                "experiment_id": "experiment-1", "created_utc": "2026-09-08T11:57:48Z",
                "sweep_file": "studies/10/variants/reservoir/sweep.yaml",
                "dispatcher_commit": "a" * 40,
            }
            (labeled / "beaker_resumable.json").write_text(json.dumps(record))
            records = load_records(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["_schema"], "launcher")
            findings = []
            check_records_parse(records, findings)
            self.assertEqual(findings, [])

    def test_records_in_per_launch_subdirectories_are_loaded(self):
        """`launch_beaker_resumable.py` output lives in `launch_record/<label>/`.

        A non-recursive glob never saw these, so a strict check passed without auditing
        them (study 08 had 13 such records and the validator reported one).
        """
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "launch_record"
            for label in ("beaker_a", "beaker_b"):
                (root / label).mkdir(parents=True)
                record = {
                    "mode": "resumable-grid", "n_tasks": 1,
                    "experiment_id": f"experiment-{label}", "created_utc": "2026-09-08T11:57:48Z",
                    "sweep_file": "sweep.yaml", "dispatcher_commit": "a" * 40,
                }
                (root / label / "beaker_resumable.json").write_text(json.dumps(record))
            records = load_records(root)
            self.assertEqual(sorted(r["experiment_id"] for r in records),
                             ["experiment-beaker_a", "experiment-beaker_b"])
            findings = []
            check_records_parse(records, findings)
            self.assertEqual(findings, [])

    def test_beaker_reconciliation_is_scoped_to_the_group_being_checked(self):
        """Recursive discovery loads every launch of a variant, which can span several W&B groups.

        `check_beaker` flags recorded experiments it did not see in the scan; that comparison must
        only use the records for the group being checked, or every other group's launches are
        reported as missing.
        """
        records = [
            {"experiment_id": "exp-a1", "wandb_group": "grp-a"},
            {"experiment_id": "exp-a2", "_meta": {"wandb_groups": ["grp-a", "grp-b"]}},
            {"experiment_id": "exp-b1", "wandb_group": "grp-b"},
            {"experiment_id": "exp-none"},
        ]
        found_in_scan_of_a = {"exp-a1", "exp-a2"}
        for_a = experiment_ids_in(recorded_for_group(records, "grp-a"))
        self.assertEqual(for_a, {"exp-a1", "exp-a2"})
        self.assertEqual(for_a - found_in_scan_of_a, set())          # nothing falsely missing
        # The unscoped comparison, which is what the old code did, reports other groups' launches.
        self.assertEqual(experiment_ids_in(records) - found_in_scan_of_a, {"exp-b1", "exp-none"})
        self.assertEqual(experiment_ids_in(recorded_for_group(records, "grp-b")), {"exp-a2", "exp-b1"})
        self.assertEqual(recorded_for_group(records, "grp-unknown"), [])

    def test_check_beaker_does_not_report_other_groups_launches_as_missing(self):
        """End to end through `check_beaker`, against a fake `beaker` module."""
        def env(group):
            return [types.SimpleNamespace(name="WANDB_RUN_GROUP", value=group)]
        def group_a_spec(envs):
            return types.SimpleNamespace(tasks=[types.SimpleNamespace(env_vars=envs)])

        specs = {"exp-a": group_a_spec(env("grp-a")), "exp-b": group_a_spec(env("grp-b"))}

        class FakeBeaker:
            def __init__(self, *_a, **_k):
                self.workspace = types.SimpleNamespace(
                    experiments=lambda limit=None: [types.SimpleNamespace(id=i) for i in specs])
                self.experiment = types.SimpleNamespace(spec=lambda exp_id: specs[exp_id])

        fake = types.ModuleType("beaker")
        fake.Beaker, fake.Config = FakeBeaker, lambda **_k: None
        records = [{"experiment_id": "exp-a", "wandb_group": "grp-a", "job_refs": [
                        {"type": "beaker_experiment", "id": "exp-a"}]},
                   {"experiment_id": "exp-b", "wandb_group": "grp-b", "job_refs": [
                        {"type": "beaker_experiment", "id": "exp-b"}]},
                   {"experiment_id": "exp-gone", "wandb_group": "grp-b", "job_refs": [
                        {"type": "beaker_experiment", "id": "exp-gone"}]}]
        findings = []
        with mock.patch.dict(sys.modules, {"beaker": fake}), mock.patch.dict(os.environ, {"BEAKER_TOKEN": "x"}):
            check_beaker("grp-a", records, "ws", 50, findings)
        text = " ".join(str(f) for f in findings)
        self.assertNotIn("exp-b", text)       # another group's launch, seen or not, is not this group's business
        self.assertNotIn("exp-gone", text)    # recorded under grp-b, so not expected in grp-a's scan
        self.assertIn("all 1 Beaker experiment(s) for this group are recorded", text)

    def test_native_launcher_still_requires_a_real_commit(self):
        findings = []
        check_records_parse([{
            "_schema": "launcher", "_path": "beaker_resumable.json",
            "mode": "resumable-grid", "n_tasks": 3,
            "experiment_id": "experiment-1", "created_utc": "2026-09-08T11:57:48Z",
            "sweep_file": "sweep.yaml", "dispatcher_commit": None,
        }], findings)
        self.assertEqual(findings[0]["level"], "ERROR")

    def test_interventions_still_require_a_cause(self):
        findings = []
        check_records_parse([{
            "_schema": "current", "_path": "rescue.json", "kind": "rescue",
            "trigger": {"symptom": "stalled"},
        }], findings)
        self.assertEqual(findings[0]["level"], "ERROR")


if __name__ == "__main__":
    unittest.main()
