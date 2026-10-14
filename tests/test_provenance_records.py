"""Native launcher records and labeled launch directories are valid provenance."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies" / "util"))
from launch_record import load_records
from validate_provenance import check_records_parse


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
