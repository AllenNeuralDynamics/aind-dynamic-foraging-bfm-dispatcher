"""What a launch trains on and runs in (code/launch_inputs.py, studies#25).

Network and Beaker are stubbed: `data_record` takes the manifest fetch as an argument and
`image_record_beaker` the client factory, so these tests need neither.
"""

import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import launch_inputs as li  # noqa: E402
from provenance import v2_record_fields  # noqa: E402

DIGEST = "a" * 64


def fetch(snapshot):
    return DIGEST, {"latest_build_id": "20260630T054811Z", "n_sessions": 23585}


class DataRecordTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data").mkdir()
        (self.root / "config.yaml").write_text("defaults:\n  - data: mice_snapshot\n  - model: disrnn\n")
        (self.root / "data/mice_snapshot.yaml").write_text("type: mice_snapshot\nsnapshot: '20260603'\n")
        (self.root / "data/mice_live.yaml").write_text("type: mice_snapshot\nsnapshot: null\n")
        (self.root / "data/synthetic_hierarchical.yaml").write_text("type: synthetic_hierarchical\n")
        (self.root / "data/mice.yaml").write_text("type: mice\n")

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, command=(), parameters=None):
        return li.data_record({"command": ["python", "-m", "run_hpc", *command], "parameters": parameters or {}},
                              self.root, fetch=fetch)

    def test_pinned_snapshot_is_digested(self):
        d = self.record(["data=mice_snapshot"])
        self.assertEqual(d, {"source": "snapshot", "config": "mice_snapshot", "type": "mice_snapshot",
                             "snapshot": "20260603", "manifest_sha256": DIGEST, "build_id": "20260630T054811Z",
                             "uri": "s3://aind-scratch-data/aind-dynamic-foraging-cache/snapshots/20260603/",
                             "n_sessions": 23585})
        self.assertIsNone(li.live_data_refusal(d))

    def test_defaults_and_overrides(self):
        self.assertEqual(self.record()["config"], "mice_snapshot")                    # Hydra default
        self.assertEqual(self.record(["data.snapshot=20260701"])["snapshot"], "20260701")
        self.assertEqual(self.record(parameters={"data.snapshot": {"value": "20260702"}})["snapshot"], "20260702")

    def test_live_and_unpinned_data_are_refused(self):
        for d in (self.record(["data=mice_live"]), self.record(["data=mice"])):
            self.assertEqual((d["source"], d["snapshot"]), ("live", None))
            self.assertIn("--allow-live-data", li.live_data_refusal(d))

    def test_generated_data_needs_no_snapshot(self):
        d = self.record(["data=synthetic_hierarchical"])
        self.assertEqual(d, {"source": "generated", "config": "synthetic_hierarchical", "type": "synthetic_hierarchical"})
        self.assertIsNone(li.live_data_refusal(d))

    def test_a_swept_snapshot_is_refused(self):
        with self.assertRaises(li.DataConfigError):
            self.record(parameters={"data.snapshot": {"values": ["20260603", "20260701"]}})

    def test_unreadable_manifest_keeps_the_snapshot(self):
        d = li.data_record({"command": ["data=mice_snapshot"]}, self.root, fetch=lambda s: (None, {}))
        self.assertEqual((d["source"], d["snapshot"], d["manifest_sha256"]), ("snapshot", "20260603", None))


class ImageRecordTest(unittest.TestCase):
    TEMPLATE = {"tasks": [{"image": {"beaker": "han-hou/disrnn-wrapper-main-20260712"}}]}

    def test_beaker_name_resolves_to_its_id(self):
        client = types.SimpleNamespace(image=types.SimpleNamespace(get=lambda name: types.SimpleNamespace(id="01KIMAGE")))
        self.assertEqual(li.image_record_beaker(self.TEMPLATE, "ws", client_factory=lambda ws: client),
                         {"platform": "beaker", "name": "han-hou/disrnn-wrapper-main-20260712", "id": "01KIMAGE"})

    def test_unresolvable_beaker_image_keeps_its_name(self):
        def broken(ws):
            raise KeyError("BEAKER_TOKEN")
        self.assertEqual(li.image_record_beaker(self.TEMPLATE, "ws", client_factory=broken)["id"], None)
        self.assertIsNone(li.image_record_beaker({"tasks": [{"image": {"docker": "x"}}]}, "ws"))

    def test_hpc_env_digest_follows_its_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "envs" / "bfm"
            sp = prefix / "lib" / "python3.12" / "site-packages"
            (prefix / "conda-meta").mkdir(parents=True)
            sp.mkdir(parents=True)
            (prefix / "conda-meta" / "numpy-2.1.0-py312.json").write_text("{}")
            (sp / "jax-0.4.30.dist-info").mkdir()
            first = li.image_record_hpc([sp])
            self.assertEqual(first["conda_env"], str(prefix))
            self.assertRegex(first["lock_sha256"], "^[0-9a-f]{64}$")
            (sp / "jax-0.4.31.dist-info").mkdir()
            self.assertNotEqual(li.image_record_hpc([sp])["lock_sha256"], first["lock_sha256"])
        self.assertEqual(li.image_record_hpc([]), {"platform": "hpc", "conda_env": "unresolved", "lock_sha256": None})


class RecordFieldsTest(unittest.TestCase):
    def test_v2_record_carries_data_and_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            sweep = Path(tmp) / "studies" / "12-gru-x" / "variants" / "base" / "sweep.yaml"
            sweep.parent.mkdir(parents=True)
            sweep.write_text("name: base\n")
            data = {"source": "generated", "config": "synthetic", "type": "synthetic"}
            image = {"platform": "hpc", "conda_env": "/envs/bfm", "lock_sha256": None}
            rec = v2_record_fields({"data": data, "image": image}, sweep_path=sweep, platform="hpc",
                                   wandb_group="base@20261026-000000", label="base", note="a launch note",
                                   job_refs=[{"type": "slurm_array_job", "id": "1"}], manifests=["sweep.yaml"],
                                   n_tasks=1, produced_by="launch_hpc.py")
            self.assertEqual((rec["data"], rec["image"]), (data, image))


if __name__ == "__main__":
    unittest.main()
