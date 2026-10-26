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
        self.assertEqual(self.record(["${args_no_hyphens}"], {"data.snapshot": {"value": "20260702"}})["snapshot"], "20260702")
        # a command without an ${args...} macro never receives the parameters
        self.assertEqual(self.record(parameters={"data.snapshot": {"value": "20260702"}})["snapshot"], "20260603")

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
            self.record(["${args_no_hyphens}"], {"data.snapshot": {"values": ["20260603", "20260701"]}})

    def test_hydra_prefixes_and_nulls_clear_the_snapshot(self):
        for params in ({"++data.snapshot": {"value": "null"}}, {"+data.snapshot": {"value": None}}):
            d = self.record(["data=mice_snapshot", "${args_no_hyphens}"], params)
            self.assertEqual(d["source"], "live")
        self.assertEqual(self.record(["data=mice_snapshot", "~data.snapshot"])["source"], "live")

    def test_overrides_apply_in_command_order(self):
        params = {"data.snapshot": {"value": "20260701"}}
        # the swept parameter expands at ${args...}, so it wins over an earlier command item ...
        self.assertEqual(self.record(["data.snapshot=20260603", "${args_no_hyphens}"], params)["snapshot"], "20260701")
        # ... and loses to a later one
        self.assertEqual(self.record(["${args_no_hyphens}", "data.snapshot=20260603"], params)["snapshot"], "20260603")

    def test_several_data_configs_are_refused(self):
        with self.assertRaises(li.DataConfigError):
            self.record(["${args_no_hyphens}"], {"data": {"values": ["mice", "mice_snapshot"]}})
        # a sweep of one value is fine
        self.assertEqual(self.record(["${args_no_hyphens}"], {"data": {"values": ["mice_snapshot"]}})["config"], "mice_snapshot")

    def test_configs_are_read_through_the_given_reader(self):
        """The Beaker launchers pass the configs at the run's pinned dispatcher ref."""
        at_ref = {"config.yaml": "defaults:\n  - data: mice_snapshot\n",
                  "data/mice_snapshot.yaml": "type: mice_snapshot\nsnapshot: '20260801'\n"}
        d = li.data_record({"command": []}, at_ref.get, fetch=fetch)
        self.assertEqual(d["snapshot"], "20260801")
        warnings = []
        reader = li.runtime_config_reader(self.root, "f" * 40, warnings)          # not a git checkout
        self.assertTrue(warnings and reader("config.yaml") is None)              # falls back, and says so

    def test_hydra_scalars_quotes_and_interpolations(self):
        self.assertEqual(self.record(["data=mice_snapshot", "data.snapshot=NULL"])["source"], "live")
        self.assertEqual(self.record(["data=mice_snapshot", "data.snapshot='20260701'"])["snapshot"], "20260701")
        self.assertEqual(self.record(["data=mice_snapshot", "data.snapshot='null'"])["snapshot"], "null")  # a quoted literal
        with self.assertRaises(li.DataConfigError):
            self.record(["data=mice_snapshot", "data.snapshot=${oc.env:SNAPSHOT,null}"])

    def test_a_non_hydra_runner_reads_external_data(self):
        sweep = {"command": ["python", "-m", "external_transfer.gru_transfer", "--dataset=zid", "${args_no_hyphens}"]}
        d = li.data_record(sweep, self.root, fetch=fetch)
        self.assertEqual(d, {"source": "external", "config": "external_transfer.gru_transfer --dataset=zid", "type": None})
        self.assertIsNone(li.live_data_refusal(d))

    def test_quoted_interpolation_and_script_multirun(self):
        with self.assertRaises(li.DataConfigError):
            self.record(["data=mice_snapshot", "data.snapshot='${oc.env:SNAPSHOT,null}'"])
        # `-m` after a script is Hydra multirun, not a module: still a Hydra run, so live data is refused
        d = li.data_record({"command": ["python", "run_hpc.py", "-m", "data=mice_snapshot", "data.snapshot=null"]},
                           self.root, fetch=fetch)
        self.assertEqual((d["source"], d["config"]), ("live", "mice_snapshot"))
        self.assertEqual(li._runner({"program": "run_hpc.py", "command": ["${env}", "${interpreter}", "${program}", "${args}"]}), "run_hpc")

    def test_dataset_given_as_a_separate_token(self):
        d = li.data_record({"command": ["python", "-m", "external_transfer.gru_transfer", "--dataset", "zid"]}, self.root)
        self.assertEqual(d["config"], "external_transfer.gru_transfer --dataset=zid")

    def test_hydra_sweep_expressions_are_refused(self):
        for value in ("20260603,null", "choice(20260603,20260701)"):
            with self.assertRaises(li.DataConfigError):
                li.data_record({"command": ["python", "run_hpc.py", "-m", "data=mice_snapshot", f"data.snapshot={value}"]},
                               self.root, fetch=fetch)

    def test_native_args_macro_maps_parameters_to_flags(self):
        base = ["python", "-m", "external_transfer.gru_transfer", "--dataset=zid"]
        params = {"dataset": {"value": "grossman"}}
        d = li.data_record({"command": [*base, "${args}"], "parameters": params}, self.root)
        self.assertEqual(d["config"], "external_transfer.gru_transfer --dataset=grossman")    # ${args} -> --dataset
        d = li.data_record({"command": [*base, "${args_no_hyphens}"], "parameters": params}, self.root)
        self.assertEqual(d["config"], "external_transfer.gru_transfer --dataset=zid")         # dataset=..., not a flag

    def test_tasks_on_different_images_are_refused(self):
        spec = {"tasks": [{"image": {"beaker": "a/img-1"}}, {"image": {"beaker": "a/img-2"}}]}
        with self.assertRaises(li.DataConfigError):
            li.image_record_beaker(spec, "ws", client_factory=lambda ws: None)

    def test_wrapper_config_root_is_the_dispatcher_beside_the_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "aind-disrnn-dispatcher" / "code" / "config").mkdir(parents=True)
            (root / "aind-dynamic-foraging-bfm-wrapper").mkdir()
            self.assertEqual(li.wrapper_config_root(root / "aind-dynamic-foraging-bfm-wrapper"),
                             root.resolve() / "aind-disrnn-dispatcher" / "code" / "config")

    def test_tasks_on_different_dispatcher_commits_are_refused(self):
        self.assertEqual(li.runtime_dispatcher_sha({"aind-dynamic-foraging-bfm-dispatcher": "a" * 40}), "a" * 40)
        with self.assertRaises(li.DataConfigError):
            li.runtime_dispatcher_sha({"aind-dynamic-foraging-bfm-dispatcher": ["a" * 40, "b" * 40]})

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
