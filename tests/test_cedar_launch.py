"""Unit tests for code/launch_cedar.py: plan, commands, image record, session start, record.

No Cedar call is made: every ``cedar``/``ssh`` call goes through a fake ``run``. As in
test_beaker_launch.py, ``requests`` and ``beaker-py`` are stubbed so CI needs PyYAML alone.
"""

import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

for _name in ("requests", "beaker"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
if not hasattr(sys.modules["beaker"], "Beaker"):
    sys.modules["beaker"].Beaker = object
    sys.modules["beaker"].Config = object

import launch_cedar as lc  # noqa: E402

SWEEP = """\
entity: NeuralDynamics
project: dfbfm_17-test
name: cedar-smoke
method: grid
command:
  - python
  - -m
  - run_hpc
  - data=synthetic
  - model.training.n_steps=2
  - ${args_no_hyphens}
parameters:
  seed:
    values: [0, 1]
"""

SESSION = """\
workspace: dfbfm
cluster: octo.hub-usc1-gcp-a100
image: dfbfm-wrapper:20261109-wf390d4f
gpu: 1
cpu: 8
ram: 64Gi
max_live_time: 2h
secrets:
  wandb-api-key-neuraldynamics: WANDB_API_KEY
env:
  WRAPPER_REF: main
  DISPATCHER_REF: main
  FORAGING_MODELS_REF: main
  WANDB_RUN_ID: must-be-dropped
"""


class Done:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


class CedarLaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.variant = Path(self.tmp.name) / "studies" / "17-x" / "variants" / "cedar-smoke"
        self.variant.mkdir(parents=True)
        self.sweep = self.variant / "sweep.yaml"
        self.sweep.write_text(SWEEP)
        self.session_file = self.variant / "session_cedar.yaml"
        self.session_file.write_text(SESSION)
        self.cfg = lc.load_session_config(str(self.session_file))

    def tearDown(self):
        self.tmp.cleanup()

    def plan(self):
        return lc.build_plan(str(self.sweep), self.cfg, "smoke", "first Cedar launch of study 17", {})

    def env(self, session):
        return {e["name"]: e["value"] for e in session["envVars"]}

    def test_session_config_needs_every_key_and_a_tag(self):
        self.session_file.write_text(SESSION.replace("gpu: 1\n", ""))
        with self.assertRaises(SystemExit):
            lc.load_session_config(str(self.session_file))
        self.session_file.write_text(SESSION.replace(":20261109-wf390d4f", ""))
        with self.assertRaises(SystemExit):
            lc.load_session_config(str(self.session_file))

    def test_one_session_per_grid_point(self):
        plan = self.plan()
        self.assertRegex(plan["group"], r"^cedar-smoke@\d{8}-\d{6}$")
        names = [s["name"] for s in plan["sessions"]]
        self.assertEqual(len(names), 2)
        self.assertEqual(len(set(names)), 2)
        self.assertTrue(all(n.startswith("cedar-smoke-") for n in names))
        self.assertEqual([s["overrides"] for s in plan["sessions"]], [["seed=0"], ["seed=1"]])
        self.assertEqual([f"seed={i}" in s["run_cmd"] for i, s in enumerate(plan["sessions"])], [True, True])
        self.assertNotIn("${args_no_hyphens}", plan["sessions"][0]["run_cmd"])

    def test_env_carries_group_meta_and_refs_but_not_managed_overrides(self):
        plan = self.plan()
        env = self.env(plan["sessions"][0])
        self.assertEqual(env["WANDB_RUN_GROUP"], plan["group"])
        self.assertEqual(plan["group"], "cedar-smoke@" + env["BFM_META_LAUNCH_ID"])
        self.assertEqual(env["WANDB_RESUME"], "allow")
        self.assertNotEqual(env["WANDB_RUN_ID"], "must-be-dropped")
        self.assertEqual(env["BFM_META_VARIANT"], "cedar-smoke")
        self.assertEqual(env["BFM_META_LABEL"], "smoke")
        for ref in ("WRAPPER_REF", "DISPATCHER_REF", "FORAGING_MODELS_REF"):
            self.assertIn(ref, env)

    def test_create_command_names_secrets_not_their_values(self):
        s = self.plan()["sessions"][0]
        cmd = lc.create_command(self.cfg, s)
        self.assertEqual(cmd[1:4], ["session", "create", s["name"]])
        self.assertIn("--all-ssh-keys", cmd)
        self.assertIn("wandb-api-key-neuraldynamics=WANDB_API_KEY", cmd)
        self.assertIn("octo.hub-usc1-gcp-a100", cmd)
        self.assertIn(f"WANDB_RUN_ID={self.env(s)['WANDB_RUN_ID']}", cmd)

    def test_kickoff_detaches_and_quotes_the_command(self):
        s = {"run_cmd": ["python", "-m", "run_hpc", "data.subject_ids=[1, 2]", "note=it's"]}
        line = lc.kickoff_command(s)
        self.assertIn("nohup setsid bash", line)
        self.assertIn('< /dev/null &', line)
        # the setup ends with `;`, so the trailing `&` backgrounds the run alone, not an
        # `&&` chain whose subshell would hold the SSH connection open
        setup, _, rest = line.partition("; ")
        self.assertIn("|| exit 1", setup)
        self.assertNotIn("&&", rest)
        self.assertIn('BFM_RESUMABLE_OUTPUT_DIR="$CEDAR_OUTPUTS_DIR/run"', line)
        self.assertIn("'data.subject_ids=[1, 2]'", line)
        self.assertIn("'note=it'\"'\"'s'", line)

    def test_image_record_resolves_the_tag_to_its_digest(self):
        versions = {"versions": [{"digest": "sha256:" + "a" * 64, "tags": ["other"]},
                                 {"digest": "sha256:" + "b" * 64, "tags": ["20261109-wf390d4f"]}]}
        rec = lc.image_record_cedar(self.cfg, run=lambda *a, **k: Done(stdout=json.dumps(versions)))
        self.assertEqual(rec, {"platform": "cedar", "name": "dfbfm-wrapper:20261109-wf390d4f",
                               "digest": "sha256:" + "b" * 64})
        rec = lc.image_record_cedar(self.cfg, run=lambda *a, **k: Done(returncode=3, stderr="NotFound"))
        self.assertIsNone(rec["digest"])

    def test_create_then_start_and_fail_loudly(self):
        s = self.plan()["sessions"][0]
        calls = []

        def run(cmd, **kw):
            calls.append(cmd)
            if cmd[1:3] == ["session", "get"]:
                return Done(stdout=json.dumps({"session": {"id": "sess_01MEXAMPLE"}}))
            if cmd[0] == "ssh":
                return Done(stdout="started\n")
            return Done()
        lc.create_session(self.cfg, s, run=run)
        self.assertEqual(lc.session_id(self.cfg, s, run=run), "sess_01MEXAMPLE")
        lc.start_run(self.cfg, s, "sess_01MEXAMPLE", run=run)
        self.assertEqual(calls[-1][-2], f"cedar-dfbfm-{s['name']}")

        with self.assertRaisesRegex(RuntimeError, "did not start"):
            lc.create_session(self.cfg, s, run=lambda cmd, **kw: Done(returncode=9, stderr="NOT_PROGRESSED"))
        with self.assertRaisesRegex(RuntimeError, "id could not be read"):
            lc.session_id(self.cfg, s, run=lambda cmd, **kw: Done(stdout="{}"))
        with self.assertRaisesRegex(RuntimeError, "its run did not"):
            lc.start_run(self.cfg, s, "sess_x", run=lambda cmd, **kw: Done(returncode=255))

    def test_record_is_v2_cedar_with_one_ref_per_session(self):
        plan = self.plan()
        refs = [{"type": "cedar_session", "id": f"sess_{i}", "name": s["name"]} for i, s in enumerate(plan["sessions"])]
        sha = "a" * 40
        provenance = {"studies_git_commit": sha, "dispatcher_git_commit": sha, "wrapper_git_commit": sha,
                      "image": {"platform": "cedar", "name": self.cfg["image"], "digest": "sha256:" + "b" * 64}}
        out = self.variant / "launch_record" / "smoke"
        (self.variant / "launch_record").mkdir()
        lc.save_record(str(self.sweep), str(self.session_file), plan, refs, out, provenance, None,
                       "smoke", "first Cedar launch of study 17")
        rec = json.loads((out / lc.RECORD_NAME).read_text())
        self.assertEqual(rec["platform"], "cedar")
        self.assertEqual(rec["kind"], "initial")
        self.assertEqual([r["id"] for r in rec["job_refs"]], ["sess_0", "sess_1"])
        self.assertEqual(rec["manifests"], ["sweep.yaml", "session_cedar.yaml", lc.RENDERED_NAME])
        self.assertTrue((out / "sweep.yaml").exists() and (out / "session_cedar.yaml").exists())
        self.assertEqual(rec["image"]["digest"], "sha256:" + "b" * 64)

    def test_runs_log_to_the_sweeps_entity_and_project(self):
        cmd = self.plan()["sessions"][0]["run_cmd"]
        self.assertIn("wandb.entity=NeuralDynamics", cmd)
        self.assertIn("wandb.project=dfbfm_17-test", cmd)
        # a command that already says so is not doubled; one that disagrees is refused
        self.sweep.write_text(SWEEP.replace("  - data=synthetic\n", "  - data=synthetic\n  - wandb.project=dfbfm_17-test\n"))
        self.assertEqual(self.plan()["sessions"][0]["run_cmd"].count("wandb.project=dfbfm_17-test"), 1)
        self.sweep.write_text(SWEEP.replace("  - data=synthetic\n", "  - data=synthetic\n  - wandb.project=other\n"))
        with self.assertRaises(SystemExit):
            self.plan()
        self.sweep.write_text(SWEEP + "  wandb.project:\n    values: [a, b]\n")
        with self.assertRaises(SystemExit):
            self.plan()

    def test_a_used_output_dir_is_refused(self):
        from unittest import mock
        out = Path(self.tmp.name) / "dry"
        out.mkdir()
        (out / "cedar_launch.json").write_text("{}")
        argv = ["launch_cedar.py", "--sweep", str(self.sweep), "--session", str(self.session_file),
                "--output-dir", str(out), "--dry-run"]
        with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as cm:
            lc.main()
        self.assertIn("not empty", str(cm.exception))

    def test_a_session_without_a_readable_id_is_still_listed(self):
        plan = self.plan()
        refs = [{"type": "cedar_session", "id": None, "name": plan["sessions"][0]["name"]}]
        out = Path(self.tmp.name) / "partial"
        lc.save_record(str(self.sweep), str(self.session_file), plan, refs, out, {}, None, "p", "a partial launch")
        rec = json.loads((out / lc.RECORD_NAME).read_text())
        self.assertEqual((rec["n_tasks"], rec["n_planned"]), (1, 2))
        self.assertEqual([s["created"] for s in rec["sessions"]], [True, False])

    def test_ctrl_c_still_writes_the_record(self):
        from unittest import mock
        out = Path(self.tmp.name) / "interrupted"
        created = []

        def create(cfg, s, run=None):
            if created:
                raise KeyboardInterrupt
            created.append(s["name"])
        argv = ["launch_cedar.py", "--sweep", str(self.sweep), "--session", str(self.session_file),
                "--output-dir", str(out), "--label", "ctrl-c", "--note", "an interrupted launch"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.multiple(lc, launch_provenance=mock.DEFAULT, check_studies_clean=mock.DEFAULT,
                                    pin_runtime_refs=mock.DEFAULT, runtime_config_reader=mock.DEFAULT,
                                    data_record=mock.DEFAULT, image_record_cedar=mock.DEFAULT,
                                    session_id=mock.DEFAULT, start_run=mock.DEFAULT) as m, \
                mock.patch.object(lc, "create_session", side_effect=create), \
                mock.patch.object(lc.subprocess, "run", return_value=Done()):
            m["launch_provenance"].return_value = {}
            m["check_studies_clean"].return_value = None
            m["pin_runtime_refs"].return_value = {}
            m["data_record"].return_value = {"source": "generated", "config": "synthetic", "type": "synthetic"}
            m["image_record_cedar"].return_value = {"platform": "cedar", "name": "x:y", "digest": None}
            m["session_id"].return_value = "sess_1"
            with self.assertRaises(KeyboardInterrupt):
                lc.main()
        rec = json.loads((out / lc.RECORD_NAME).read_text())
        self.assertEqual([(s["created"], s["id"]) for s in rec["sessions"]], [(True, "sess_1"), (False, None)])

    def test_dry_run_refuses_an_output_dir_inside_the_variant(self):
        from unittest import mock
        argv = ["launch_cedar.py", "--sweep", str(self.sweep), "--session", str(self.session_file),
                "--output-dir", str(self.variant / "launch_record" / "dry"), "--dry-run"]
        with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as cm:
            lc.main()
        self.assertIn("outside the variant", str(cm.exception))

    def test_record_without_sessions_is_not_v2(self):
        out = self.variant / "launch_record" / "dry"
        lc.save_record(str(self.sweep), str(self.session_file), self.plan(), [], out, {}, None, "dry", "a dry run only")
        rec = json.loads((out / lc.RECORD_NAME).read_text())
        self.assertNotIn("platform", rec)          # v2 needs a job ref; a dry run has none
        self.assertEqual(rec["mode"], "cedar-grid")


if __name__ == "__main__":
    unittest.main()
