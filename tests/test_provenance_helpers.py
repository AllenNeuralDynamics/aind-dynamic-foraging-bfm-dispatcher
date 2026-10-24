"""Two-root launch provenance (dispatcher #172, studies split #124).

The contract these tests pin down: a launch record names the studies commit that held the
sweep and the dispatcher commit that ran the launcher as two separate fields, and the two
differ when the sweep lives in another repository. Every test builds that second repository
with ``git init`` in a temp dir, so nothing here depends on the checkout state of this repo
beyond it being a git repo.
"""

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies" / "util"))

for _name in ("requests", "beaker"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
if not hasattr(sys.modules["beaker"], "Beaker"):
    sys.modules["beaker"].Beaker = object
    sys.modules["beaker"].Config = object

import provenance  # noqa: E402
import launch_beaker  # noqa: E402
import launch_beaker_resumable as resumable  # noqa: E402
import launch_hpc  # noqa: E402
from _meta import build_meta  # noqa: E402
from launch_record import load_records, refs_in  # noqa: E402
from validate_provenance import check_records_parse  # noqa: E402

DISPATCHER_ROOT = Path(__file__).resolve().parents[1]
SWEEP = """\
entity: AIND-disRNN
project: ai_hub_test
name: unit-test-sweep
method: grid
command: [python, -m, run_hpc, data=synthetic, "${args_no_hyphens}"]
parameters:
  seed:
    values: [1]
"""
EXPERIMENT = """\
version: v2
tasks:
  - name: t
    image: {beaker: img}
    command: [bash, /workspace/entrypoint.sh, wandb, agent, "<SWEEP_ID>"]
    resources: {gpuCount: 1}
    envVars:
      - {name: WRAPPER_REF, value: main}
    result: {path: /results}
"""


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def make_studies_repo(tmp: Path) -> tuple[Path, Path, str]:
    """A second git repo laid out like the studies repo; returns (root, sweep, head sha)."""
    root = tmp / "studies-repo"
    variant = root / "studies" / "08-hb-vs-gru-heldout" / "variants" / "stan-arm"
    variant.mkdir(parents=True)
    sweep = variant / "sweep.yaml"
    sweep.write_text(SWEEP)
    (root / "README.md").write_text("studies\n")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root, sweep, _git(root, "rev-parse", "HEAD")


class TestLaunchProvenance(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="prov_"))
        self.root, self.sweep, self.head = make_studies_repo(self.tmp)
        self.dispatcher_head = _git(DISPATCHER_ROOT, "rev-parse", "HEAD")

    def test_studies_and_dispatcher_shas_are_distinct_and_correct(self):
        prov = provenance.launch_provenance(self.sweep, DISPATCHER_ROOT)
        self.assertEqual(prov["studies_git_commit"], self.head)
        self.assertEqual(prov["dispatcher_git_commit"], self.dispatcher_head)
        self.assertNotEqual(prov["studies_git_commit"], prov["dispatcher_git_commit"])
        self.assertIs(prov["studies_git_dirty"], False)
        self.assertEqual(prov["studies_dirty_paths"], [])

    def test_runtime_shas_are_recorded_only_when_full_shas(self):
        prov = provenance.launch_provenance(self.sweep, DISPATCHER_ROOT, {
            "aind-dynamic-foraging-bfm-wrapper": "a" * 40,
            "aind-dynamic-foraging-models": "main",   # a branch is not provenance
        })
        self.assertEqual(prov["wrapper_git_commit"], "a" * 40)
        self.assertIsNone(prov["foraging_models_git_commit"])

    def test_runtime_shas_from_pins_collapses_one_sha_and_keeps_two(self):
        shas = provenance.runtime_shas_from_pins({
            ("aind-dynamic-foraging-bfm-wrapper", "main"): "a" * 40,
            ("aind-dynamic-foraging-bfm-wrapper", "feat"): "a" * 40,
            ("aind-dynamic-foraging-models", "v1"): "b" * 40,
            ("aind-dynamic-foraging-models", "v2"): "c" * 40,
        })
        self.assertEqual(shas["aind-dynamic-foraging-bfm-wrapper"], "a" * 40)
        self.assertEqual(shas["aind-dynamic-foraging-models"], ["b" * 40, "c" * 40])

    def test_outside_a_repo_everything_is_none(self):
        loose = self.tmp / "loose" / "sweep.yaml"
        loose.parent.mkdir()
        loose.write_text(SWEEP)
        prov = provenance.launch_provenance(loose, DISPATCHER_ROOT)
        self.assertIsNone(prov["studies_git_commit"])
        self.assertEqual(prov["studies_dirty_paths"], [])

    def test_dirty_study_dir_is_refused_and_an_unrelated_edit_is_not(self):
        (self.root / "README.md").write_text("edited elsewhere\n")
        prov = provenance.launch_provenance(self.sweep, DISPATCHER_ROOT)
        self.assertIs(prov["studies_git_dirty"], True)          # recorded ...
        self.assertIsNone(provenance.check_studies_clean(prov, None))  # ... but not refused

        self.sweep.write_text(SWEEP + "  extra:\n    values: [2]\n")
        prov = provenance.launch_provenance(self.sweep, DISPATCHER_ROOT)
        self.assertEqual(prov["studies_dirty_paths"],
                         ["studies/08-hb-vs-gru-heldout/variants/stan-arm/sweep.yaml"])
        with self.assertRaises(provenance.DirtyStudiesError):
            provenance.check_studies_clean(prov, None)
        override = provenance.check_studies_clean(prov, "probe of an unsaved sweep edit")
        self.assertEqual(override["reason"], "probe of an unsaved sweep edit")
        self.assertEqual(provenance.record_fields(prov, override)["studies_dirty_override"],
                         override)


class TestBeakerLaunchersCarryBothRoots(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="prov_beaker_"))
        self.root, self.sweep, self.head = make_studies_repo(self.tmp)
        self.exp = self.tmp / "experiment.yaml"
        self.exp.write_text(EXPERIMENT)
        self.dispatcher_head = _git(DISPATCHER_ROOT, "rev-parse", "HEAD")

    def test_resumable_spec_passes_the_studies_commit_to_the_wrapper(self):
        spec = resumable.build_spec(str(self.sweep), str(self.exp))
        for task in spec["tasks"]:
            env = {e["name"]: e["value"] for e in task["envVars"]}
            self.assertEqual(env["BFM_META_STUDIES_COMMIT"], self.head)
            self.assertNotEqual(env["BFM_META_STUDIES_COMMIT"], self.dispatcher_head)
            self.assertEqual(env["BFM_META_STUDY"], "08-hb-vs-gru-heldout")

    def test_resumable_record_carries_the_contract_fields(self):
        prov = provenance.launch_provenance(self.sweep, DISPATCHER_ROOT)
        out = self.tmp / "out"
        resumable.save_record(str(self.sweep), "exp-1", 1, out, provenance=prov,
                              runtime_shas={"aind-dynamic-foraging-bfm-wrapper": "a" * 40,
                                            "aind-dynamic-foraging-models": "b" * 40})
        rec = json.loads((out / "beaker_resumable.json").read_text())
        self.assertEqual(rec["studies_git_commit"], self.head)
        self.assertEqual(rec["dispatcher_git_commit"], self.dispatcher_head)
        self.assertEqual(rec["dispatcher_commit"], self.dispatcher_head)  # legacy key kept
        self.assertEqual(rec["wrapper_git_commit"], "a" * 40)
        self.assertEqual(rec["foraging_models_git_commit"], "b" * 40)
        self.assertIsNone(rec["studies_dirty_override"])
        # and the validator accepts it as a launcher record
        findings = []
        check_records_parse(load_records(out), findings)
        self.assertEqual(findings, [])

    def test_native_record_carries_the_contract_fields(self):
        prov = provenance.launch_provenance(self.sweep, DISPATCHER_ROOT)
        out = self.tmp / "out_native"
        launch_beaker.save_record(str(self.sweep), "ent/proj/abc123", None, out, provenance=prov,
                                  runtime_shas={"aind-dynamic-foraging-bfm-wrapper": "a" * 40})
        rec = json.loads((out / "beaker_sweep.json").read_text())
        self.assertEqual(rec["studies_git_commit"], self.head)
        self.assertEqual(rec["dispatcher_git_commit"], self.dispatcher_head)
        self.assertEqual(rec["wrapper_git_commit"], "a" * 40)
        self.assertIsNone(rec["foraging_models_git_commit"])
        self.assertEqual(refs_in(load_records(out), "wandb_sweep"), {"abc123"})

    def test_native_render_manages_the_studies_commit_env(self):
        # A template that already carries the key is replaced, not doubled.
        doubled = EXPERIMENT.replace("      - {name: WRAPPER_REF, value: main}",
                                     "      - {name: BFM_META_STUDIES_COMMIT, value: stale}\n"
                                     "      - {name: WRAPPER_REF, value: main}")
        exp = self.tmp / "doubled.yaml"
        exp.write_text(doubled)
        spec = launch_beaker._render_experiment(
            str(exp), "e/p/s", "g@1", [{"name": "BFM_META_STUDIES_COMMIT", "value": "s" * 40}])
        env = [e for e in spec["tasks"][0]["envVars"] if e["name"] == "BFM_META_STUDIES_COMMIT"]
        self.assertEqual([e["value"] for e in env], ["s" * 40])


class TestHpcLauncherCarriesBothRoots(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="prov_hpc_"))
        self.root, self.sweep, self.head = make_studies_repo(self.tmp)

    def test_lineage_and_meta_injection_name_the_studies_commit(self):
        prov = provenance.launch_provenance(self.sweep, DISPATCHER_ROOT)
        lineage = launch_hpc._studies_lineage(prov)
        self.assertEqual(lineage["studies_commit"], self.head)
        self.assertEqual(lineage["studies_git_dirty"], "no")
        cfg = launch_hpc._inject_lineage_into_command({"command": ["python"]}, lineage)
        self.assertTrue(any(c.startswith("+meta.studies_commit=") for c in cfg["command"]))

    def test_hpc_record_is_a_launcher_record_the_validator_reads(self):
        prov = provenance.launch_provenance(self.sweep, DISPATCHER_ROOT, {
            "aind-dynamic-foraging-bfm-wrapper": "a" * 40})
        out = self.tmp / "out_hpc"
        launch_hpc._save_record(
            out, sweep_file=self.sweep, sweep_path="AIND-disRNN/proj/xyz789", group="stan-arm@1",
            array_jobid="123", autostop_jobid="124", n_array_tasks=2, agent_count=1,
            mode="cpu", provenance=prov, dirty_override=None,
        )
        rec = json.loads((out / "hpc_sweep.json").read_text())
        self.assertEqual(rec["mode"], "hpc-sweep")
        self.assertEqual(rec["wandb_sweep_id"], "xyz789")
        self.assertEqual(rec["studies_git_commit"], self.head)
        self.assertEqual(rec["wrapper_git_commit"], "a" * 40)
        self.assertTrue((out / "sweep.yaml").exists())
        records = load_records(out)
        self.assertEqual(records[0]["_schema"], "launcher")
        self.assertEqual(refs_in(records, "wandb_sweep"), {"xyz789"})
        findings = []
        check_records_parse(records, findings)
        self.assertEqual(findings, [])


class TestAnalysisMetaNamesTheStudiesCheckout(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="prov_meta_"))
        self.root, self.sweep, self.head = make_studies_repo(self.tmp)
        self.study_root = self.root / "studies" / "08-hb-vs-gru-heldout"
        rec_dir = self.study_root / "variants" / "stan-arm" / "launch_record" / "beaker_x"
        rec_dir.mkdir(parents=True)
        (rec_dir / "beaker_resumable.json").write_text(json.dumps({
            "mode": "resumable-grid", "wandb_group": "stan-arm@1",
            "dispatcher_git_commit": "d" * 40, "dispatcher_commit": "d" * 40,
            "studies_git_commit": "1" * 40,
            "wrapper_git_commit": "2" * 40, "foraging_models_git_commit": "f" * 40,
        }))
        (self.study_root / "variants" / "stan-arm" / "launch_record" / "hpc_stan.json").write_text(
            json.dumps({"_meta": {"wandb_groups": ["stan-arm@2"]}, "kind": "resubmit",
                        "launch": {"revisions": {"dispatcher_commit": "e" * 40,
                                                 "wrapper_commit": "2" * 40}}}))

    def test_source_shas_come_from_the_launch_records_of_the_groups_read(self):
        meta = build_meta("analysis/x.py", ["stan-arm@1", "stan-arm@2"], study_root=self.study_root)
        self.assertEqual(meta["source_dispatcher_git_shas"], ["d" * 40, "e" * 40])
        self.assertEqual(meta["source_wrapper_git_shas"], ["2" * 40])
        self.assertEqual(meta["source_foraging_models_git_shas"], ["f" * 40])
        self.assertEqual(meta["source_studies_git_shas"], ["1" * 40])
        self.assertEqual(meta["provenance_findings"], [])
        self.assertNotIn("dispatcher_git_sha", meta)   # never its own HEAD under that name
        self.assertRegex(meta["studies_git_sha"], r"^[0-9a-f]{40}$")
        self.assertIn(meta["studies_git_dirty"], (True, False))

    def test_a_group_without_a_record_is_an_explicit_finding_not_a_guess(self):
        meta = build_meta("analysis/x.py", ["stan-arm@1", "stan-arm@none"], study_root=self.study_root)
        self.assertEqual(meta["source_dispatcher_git_shas"], ["d" * 40])
        self.assertEqual(meta["provenance_findings"],
                         ["no launch record found for W&B group(s): stan-arm@none"])

    def test_without_a_study_root_the_arrays_are_empty_with_a_finding(self):
        meta = build_meta("analysis/x.py", ["g@1"])
        self.assertEqual(meta["source_dispatcher_git_shas"], [])
        self.assertTrue(meta["provenance_findings"])


if __name__ == "__main__":
    unittest.main()
