"""Two-root launch provenance (dispatcher #172, studies split #124).

The contract these tests pin down: a launch record names the studies commit that held the
sweep and the dispatcher commit that ran the launcher as two separate fields, and the two
differ when the sweep lives in another repository. Every test builds that second repository
with ``git init`` in a temp dir, so nothing here depends on the checkout state of this repo
beyond it being a git repo.

Whether the studies repo's validator reads these records is tested in that repo, against
fixture records these launchers wrote (``tests/fixtures/launcher_records/``, #175).
"""

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

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


def _git(cwd: Path, *args: str, ambient: bool = False) -> str:
    """git in ``cwd``. Fixture commands run with git's discovery variables cleared, so a
    sandbox ``GIT_DIR``/``GIT_WORK_TREE`` exported for the dispatcher can never make a
    temp-repo ``init``/``add``/``commit`` touch the dispatcher; ``ambient=True`` keeps the
    env, for the dispatcher's own HEAD (which needs it in the sandbox)."""
    env = dict(os.environ)
    if not ambient:
        for var in provenance._GIT_DISCOVERY_VARS:
            env.pop(var, None)
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True,
                          text=True, env=env).stdout.strip()


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
        self.dispatcher_head = _git(DISPATCHER_ROOT, "rev-parse", "HEAD", ambient=True)

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

    def test_outside_a_repo_everything_is_none_and_the_failure_is_a_warning(self):
        loose = self.tmp / "loose" / "sweep.yaml"
        loose.parent.mkdir()
        loose.write_text(SWEEP)
        prov = provenance.launch_provenance(loose, DISPATCHER_ROOT)
        self.assertIsNone(prov["studies_git_commit"])
        # not a repository: cleanliness is unknown (None), never "clean" ([]), and refused
        self.assertIsNone(prov["studies_dirty_paths"])
        with self.assertRaises(provenance.DirtyStudiesError):
            provenance.check_studies_clean(prov, None)
        override = provenance.check_studies_clean(prov, "ad-hoc sweep outside any repo")
        self.assertIsNone(override["paths"])
        self.assertIn("not a git repository", override["unverified"])
        # #109: a None commit is never silent -- the path and git's message are reported
        self.assertEqual(len(prov["provenance_warnings"]), 1)
        self.assertIn(str(loose.parent.resolve()), prov["provenance_warnings"][0])
        self.assertIn("not a git repository", prov["provenance_warnings"][0])

    def test_sandbox_git_env_for_the_dispatcher_does_not_hijack_the_studies_query(self):
        # The sandbox exports GIT_DIR/GIT_WORK_TREE for the dispatcher tree (#109). They must
        # keep working for the dispatcher and be ignored for a query inside another checkout.
        env = {"GIT_DIR": str(DISPATCHER_ROOT / ".git"), "GIT_WORK_TREE": str(DISPATCHER_ROOT)}
        with mock.patch.dict(os.environ, env):
            prov = provenance.launch_provenance(self.sweep, DISPATCHER_ROOT)
        self.assertEqual(prov["studies_git_commit"], self.head)
        self.assertEqual(prov["dispatcher_git_commit"], self.dispatcher_head)
        self.assertEqual(prov["provenance_warnings"], [])

    def test_installed_vcs_commit_reads_pips_direct_url(self):
        sp = self.tmp / "env" / "lib" / "python3.12" / "site-packages"
        info = sp / "aind_dynamic_foraging_models-0.15.2.dist-info"
        info.mkdir(parents=True)
        self.assertEqual(provenance.installed_vcs_commit([sp], "aind_dynamic_foraging_models"),
                         {"commit": None, "version": "0.15.2", "requested_revision": None})
        (info / "direct_url.json").write_text(json.dumps({
            "url": "https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models.git",
            "vcs_info": {"commit_id": "8" * 40, "requested_revision": "feat/x", "vcs": "git"}}))
        got = provenance.installed_vcs_commit([sp], "aind_dynamic_foraging_models")
        self.assertEqual(got, {"commit": "8" * 40, "version": "0.15.2", "requested_revision": "feat/x"})
        self.assertEqual(provenance.installed_vcs_commit([sp], "other")["version"], None)

    def test_an_ignored_or_untracked_sweep_is_refused_even_though_status_is_clean(self):
        (self.root / ".gitignore").write_text("results/\n")
        _git(self.root, "add", ".gitignore")
        _git(self.root, "commit", "-q", "-m", "ignore results")
        ignored = self.root / "studies" / "08-hb-vs-gru-heldout" / "results" / "sweep.yaml"
        ignored.parent.mkdir()
        ignored.write_text(SWEEP)
        prov = provenance.launch_provenance(ignored, DISPATCHER_ROOT)
        self.assertEqual(prov["studies_dirty_paths"], [])        # git status says nothing ...
        self.assertIs(prov["studies_sweep_tracked"], False)      # ... but the sweep is not committed
        self.assertTrue(any("not tracked" in w for w in prov["provenance_warnings"]))
        with self.assertRaises(provenance.DirtyStudiesError):
            provenance.check_studies_clean(prov, None)
        override = provenance.check_studies_clean(prov, "scratch sweep")
        self.assertTrue(override["untracked_sweep"])
        # the committed sweep is tracked and passes
        self.assertIs(provenance.launch_provenance(self.sweep, DISPATCHER_ROOT)["studies_sweep_tracked"], True)

    def test_a_failed_status_query_is_unknown_not_clean(self):
        prov = provenance.launch_provenance(self.sweep, DISPATCHER_ROOT)
        self.assertEqual(prov["studies_dirty_paths"], [])
        with mock.patch.object(provenance, "_git", return_value=(None, "fatal: index locked")):
            paths, err = provenance.dirty_paths(self.root, self.root / "studies")
        self.assertIsNone(paths)
        self.assertIn("fatal: index locked", err)
        broken = {**prov, "studies_dirty_paths": None,
                  "provenance_warnings": ["studies: " + err]}
        with self.assertRaises(provenance.DirtyStudiesError) as ctx:
            provenance.check_studies_clean(broken, None)
        self.assertIn("could not verify", str(ctx.exception))

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


class TestStudyPathAnchor(unittest.TestCase):
    """#182: a checkout whose own path contains `studies` must not become the study."""

    NESTED = Path("/scratch/wt166/studies/studies/09-gru-cross-species-transfer/variants/"
                  "gru-adaptation-budget/sweep_k_chen.yaml")

    def test_a_studies_named_checkout_still_resolves_the_study_and_variant(self):
        self.assertEqual(provenance.study_and_variant(self.NESTED),
                         ("09-gru-cross-species-transfer", "gru-adaptation-budget"))
        self.assertEqual(resumable._study_variant(self.NESTED),
                         ("09-gru-cross-species-transfer", "gru-adaptation-budget"))
        self.assertEqual(launch_hpc._study_variant(self.NESTED),
                         ("09-gru-cross-species-transfer", "gru-adaptation-budget"))
        self.assertEqual(provenance.study_dir(self.NESTED),
                         Path("/scratch/wt166/studies/studies/09-gru-cross-species-transfer"))

    def test_the_plain_layout_and_off_layout_paths_are_unchanged(self):
        plain = Path("/repo/studies/01-gru-scaling-law/variants/nxd-grid/sweep.yaml")
        self.assertEqual(provenance.study_and_variant(plain), ("01-gru-scaling-law", "nxd-grid"))
        self.assertEqual(provenance.study_and_variant(Path("/repo/studies/08/sweep.yaml")),
                         ("08", None))
        self.assertEqual(resumable._study_variant(Path("/tmp/adhoc_sweep.yaml")),
                         ("adhoc", "adhoc-sweep"))
        self.assertEqual(provenance.study_dir(Path("/tmp/x/sweep.yaml")), Path("/tmp/x"))


class TestBeakerLaunchersCarryBothRoots(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="prov_beaker_"))
        self.root, self.sweep, self.head = make_studies_repo(self.tmp)
        self.exp = self.tmp / "experiment.yaml"
        self.exp.write_text(EXPERIMENT)
        self.dispatcher_head = _git(DISPATCHER_ROOT, "rev-parse", "HEAD", ambient=True)

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
        spec = resumable.build_spec(str(self.sweep), str(self.exp), provenance=prov)
        resumable.save_record(str(self.sweep), "exp-1", 1, out, provenance=prov,
                              runtime_shas={"aind-dynamic-foraging-bfm-wrapper": "a" * 40,
                                            "aind-dynamic-foraging-models": "b" * 40},
                              wandb_group=resumable.spec_group(spec))
        rec = json.loads((out / "beaker_resumable.json").read_text())
        # the exact rendered group, so _meta can find this record for the groups a report reads
        self.assertRegex(rec["wandb_group"], r"^stan-arm@\d{8}-\d{6}$")
        self.assertEqual(rec["wandb_group"], resumable.spec_group(spec))
        self.assertEqual(rec["studies_git_commit"], self.head)
        self.assertEqual(rec["dispatcher_git_commit"], self.dispatcher_head)
        self.assertEqual(rec["dispatcher_commit"], self.dispatcher_head)  # legacy key kept
        self.assertEqual(rec["wrapper_git_commit"], "a" * 40)
        self.assertEqual(rec["foraging_models_git_commit"], "b" * 40)
        self.assertIsNone(rec["studies_dirty_override"])

    def test_native_record_carries_the_contract_fields(self):
        prov = provenance.launch_provenance(self.sweep, DISPATCHER_ROOT)
        out = self.tmp / "out_native"
        launch_beaker.save_record(str(self.sweep), "ent/proj/abc123", None, out, provenance=prov,
                                  runtime_shas={"aind-dynamic-foraging-bfm-wrapper": "a" * 40},
                                  wandb_group="stan-arm@20261023-000000")
        rec = json.loads((out / "beaker_sweep.json").read_text())
        self.assertEqual(rec["wandb_group"], "stan-arm@20261023-000000")
        self.assertEqual(rec["studies_git_commit"], self.head)
        self.assertEqual(rec["dispatcher_git_commit"], self.dispatcher_head)
        self.assertEqual(rec["wrapper_git_commit"], "a" * 40)
        self.assertIsNone(rec["foraging_models_git_commit"])
        self.assertEqual(rec["wandb_sweep_id"], "abc123")

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

    def test_git_info_resolves_the_target_worktree_under_the_sandbox_env(self):
        # The wrapper lineage must describe the wrapper even when the dispatcher's
        # GIT_DIR/GIT_WORK_TREE are exported.
        env = {"GIT_DIR": str(DISPATCHER_ROOT / ".git"), "GIT_WORK_TREE": str(DISPATCHER_ROOT)}
        with mock.patch.dict(os.environ, env):
            info = launch_hpc._git_info(self.root)
        self.assertEqual(info["commit"], self.head)
        self.assertEqual(info["dirty"], "no")

    def test_lineage_and_meta_injection_name_the_studies_commit(self):
        prov = provenance.launch_provenance(self.sweep, DISPATCHER_ROOT)
        lineage = launch_hpc._studies_lineage(prov)
        self.assertEqual(lineage["studies_commit"], self.head)
        self.assertEqual(lineage["studies_git_dirty"], "no")
        cfg = launch_hpc._inject_lineage_into_command({"command": ["python"]}, lineage)
        self.assertTrue(any(c.startswith("+meta.studies_commit=") for c in cfg["command"]))

    def test_runtime_env_site_packages_follows_conda_sh_and_the_slurm_script(self):
        prefix = self.tmp / "miniforge3"
        sp = prefix / "envs" / "dynamic-foraging-bfm-cpu" / "lib" / "python3.12" / "site-packages"
        sp.mkdir(parents=True)
        conda_sh = prefix / "etc" / "profile.d" / "conda.sh"
        conda_sh.parent.mkdir(parents=True)
        conda_sh.write_text("")
        script = self.tmp / "wandb_sweep_cpu.slurm"
        script.write_text("source $CONDA_SH\nconda activate dynamic-foraging-bfm-cpu 2>/dev/null || conda activate disrnn-cpu\n")
        self.assertEqual(launch_hpc._runtime_env_site_packages(str(conda_sh), script), [sp.resolve()])
        self.assertEqual(launch_hpc._runtime_env_site_packages(None, script), [])
        # the scripts fall back to the pre-rename env name; follow the same order they do
        legacy = prefix / "envs" / "disrnn-cpu" / "lib" / "python3.12" / "site-packages"
        legacy.mkdir(parents=True)
        (sp.parent.parent.parent).rename(prefix / "envs" / "gone")
        self.assertEqual(launch_hpc._runtime_env_site_packages(str(conda_sh), script), [legacy.resolve()])

    def test_hpc_record_can_be_written_before_submission_and_filled_in_after(self):
        prov = provenance.launch_provenance(self.sweep, DISPATCHER_ROOT)
        out = self.tmp / "out_hpc_early"
        kw = dict(sweep_file=self.sweep, group="stan-arm@1", n_array_tasks=2, agent_count=1,
                  mode="cpu", provenance=prov, dirty_override=None)
        early = launch_hpc._save_record(out, sweep_path=None, array_jobid=None, autostop_jobid=None, **kw)
        self.assertIsNone(early["wandb_sweep_id"])
        final = launch_hpc._save_record(out, sweep_path="e/p/xyz", array_jobid="7", autostop_jobid="8", **kw)
        rec = json.loads((out / "hpc_sweep.json").read_text())
        self.assertEqual((rec["wandb_sweep_id"], rec["slurm_array_job_id"]), ("xyz", "7"))
        self.assertEqual(rec["wandb_sweep_id"], final["wandb_sweep_id"])
        # an unwritable destination fails at the early write, before anything is submitted
        blocker = self.tmp / "a_file"
        blocker.write_text("")
        with self.assertRaises(OSError):
            launch_hpc._save_record(blocker, sweep_path=None, array_jobid=None, autostop_jobid=None, **kw)

    def test_hpc_record_carries_the_contract_fields(self):
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


class TestLaunchRecordV2(unittest.TestCase):
    """Records match the studies repo's schemas/launch-record.v2.schema.json and the launchers
    refuse, before submitting, a launch that schema or ADR 0001 would reject
    (AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies#14)."""

    NOTE = "baseline launch: does the example converge?"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="prov_v2_"))
        self.root, self.sweep, self.head = make_studies_repo(self.tmp)
        self.exp = self.tmp / "experiment.yaml"
        self.exp.write_text(EXPERIMENT)
        self.prov = provenance.launch_provenance(self.sweep, DISPATCHER_ROOT, {
            "aind-dynamic-foraging-bfm-wrapper": "a" * 40})

    def assert_v2(self, rec: dict, platform: str, manifests: list[str], out: Path) -> None:
        self.assertEqual(rec["_meta"]["schema_version"], 2)
        self.assertEqual(rec["kind"], "initial")
        self.assertEqual(rec["platform"], platform)
        self.assertEqual((rec["label"], rec["note"]), ("arm", self.NOTE))
        self.assertTrue(rec["job_refs"])
        self.assertTrue(all(r["id"] for r in rec["job_refs"]))
        self.assertEqual(rec["revisions"]["studies"], self.head)
        self.assertRegex(rec["revisions"]["dispatcher"], r"^[0-9a-f]{40}$")
        self.assertEqual(rec["revisions"]["wrapper"], "a" * 40)
        self.assertNotIn("foraging_models", rec["revisions"])   # unresolved: left out, not null
        self.assertEqual(rec["manifests"], manifests)
        self.assertTrue((out / manifests[0]).exists())
        # legacy keys stay for older readers (validate_provenance.py, load_records)
        self.assertRegex(rec["dispatcher_commit"], r"^[0-9a-f]{40}$")
        self.assertIn("created_utc", rec)

    def test_resumable_record_is_v2(self):
        out = self.tmp / "out"
        spec = resumable.build_spec(str(self.sweep), str(self.exp), provenance=self.prov)
        resumable.save_record(str(self.sweep), "01KEXP", 1, out, provenance=self.prov,
                              wandb_group=resumable.spec_group(spec), label="arm", note=self.NOTE)
        rec = json.loads((out / "beaker_resumable.json").read_text())
        self.assert_v2(rec, "beaker", ["sweep.yaml", "experiment_resumable_submitted.yaml"], out)
        self.assertEqual(rec["job_refs"], [{"type": "beaker_experiment", "id": "01KEXP"}])
        self.assertEqual(rec["n_tasks"], 1)

    def test_native_record_is_v2(self):
        out = self.tmp / "out_native"
        launch_beaker.save_record(str(self.sweep), "ent/proj/abc123", "01KEXP", out,
                                  provenance=self.prov, wandb_group="stan-arm@20261023-000000",
                                  label="arm", note=self.NOTE, n_tasks=4)
        rec = json.loads((out / "beaker_sweep.json").read_text())
        self.assert_v2(rec, "beaker", ["sweep.yaml", "experiment_submitted.yaml"], out)
        self.assertEqual({r["type"] for r in rec["job_refs"]}, {"wandb_sweep", "beaker_experiment"})

    def test_hpc_record_is_v2_and_drops_refs_not_yet_known(self):
        out = self.tmp / "out_hpc"
        kw = dict(sweep_file=self.sweep, group="stan-arm@20261023-000000", n_array_tasks=2,
                  agent_count=1, mode="cpu", provenance=self.prov, dirty_override=None,
                  label="arm", note=self.NOTE)
        early = launch_hpc._save_record(out, sweep_path=None, array_jobid=None, autostop_jobid=None, **kw)
        self.assertNotIn("kind", early)          # nothing submitted yet: legacy shape
        launch_hpc._save_record(out, sweep_path="e/p/xyz", array_jobid="7", autostop_jobid="8", **kw)
        rec = json.loads((out / "hpc_sweep.json").read_text())
        self.assert_v2(rec, "hpc", ["sweep.yaml"], out)
        self.assertEqual(rec["job_refs"], [{"type": "wandb_sweep", "id": "xyz"},
                                           {"type": "slurm_array_job", "id": "7"}])

    def test_study_launch_identity_is_checked_before_submitting(self):
        ok = "stan-arm@20261023-000000"
        provenance.check_launch_identity(self.sweep, ok, "arm", self.NOTE)        # no raise
        cases = {
            "other@20261023-000000": "is not stan-arm@",
            "stan-arm@2026": "is not stan-arm@",
        }
        for group, msg in cases.items():
            with self.assertRaisesRegex(provenance.LaunchIdentityError, msg):
                provenance.check_launch_identity(self.sweep, group, "arm", self.NOTE)
        with self.assertRaisesRegex(provenance.LaunchIdentityError, "--label"):
            provenance.check_launch_identity(self.sweep, ok, None, self.NOTE)
        with self.assertRaisesRegex(provenance.LaunchIdentityError, "--note"):
            provenance.check_launch_identity(self.sweep, ok, "arm", "short")

    def test_bad_variant_directory_name_is_refused(self):
        bad = self.sweep.parent.parent / "Stan_Arm!" / "sweep.yaml"
        bad.parent.mkdir()
        bad.write_text(SWEEP)
        with self.assertRaisesRegex(provenance.LaunchIdentityError, "variant directory"):
            provenance.check_launch_identity(bad, "Stan_Arm!@20261023-000000", "arm", self.NOTE)

    def test_adhoc_record_keeps_the_legacy_shape(self):
        adhoc = self.tmp / "sweep_mvp.yaml"
        adhoc.write_text(SWEEP)
        out = self.tmp / "out_adhoc"
        resumable.save_record(str(adhoc), "01KEXP", 1, out, provenance=self.prov,
                              wandb_group="sweep-mvp@20261023-000000")
        rec = json.loads((out / "beaker_resumable.json").read_text())
        self.assertNotIn("_meta", rec)
        self.assertNotIn("kind", rec)
        self.assertEqual(rec["mode"], "resumable-grid")

    def test_no_submit_preview_keeps_the_legacy_shape(self):
        out = self.tmp / "out_preview"
        resumable.save_record(str(self.sweep), None, 1, out, provenance=self.prov,
                              wandb_group="stan-arm@20261023-000000", label="arm", note=self.NOTE)
        rec = json.loads((out / "beaker_resumable.json").read_text())
        self.assertNotIn("_meta", rec)
        self.assertEqual(rec["mode"], "resumable-grid")

    def test_native_task_count_sums_replicas(self):
        exp = self.tmp / "replicas.yaml"
        exp.write_text("tasks:\n  - name: a\n    replicas: 4\n  - name: b\n")
        self.assertEqual(launch_beaker._task_count(str(exp)), 5)

    def test_hpc_record_lists_the_submitted_sweep_once_archived(self):
        out = self.tmp / "out_hpc_sub"
        kw = dict(sweep_file=self.sweep, group="stan-arm@20261023-000000", n_array_tasks=2,
                  agent_count=1, mode="cpu", provenance=self.prov, dirty_override=None,
                  label="arm", note=self.NOTE, sweep_path="e/p/xyz", array_jobid="7",
                  autostop_jobid=None)
        self.assertEqual(launch_hpc._save_record(out, **kw)["manifests"], ["sweep.yaml"])
        (out / launch_hpc.SUBMITTED_SWEEP).write_text("name: patched\n")
        self.assertEqual(launch_hpc._save_record(out, **kw)["manifests"],
                         ["sweep.yaml", launch_hpc.SUBMITTED_SWEEP])

    def test_adhoc_sweep_off_the_studies_layout_is_not_checked(self):
        adhoc = self.tmp / "sweep_mvp.yaml"
        adhoc.write_text(SWEEP)
        provenance.check_launch_identity(adhoc, "sweep_mvp@20261023-000000", None, None)


if __name__ == "__main__":
    unittest.main()
