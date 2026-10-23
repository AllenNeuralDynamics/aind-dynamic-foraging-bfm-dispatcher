"""Unit tests for the dispatcher's launcher helpers.

These are the pure functions -- strings, dicts, paths -- that every launch goes
through and that nothing previously covered. See #97: the whole rename
(RUNTIME_REF_REPOSITORIES, BFM_* emission, the nested oc.env fallbacks, the
SLURM export) was verified by one --no-submit dry-run and `bash -n`.

Only `yaml` is needed beyond the stdlib; nothing here touches SLURM, Beaker,
W&B or the network.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import launch_hpc  # noqa: E402


class TestParseSweepId(unittest.TestCase):
    def test_extracts_from_wandb_agent_line(self):
        out = "wandb: Run sweep agent with:\nwandb agent AIND-disRNN/proj/abc123\n"
        self.assertEqual(launch_hpc._parse_sweep_id(out, ""), "AIND-disRNN/proj/abc123")

    def test_reads_stderr_too(self):
        self.assertEqual(
            launch_hpc._parse_sweep_id("", "wandb agent ent/pro/xyz789"),
            "ent/pro/xyz789",
        )

    def test_bare_id_without_entity_is_rejected(self):
        # "Created sweep with ID: abc" has no entity/project, so it cannot be
        # used as an agent target; the helper returns None rather than a
        # half-qualified id that fails later at submit time.
        self.assertIsNone(launch_hpc._parse_sweep_id("Created sweep with ID: abc123", ""))

    def test_no_match_returns_none(self):
        self.assertIsNone(launch_hpc._parse_sweep_id("nothing here", ""))


class TestArraySpec(unittest.TestCase):
    def test_extract_equals_form(self):
        self.assertEqual(launch_hpc._extract_array_spec("--array=0-7 --mem=4G"), "0-7")

    def test_extract_space_form(self):
        self.assertEqual(launch_hpc._extract_array_spec("--array 0-3"), "0-3")

    def test_absent_returns_none(self):
        self.assertIsNone(launch_hpc._extract_array_spec("--mem=4G"))

    def test_count_simple_range(self):
        self.assertEqual(launch_hpc._count_array_tasks("0-7"), 8)

    def test_count_respects_throttle_suffix(self):
        # "%4" caps concurrency; it must not change the task count.
        self.assertEqual(launch_hpc._count_array_tasks("0-7%4"), 8)

    def test_count_with_step(self):
        self.assertEqual(launch_hpc._count_array_tasks("0-10:2"), 6)

    def test_count_comma_list(self):
        self.assertEqual(launch_hpc._count_array_tasks("1,3,5"), 3)

    def test_count_single_index(self):
        self.assertEqual(launch_hpc._count_array_tasks("4"), 1)

    def test_rejects_inverted_range(self):
        with self.assertRaises(ValueError):
            launch_hpc._count_array_tasks("7-0")

    def test_rejects_zero_step(self):
        with self.assertRaises(ValueError):
            launch_hpc._count_array_tasks("0-8:0")


class TestLoadExportEnv(unittest.TestCase):
    def _write(self, body: str) -> Path:
        d = Path(tempfile.mkdtemp(prefix="user_env_"))
        p = d / "user.env"
        p.write_text(body)
        self.addCleanup(lambda: p.unlink(missing_ok=True))
        return p

    def test_reads_exports_and_ignores_other_lines(self):
        p = self._write(
            "# a comment\n"
            "\n"
            "export A=1\n"
            "NOT_EXPORTED=2\n"
            "export B=two\n"
        )
        values = launch_hpc._load_export_env(p)
        self.assertEqual(values["A"], "1")
        self.assertEqual(values["B"], "two")
        self.assertNotIn("NOT_EXPORTED", values)

    @mock.patch.dict(os.environ, {"HOME": "/home/someone"}, clear=False)
    def test_expands_variables_from_the_environment(self):
        p = self._write('export OUT="$HOME/outputs"\n')
        self.assertEqual(launch_hpc._load_export_env(p)["OUT"], "/home/someone/outputs")

    def test_expands_variables_defined_earlier_in_the_same_file(self):
        p = self._write("export BASE=/scratch\nexport SUB=$BASE/runs\n")
        self.assertEqual(launch_hpc._load_export_env(p)["SUB"], "/scratch/runs")


class TestPathForMeta(unittest.TestCase):
    def test_relative_when_inside_root(self):
        root = Path("/repo")
        self.assertEqual(
            launch_hpc._path_for_meta(Path("/repo/studies/01/sweep.yaml"), root),
            "studies/01/sweep.yaml",
        )

    def test_absolute_when_outside_root(self):
        # A sweep passed from elsewhere must still be recorded, not crash.
        self.assertEqual(
            launch_hpc._path_for_meta(Path("/tmp/other/sweep.yaml"), Path("/repo")),
            "/tmp/other/sweep.yaml",
        )


class TestSeattleLaunchId(unittest.TestCase):
    def test_shape_is_stable(self):
        # AGENTS.md §7: human-facing stamps are Seattle time.
        lid = launch_hpc._seattle_launch_id()
        self.assertRegex(lid, r"^\d{8}-\d{6}$")


class TestGitInfo(unittest.TestCase):
    def test_reports_unknown_outside_a_repo(self):
        with tempfile.TemporaryDirectory() as td:
            info = launch_hpc._git_info(Path(td))
            self.assertEqual(info["commit"], "unknown")
            self.assertEqual(info["dirty"], "no")

    def test_reports_a_real_repo(self):
        info = launch_hpc._git_info(Path(__file__).resolve().parents[1])
        self.assertRegex(info["commit"], r"^[0-9a-f]{40}$")
        self.assertIn(info["dirty"], {"yes", "no"})


class TestRuntimeRefRepositories(unittest.TestCase):
    """The map the Beaker launcher resolves runtime SHAs against (#77).

    Read from source with `ast` rather than imported: beaker_client pulls in
    requests and beaker-py at module scope, and neither is worth installing in
    CI to assert one dict literal. Parsing also checks the declaration itself
    rather than whatever a runtime patch might have left behind.
    """

    def test_names_the_post_rename_repos(self):
        import ast

        source = (
            Path(__file__).resolve().parents[1] / "code" / "beaker_client.py"
        ).read_text()
        found = None
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "RUNTIME_REF_REPOSITORIES"
                for t in node.targets
            ):
                found = ast.literal_eval(node.value)
                break

        self.assertIsNotNone(found, "RUNTIME_REF_REPOSITORIES not found in beaker_client.py")
        self.assertEqual(
            found,
            {
                "WRAPPER_REF": "aind-dynamic-foraging-bfm-wrapper",
                "DISPATCHER_REF": "aind-dynamic-foraging-bfm-dispatcher",
                "FORAGING_MODELS_REF": "aind-dynamic-foraging-models",
            },
        )


class TestHeldoutOverrideInjection(unittest.TestCase):
    """The held-out override is appended only for models whose config has a `training` block.

    Getting this wrong fails in one of two ways: a cognitive baseline (no `model.training`) dies
    in Hydra composition, or a neural model silently loses every `heldout/*` metric.
    """

    OVERRIDE = "auto_heldout_finetune.output_root"

    def _config_dir(self, models: dict):
        """A temp `code/` tree holding the given model configs; returns a launch_hpc `__file__`."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        model_dir = Path(temp.name) / "config" / "model"
        model_dir.mkdir(parents=True)
        for name, text in models.items():
            (model_dir / f"{name}.yaml").write_text(text)
        return str(Path(temp.name) / "launch_hpc.py")

    def _injected(self, models: dict, command: list) -> bool:
        with mock.patch.object(launch_hpc, "__file__", self._config_dir(models)):
            cfg = launch_hpc._inject_lineage_into_command({"command": list(command)}, {})
        return any(self.OVERRIDE in str(c) for c in cfg["command"])

    def test_model_with_training_receives_the_override(self):
        self.assertTrue(self._injected({"gru": "type: gru\ntraining:\n  lr: 0.001\n"}, ["python", "model=gru"]))

    def test_model_without_training_does_not(self):
        self.assertFalse(self._injected({"hb": "type: hb\nnum_chains: 4\n"}, ["python", "model=hb"]))

    def test_an_explicit_override_in_the_sweep_is_not_duplicated(self):
        cmd = ["python", "model=gru", f"model.training.{self.OVERRIDE}=/x"]
        with mock.patch.object(launch_hpc, "__file__", self._config_dir({"gru": "training: {}\n"})):
            out = launch_hpc._inject_lineage_into_command({"command": cmd}, {})["command"]
        self.assertEqual(sum(self.OVERRIDE in str(c) for c in out), 1)

    def test_unknown_or_unselected_model_keeps_the_old_behaviour(self):
        self.assertTrue(self._injected({}, ["python", "model=not_a_config"]))
        self.assertTrue(self._injected({"gru": "training: {}\n"}, ["python"]))

    def test_malformed_or_non_mapping_config_keeps_the_old_behaviour(self):
        self.assertTrue(self._injected({"bad": "training: [unclosed\n"}, ["python", "model=bad"]))
        self.assertTrue(self._injected({"listy": "- a\n- b\n"}, ["python", "model=listy"]))

    def test_unreadable_config_keeps_the_old_behaviour(self):
        models = {"hb": "num_chains: 4\n"}
        with mock.patch.object(launch_hpc, "__file__", self._config_dir(models)), \
             mock.patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            self.assertTrue(launch_hpc._model_has_training_block(["python", "model=hb"]))


if __name__ == "__main__":
    unittest.main()
