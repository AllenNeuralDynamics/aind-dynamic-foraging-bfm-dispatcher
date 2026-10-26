"""Spec-rendering tests for the Beaker launchers.

These cover what the rename actually changed on the launch path -- the
``BFM_META_*`` provenance block, ``BFM_RESUMABLE_OUTPUT_DIR``, grid expansion
and study/variant derivation -- without submitting anything.

Two things are deliberately avoided:

* **No real launch.** A PR gate must not spend GPU budget or depend on Beaker
  being reachable.
* **No ``--no-submit``.** ``launch_beaker.py --no-submit`` calls
  ``create_sweep()`` first and leaves a real W&B sweep behind, so it is not
  side-effect-free. The functions below are the seams to test instead:
  ``_render_experiment`` documents itself as pure, and ``build_spec`` does its
  work before ``main()`` calls ``pin_runtime_refs``, which is the only network
  step.

``beaker_client`` imports ``requests`` and ``beaker-py`` at module scope, so
both are stubbed here rather than installed -- CI needs PyYAML alone.
"""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

# Stub the heavy transitive imports before the launchers pull them in.
for _name in ("requests", "beaker"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
if not hasattr(sys.modules["beaker"], "Beaker"):
    sys.modules["beaker"].Beaker = object
    sys.modules["beaker"].Config = object

import launch_beaker  # noqa: E402
import launch_beaker_resumable as resumable  # noqa: E402

SWEEP = """\
entity: AIND-disRNN
project: ai_hub_test
name: unit-test-sweep
method: grid
command:
  - python
  - -m
  - run_hpc
  - data=synthetic
  - ${args_no_hyphens}
parameters:
  seed:
    values: [1, 2]
  model.architecture.latent_size:
    values: [4, 8]
"""

EXPERIMENT = """\
version: v2
description: unit-test
tasks:
  - name: t
    image:
      beaker: han-hou/dynamic-foraging-bfm-wrapper-main-20260902
    command:
      - bash
      - /workspace/aind-dynamic-foraging-bfm-wrapper/beaker/entrypoint.sh
      - wandb
      - agent
      - "<SWEEP_ID>"
    context:
      priority: low
      preemptible: true
    constraints:
      cluster:
        - ai1/octo-hub-aws-l40s
    resources:
      gpuCount: 1
    envVars:
      - name: BFM_OUTPUT_DIR
        value: /results
      - name: WRAPPER_REF
        value: main
    result:
      path: /results
"""


def _write(tmp: Path, name: str, body: str) -> str:
    p = tmp / name
    p.write_text(body)
    return str(p)


class TestRenderExperiment(unittest.TestCase):
    """launch_beaker._render_experiment -- the non-resumable route."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="beaker_render_"))
        self.exp = _write(self.tmp, "experiment.yaml", EXPERIMENT)
        self.meta = [
            {"name": "BFM_META_STUDY", "value": "01-gru-scaling-law"},
            {"name": "BFM_META_VARIANT", "value": "nxd-grid"},
        ]

    def test_substitutes_the_sweep_id(self):
        spec = launch_beaker._render_experiment(self.exp, "ent/proj/abc123", "g@1", self.meta)
        self.assertIn("ent/proj/abc123", spec["tasks"][0]["command"])
        self.assertNotIn("<SWEEP_ID>", spec["tasks"][0]["command"])

    def test_injects_group_and_meta(self):
        spec = launch_beaker._render_experiment(self.exp, "e/p/s", "nxd-grid@20260902", self.meta)
        env = {e["name"]: e["value"] for e in spec["tasks"][0]["envVars"]}
        self.assertEqual(env["WANDB_RUN_GROUP"], "nxd-grid@20260902")
        self.assertEqual(env["BFM_META_STUDY"], "01-gru-scaling-law")
        self.assertEqual(env["BFM_META_VARIANT"], "nxd-grid")

    def test_preserves_unmanaged_env(self):
        spec = launch_beaker._render_experiment(self.exp, "e/p/s", "g@1", self.meta)
        env = {e["name"]: e["value"] for e in spec["tasks"][0]["envVars"]}
        self.assertEqual(env["BFM_OUTPUT_DIR"], "/results")
        self.assertEqual(env["WRAPPER_REF"], "main")

    def test_replaces_rather_than_duplicates_managed_keys(self):
        # A template that already carries a managed key must not end up with the
        # value twice -- the wrapper would read whichever came first.
        doubled = EXPERIMENT.replace(
            "      - name: BFM_OUTPUT_DIR",
            "      - name: BFM_META_STUDY\n        value: stale\n      - name: BFM_OUTPUT_DIR",
        )
        path = _write(self.tmp, "doubled.yaml", doubled)
        spec = launch_beaker._render_experiment(path, "e/p/s", "g@1", self.meta)
        names = [e["name"] for e in spec["tasks"][0]["envVars"]]
        self.assertEqual(names.count("BFM_META_STUDY"), 1)
        env = {e["name"]: e["value"] for e in spec["tasks"][0]["envVars"]}
        self.assertEqual(env["BFM_META_STUDY"], "01-gru-scaling-law")

    def test_emits_no_legacy_prefix(self):
        spec = launch_beaker._render_experiment(self.exp, "e/p/s", "g@1", self.meta)
        names = [e["name"] for e in spec["tasks"][0]["envVars"]]
        self.assertFalse([n for n in names if n.startswith("DISRNN_")])


class TestGridExpansion(unittest.TestCase):
    def test_cartesian_product(self):
        points = resumable._grid_points({"a": {"values": [1, 2]}, "b": {"values": ["x", "y"]}})
        self.assertEqual(len(points), 4)
        self.assertIn({"a": 1, "b": "x"}, points)
        self.assertIn({"a": 2, "b": "y"}, points)

    def test_single_parameter(self):
        self.assertEqual(len(resumable._grid_points({"seed": {"values": [1, 2, 3]}})), 3)

    def test_empty_parameters(self):
        # Guards the "sweep has no grid points" exit in build_spec.
        self.assertEqual(resumable._grid_points({}), [{}])


class TestStudyVariant(unittest.TestCase):
    def test_derives_from_the_studies_layout(self):
        p = Path("/repo/studies/01-gru-scaling-law/variants/nxd-grid/sweep.yaml")
        self.assertEqual(resumable._study_variant(p), ("01-gru-scaling-law", "nxd-grid"))

    def test_falls_back_to_adhoc_off_layout(self):
        study, variant = resumable._study_variant(Path("/tmp/my_sweep.yaml"))
        self.assertEqual(study, "adhoc")
        self.assertEqual(variant, "my-sweep")


class TestBuildSpec(unittest.TestCase):
    """launch_beaker_resumable.build_spec -- the preferred route."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="beaker_build_"))
        self.sweep = _write(self.tmp, "sweep.yaml", SWEEP)
        self.exp = _write(self.tmp, "experiment.yaml", EXPERIMENT)

    def test_expands_the_grid_into_one_task_each(self):
        spec = resumable.build_spec(self.sweep, self.exp)
        self.assertEqual(len(spec["tasks"]), 4)  # 2 seeds x 2 latent sizes
        self.assertEqual(len({t["name"] for t in spec["tasks"]}), 4)

    def test_every_task_carries_the_bfm_provenance_block(self):
        spec = resumable.build_spec(self.sweep, self.exp, label="lbl", note="why")
        for task in spec["tasks"]:
            env = {e["name"]: e["value"] for e in task["envVars"]}
            self.assertEqual(env["BFM_META_STUDY"], "adhoc")
            self.assertEqual(env["BFM_META_VARIANT"], "sweep")
            self.assertEqual(env["BFM_META_LABEL"], "lbl")
            self.assertEqual(env["BFM_META_NOTE"], "why")
            self.assertRegex(env["BFM_META_LAUNCH_ID"], r"^\d{8}-\d{6}$")
            self.assertRegex(env["BFM_META_CONFIG_HASH"], r"^[0-9a-f]{8}$")

    def test_sets_the_resumable_output_dir(self):
        # Only the resumable route sets this; the HPC path never does, which is
        # why an HPC smoke cannot verify it.
        spec = resumable.build_spec(self.sweep, self.exp)
        for task in spec["tasks"]:
            env = {e["name"]: e["value"] for e in task["envVars"]}
            self.assertIn("BFM_RESUMABLE_OUTPUT_DIR", env)

    def test_no_legacy_prefix_anywhere_in_the_spec(self):
        # The end-state assertion for #81/#82: a rendered spec carries no
        # DISRNN_* environment variable at all.
        spec = resumable.build_spec(self.sweep, self.exp, label="l", note="n")
        for task in spec["tasks"]:
            for entry in task["envVars"]:
                self.assertFalse(
                    entry["name"].startswith("DISRNN_"),
                    f"legacy env var {entry['name']} in rendered spec",
                )

    def test_swept_overrides_reach_the_command(self):
        spec = resumable.build_spec(self.sweep, self.exp)
        joined = [" ".join(map(str, t["command"])) for t in spec["tasks"]]
        self.assertTrue(any("seed=1" in c for c in joined))
        self.assertTrue(any("model.architecture.latent_size=8" in c for c in joined))
        # The placeholder must be consumed, never passed through to run_hpc.
        self.assertFalse([c for c in joined if "args_no_hyphens" in c])

    def test_keeps_the_entrypoint_and_image_from_the_template(self):
        spec = resumable.build_spec(self.sweep, self.exp)
        for task in spec["tasks"]:
            self.assertEqual(task["command"][0], "bash")
            self.assertIn("entrypoint.sh", task["command"][1])
            self.assertEqual(
                task["image"]["beaker"], "han-hou/dynamic-foraging-bfm-wrapper-main-20260902"
            )



class TestBeakerToken(unittest.TestCase):
    """BEAKER_TOKEN first, then the CLI config file (dispatcher #196)."""

    def setUp(self):
        import beaker_client
        self.bc = beaker_client
        self.tmp = Path(tempfile.mkdtemp(prefix="beaker_token_"))
        self.cfg = self.tmp / "config.yml"

    def token(self, env):
        from unittest import mock
        with mock.patch.dict(os.environ, env, clear=True):
            return self.bc.beaker_token()

    def test_the_environment_wins(self):
        self.cfg.write_text("user_token: from-file\n")
        self.assertEqual(self.token({"BEAKER_TOKEN": "from-env", "BEAKER_CONFIG": str(self.cfg)}), "from-env")

    def test_the_config_file_is_the_fallback(self):
        self.cfg.write_text("agent_address: x\nuser_token: from-file\n")
        self.assertEqual(self.token({"BEAKER_CONFIG": str(self.cfg)}), "from-file")

    def test_the_default_path_is_under_home(self):
        (self.tmp / ".beaker").mkdir()
        (self.tmp / ".beaker" / "config.yml").write_text("user_token: home-file\n")
        self.assertEqual(self.token({"HOME": str(self.tmp)}), "home-file")

    def test_neither_names_where_it_looked(self):
        for text in (None, "", "agent_address: x\n", "- not a mapping\n"):
            if text is not None:
                self.cfg.write_text(text)
            with self.assertRaises(KeyError) as err:
                self.token({"BEAKER_CONFIG": str(self.cfg)})
            self.assertIn(str(self.cfg), str(err.exception))


if __name__ == "__main__":
    unittest.main()
