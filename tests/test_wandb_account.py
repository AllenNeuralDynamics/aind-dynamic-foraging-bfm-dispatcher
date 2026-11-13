"""Per-study W&B account (dispatcher #225): the sweep must match the README, the key must belong
to the entity, and the key must never be printed. No network: access is checked with a fake."""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import wandb_account as wa  # noqa: E402

FAKE_KEY = "wandb_v1_FAKEKEYFORTESTS"


def make_study(root: Path, entity: str, sweep: str, project="dfbfm_12-x") -> Path:
    d = root / "studies" / "12-x"
    (d / "variants" / "base").mkdir(parents=True)
    (d / "README.md").write_text(f"---\nid: 12-x\nwandb:\n  entity: {entity}\n  project: {project}\n---\n# x\n")
    p = d / "variants" / "base" / "sweep.yaml"
    p.write_text(sweep)
    return p


HPC_ND = "entity: NeuralDynamics\nproject: dfbfm_12-x\ncommand:\n- python\n- wandb.entity=NeuralDynamics\n"
HPC_OLD = "entity: AIND-disRNN\nproject: mice\ncommand:\n- python\n"


class WandbAccountTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.keys = self.root / "keys"
        self.keys.mkdir()
        env = mock.patch.dict(os.environ, {wa.KEY_DIR_ENV: str(self.keys)})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("WANDB_API_KEY", None)

    def tearDown(self):
        os.environ.pop("WANDB_API_KEY", None)
        self.tmp.cleanup()

    def test_matching_sweep_passes(self):
        p = make_study(self.root, "NeuralDynamics", HPC_ND)
        self.assertEqual(wa.check_wandb_account(p, "hpc"), "NeuralDynamics")
        self.assertEqual(wa.check_wandb_account(p, "beaker"), "NeuralDynamics")

    def test_sweep_copied_from_an_old_study_is_refused(self):
        p = make_study(self.root, "NeuralDynamics", HPC_OLD)
        with self.assertRaisesRegex(wa.WandbAccountError, "AIND-disRNN"):
            wa.check_wandb_account(p, "hpc")

    def test_override_must_match_too(self):
        p = make_study(self.root, "NeuralDynamics", HPC_ND.replace("wandb.entity=NeuralDynamics", "wandb.entity=AIND-disRNN"))
        with self.assertRaises(wa.WandbAccountError):
            wa.check_wandb_account(p, "hpc")

    def test_hpc_needs_top_level_and_beaker_needs_override(self):
        no_top = make_study(self.root, "NeuralDynamics", "command:\n- wandb.entity=NeuralDynamics\n")
        with self.assertRaisesRegex(wa.WandbAccountError, "top-level"):
            wa.check_wandb_account(no_top, "hpc")
        wa.check_wandb_account(no_top, "beaker")
        no_override = make_study(self.root / "b", "NeuralDynamics", "entity: NeuralDynamics\ncommand:\n- python\n")
        wa.check_wandb_account(no_override, "hpc")
        with self.assertRaisesRegex(wa.WandbAccountError, "override"):
            wa.check_wandb_account(no_override, "beaker")

    def test_old_studies_and_off_layout_sweeps_are_untouched(self):
        p = make_study(self.root, "AIND-disRNN", HPC_OLD)
        self.assertEqual(wa.check_wandb_account(p, "hpc"), "AIND-disRNN")
        off = self.root / "sweep.yaml"
        off.write_text(HPC_ND)
        self.assertIsNone(wa.check_wandb_account(off, "hpc"))

    def test_key_file_is_exported_and_never_printed(self):
        (self.keys / "api_key.NeuralDynamics").write_text(FAKE_KEY + "\n")
        p = make_study(self.root, "NeuralDynamics", HPC_ND)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            wa.prepare(p, "hpc", teams_fn=lambda: ("han-hou", ["NeuralDynamics"]))
        self.assertEqual(os.environ["WANDB_API_KEY"], FAKE_KEY)
        self.assertNotIn(FAKE_KEY, out.getvalue())
        self.assertIn("NeuralDynamics", out.getvalue())

    def test_no_key_file_leaves_the_environment_alone(self):
        p = make_study(self.root, "NeuralDynamics", HPC_ND)
        with contextlib.redirect_stdout(io.StringIO()):
            wa.prepare(p, "hpc", teams_fn=lambda: ("han-hou", ["NeuralDynamics"]))
        self.assertNotIn("WANDB_API_KEY", os.environ)

    def test_key_without_access_is_refused_and_names_the_file(self):
        p = make_study(self.root, "NeuralDynamics", HPC_ND)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(wa.WandbAccountError) as cm:
            wa.prepare(p, "hpc", teams_fn=lambda: ("houhan", ["AIND-disRNN"]))
        self.assertIn("api_key.NeuralDynamics", str(cm.exception))

    def test_default_entity_is_not_checked_and_lookup_failure_only_warns(self):
        def boom():
            raise OSError("no network")
        wa.check_entity_access("AIND-disRNN", teams_fn=boom)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            wa.check_entity_access("NeuralDynamics", teams_fn=boom)
        self.assertIn("WARNING", out.getvalue())

    def test_beaker_secret_must_carry_the_entity(self):
        p = make_study(self.root, "NeuralDynamics", HPC_ND)
        spec = lambda secret: {"tasks": [{"envVars": [{"name": "WANDB_API_KEY", "secret": secret}]}]}
        with contextlib.redirect_stdout(io.StringIO()):
            wa.prepare(p, "beaker", spec=spec("han-wandb-api-key-neuraldynamics"))
            with self.assertRaisesRegex(wa.WandbAccountError, "han-wandb-api-key'|-wandb-api-key-neuraldynamics"):
                wa.prepare(p, "beaker", spec=spec("han-wandb-api-key"))
        wa.check_beaker_secret(spec("han-wandb-api-key"), "AIND-disRNN")

    def test_beaker_warns_about_the_secret(self):
        p = make_study(self.root, "NeuralDynamics", HPC_ND)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            wa.prepare(p, "beaker")
        self.assertIn("secret", out.getvalue())


if __name__ == "__main__":
    unittest.main()
