# aind-dynamic-foraging-bfm-dispatcher

**The control plane of the `dynamic-foraging-bfm` stack** — a behavioural foundation model
for rodent dynamic foraging.

This repo does not train anything. It composes a Hydra config into a job spec and submits it
to one of three backends: **Code Ocean**, **Beaker (AI Hub)**, or the **Allen on-prem SLURM
cluster**. The training itself lives in the sibling repo
[`aind-dynamic-foraging-bfm-wrapper`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper).

Splitting it this way means one set of configs, four model families and three compute
backends stay in step: to add a model variant you add a config file, not a code path.

## The stack at a glance

[![Behavioral foundation model stack](docs/diagrams/bfm-stack.png)](https://raw.githack.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/main/docs/diagrams/bfm-stack.html)

The still is a snapshot; **[open the interactive version](https://raw.githack.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/main/docs/diagrams/bfm-stack.html)**
to pan, zoom, search and trace a single relationship. Sources and further diagrams:
[`docs/diagrams/`](docs/diagrams/).

All four model families — GRU, disRNN, baseline RL and the hierarchical Bayes baseline —
reach the same `ModelTrainer.fit` through a Hydra `_target_`, and report the same held-out
likelihood (`heldout/eval_likelihood`). That shared axis is what makes them comparable.

## Your first ten minutes

The two repos must sit side by side — the wrapper resolves this repo's `code/config/` as a
sibling directory at import time:

```text
/path/to/parent/
  aind-dynamic-foraging-bfm-dispatcher/     # you are here
  aind-dynamic-foraging-bfm-wrapper/
```

You can compose and inspect a config without running a job, or having the wrapper installed:

```bash
python code/launch_CO_wrapper.py data=mice model=disrnn --cfg job
```

That prints the fully resolved config — 116 lines for this one — after defaults, includes and
CLI overrides have been applied. It starts:

```yaml
data:
  type: mice
  _target_: data_loaders.mice.MiceDatasetLoader
  subject_ids:
  - 774212
  multisubject: false
  ...
model:
  type: disrnn
  _target_: model_trainers.disrnn_trainer.DisrnnTrainer
  architecture:
    latent_size: 5
    ...
```

The `_target_` fields are the important part: they name the wrapper classes Hydra will
instantiate. Drop `--cfg job` and the same config is written to `.hydra/config.json` in the
run directory for the wrapper to pick up.

> Config values are **not** reproduced in this README on purpose. They change, and a pasted
> copy goes stale silently — run the command above for the current truth.

## What you can vary

Configs live in [`code/config/`](code/config/), composed by Hydra. `config.yaml` defaults to
`data=mice model=disrnn`.

| Group | Options | Pick one with |
|---|---|---|
| `data` | `mice`, `mice_snapshot`, `mice_snapshot_scaling`, `mice_snapshot_smallpool`, `synthetic`, `synthetic_hierarchical` (+ shared `base`) | `data=mice_snapshot` |
| `data/agent` | `rl`, `rnn`, `hierarchical_rl`, and five `hierarchical_rl_stage*` variants | `data/agent=hierarchical_rl` |
| `data/task` | `random_walk`, `coupled_block`, `uncoupled_block` | `data/task=uncoupled_block` |
| `model` | GRU: `gru`, `gru_scaling` · disRNN: `disrnn` · baseline RL: `baseline_rl`, `baseline_rl_bari`, `baseline_rl_ctt`, `baseline_rl_hattori`, `baseline_rl_rw`, `baseline_rl_losscounting` · hierarchical Bayes: `hb_hattori` | `model=hb_hattori` |

`data/agent` and `data/task` are nested config **groups**, so they take a slash, not a dot:

```bash
# synthetic data from an RL agent in an uncoupled-block task, fit by a baseline RL model
python code/launch_CO_wrapper.py \
    data=synthetic data/agent=rl data/task=uncoupled_block model=baseline_rl --cfg job
```

Any leaf value is overridable with a dot path, and `-m` sweeps the Cartesian product:

```bash
python code/launch_CO_wrapper.py -m \
    model.penalties.beta=0.0001,0.001,0.01 model.training.lr=0.0001,0.001
```

Outputs land under `$BFM_OUTPUT_DIR` (default `/results`), one directory per Hydra job id,
each with its composed config at `.hydra/config.json`.

## Three ways to run it

The same composed config drives all three. Only the launcher differs.

| Route | Launcher | Reach for it when | Detail |
|---|---|---|---|
| **Beaker / AI Hub** | `code/launch_beaker_resumable.py` (preferred), `code/launch_beaker.py` | GPU training at scale; this is where most training runs today | [`code/beaker/README.md`](code/beaker/README.md) |
| **Allen SLURM** | `code/launch_hpc.py` | CPU jobs, on-prem data, long queues | [`code/hpc/README.md`](code/hpc/README.md) |
| **Code Ocean** | `code/launch_CO_wrapper.py`; the capsule's own `code/run` | Interactive/registered capsule runs | `.codeocean/app-panel.json` |

Beaker and SLURM both go through a W&B sweep — the launcher creates the sweep, stamps
provenance into it, and submits agents. Code Ocean fans out Hydra job directories instead.

**Before any launch larger than ~4 GPUs or ~4 concurrent tasks**, check what is actually
schedulable — "free" is not "schedulable", because Beaker counts GPUs on cordoned nodes and
`sinfo` counts drained ones:

```bash
python code/check_gpu_availability.py --beaker --hpc
```

Beaker submissions go **only** to hub clusters, with no exceptions; a job sent elsewhere
silently never schedules rather than failing. The current cluster list lives in
[`code/beaker/README.md`](code/beaker/README.md), which is also where GPU sizing, priority
tiers, the image table and the resumable-run mechanics are kept.

### In Code Ocean

The capsule is the **Beaker control plane**: a Reproducible Run creates a W&B sweep from
`code/beaker/sweep_mvp.yaml`, saves a reproducibility record to `/results`, renders the sweep
id into `code/beaker/experiment_mvp.yaml` and submits it. Its single app-panel parameter is
**"Launcher args"**, passed to `code/launch_beaker.py` — leave it blank for the default, or
give it flags such as `--no-submit`, `--sweep`, `--experiment`, `--workspace`.

The historical Code Ocean compute path — composing Hydra configs and fanning out job
directories — is `code/launch_CO_wrapper.py`, which is what the examples above call.

## Studies

Scientific work is organised one folder per question under [`studies/`](studies/), named
`NN-{model}-{purpose}`:

```text
studies/01-gru-scaling-law/          # a question, one W&B project
  variants/<variant>/                # one W&B group each
    sweep.yaml  experiment.yaml      # what was launched
    launch_record/                   # SHA-pinned specs + provenance, written by the launcher
  analysis/                          # producers, curated JSON/CSV, figures, reports
  Makefile                           # regenerates every artifact
```

Ten studies exist today, covering GRU and disRNN scaling laws, a disRNN β scan, embedding
recovery, operating points at scale, timing inputs, cross-species transfer, reservoir
ablation, and the HB-vs-GRU held-out comparison.
Conventions — naming, group scheme, the `meta.*` provenance block, wrap-up — are in the
`study-conventions` skill.

## Where to read next

| For | Read |
|---|---|
| Vocabulary — subject vs session, held-out, conditioning, zero/few-shot | [`CONTEXT.md`](CONTEXT.md) |
| Working rules for humans and agents in this repo | [`AGENTS.md`](AGENTS.md) |
| Launching, studies, reporting, provenance | [`aind-behavior-foundation-model-skills/`](aind-behavior-foundation-model-skills/) |
| Why a modelling decision was made the way it was | [`docs/adr/`](docs/adr/), [`docs/design-*.md`](docs/) |
| Architecture diagrams and how to regenerate them | [`docs/diagrams/`](docs/diagrams/) |
| Interpreting a run's logs and metrics; the training code | the wrapper's `code/TRAINING.md` |

## Tests

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -t . -v
```

Pure-function coverage of the launchers only — sweep-array parsing, `user.env` loading,
provenance stamping, Beaker spec rendering and grid expansion. Nothing here talks to SLURM,
Beaker or W&B. This is also what CI runs. `code/` is deliberately not packaged as a
distribution yet; see [`docs/repo-split-plan.md`](docs/repo-split-plan.md).

## License

MIT — see [LICENSE](LICENSE). Contributor expectations: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
