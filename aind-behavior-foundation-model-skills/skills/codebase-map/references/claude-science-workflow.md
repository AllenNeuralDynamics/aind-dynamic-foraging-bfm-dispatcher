# Claude Science workflow (Mac orchestration)

How this project is driven from **Claude Science** — the agent running on the
user's Mac, 24/7. (Absorbed from the former `docs/claude-science-workflow.md`;
this file is now the canonical home. One-line rule: `AGENTS.md` §13.)

## Roles

- **Mac (Claude Science)** — the orchestration brain. Holds persistent project
  memory, artifacts, and planning; drives everything else. Runs continuously, so
  it is the durable state, not any remote shell session.
- **GitHub** (`AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher`) — the single source of
  truth for code. Both checkouts below track it.
- **HPC login node** (`hpc-code.corp.alleninstitute.org`) — a **pull-only runtime
  checkout** at `/home/han.hou/code/aind-dynamic-foraging-bfm-dispatcher`. Synced with
  `git pull`, never edited in place. Reached over SSH (VPN when remote).
- **Mac clone** (`~/Scripts/aind-dynamic-foraging-bfm-dispatcher`) — the **authoring checkout**,
  sibling to `aind-dynamic-foraging-bfm-wrapper` and `aind-dynamic-foraging-bfm-studies` (launches
  run from the studies root). Where the agent edits source and shows diffs.

## Where each task runs

| Task | Runs on | Needs VPN? |
| --- | --- | --- |
| GPU training → Beaker | Mac (beaker CLI/API + token) | No |
| CPU training → HPC SLURM | HPC login node over SSH (`git pull` to sync first) | Yes, only here |
| Rebuild the `dynamic-foraging-bfm-wrapper` image | Mac (Docker; HPC has none) | No |
| W&B post-hoc analysis | Mac sandbox (GraphQL) or HPC (SDK) | No |
| Edit / review repo code | Mac clone, native diffs | No |

Rationale for the CPU→HPC / GPU→Beaker split: first-order load balancing — HPC
has many CPU nodes, Beaker has the better GPUs. Both launchers (`launch_hpc.py`,
`launch_beaker*.py`) live in this one repo and both only *submit* (training runs
remotely), so a single checkout drives both. All results upload to W&B with
provenance — the basis for post-hoc analysis (posthoc-reporting skill).

## Division of labor: agent authors, user owns git

The Claude Science sandbox **cannot create a `.git` directory** inside a granted
host path (`mkdir <granted>/.git` → "Operation not permitted"). Consequences:

- **Cloning** a repo into `~/Scripts` must be done by the user in their own
  terminal (their local git also carries the SAML-SSO authorization for the org).
- **Editing tracked files, committing, and pushing** into an *existing* checkout
  all work through the host grant — the block is only on creating `.git`.

So the agent edits files and shows `git diff`; commit/push can be done by either
party. The user always owns the initial clone and the SSO auth.

## Sandbox recipes (canonical homes)

- **Mac → Beaker launch** (PYTHONPATH quirk, image verification, transient node
  failures): beaker-launch skill, `references/sandbox-launch.md`.
- **Reading W&B from the sandbox** (`wandb.Api()` fails; GraphQL route):
  posthoc-reporting skill, `references/wandb-graphql-sandbox.md`.
- **Pre-launch capacity check**: `python code/check_gpu_availability.py --beaker`
  (Beaker, no VPN) / `--hpc` (needs Allen network).

## Credentials (stored in Claude Science, never in the repo)

- `WANDB_API_KEY` — W&B GraphQL/API access. Entity `AIND-disRNN`, login `houhan`.
- `BEAKER_TOKEN` — Beaker CLI/API (`user_token` from `~/.beaker/config.yml`).
- `GITHUB_TOKEN` — SSO-authorized PAT for the org; used in-memory for push, never
  persisted into git config.
