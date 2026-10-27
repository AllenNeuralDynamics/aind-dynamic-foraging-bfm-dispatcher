# aind-behavior-foundation-model-skills

Agent Skills pack for the AIND behavior foundation model stack. Each skill under
`skills/<name>/` follows the open [Agent Skills](https://agentskills.io/specification)
format, so the pack is consumable by any spec-compliant tool (Claude Code, Claude
Science, the Claude API/Agent SDK, ...).

## Skills

| Skill | Use for |
|---|---|
| `codebase-map` | Orienting in the four-repo stack (dispatcher, wrapper, studies, atlas) + the Claude Science workflow |
| `beaker-launch` | Launching jobs on Beaker (AI Hub): cluster allowlist, capacity checks, sizing, resumable runs |
| `hpc-launch` | Launching on Allen on-prem SLURM via `launch_hpc.py` / sbatch |
| `wrapper-runtime` | The training/analysis runtime: run lifecycle, held-out switches, checkpoints, `run_analysis.py` |
| `study-conventions` | Study/variant folder layout, W&B group naming, provenance, study wrap-up |
| `posthoc-reporting` | Report/JSON contracts, launch records, regeneration rules |
| `git-session-isolation` | Concurrent sessions on one shared repo + dual-repo provenance and the origin-is-truth git rules (Mac + HPC) |
| `issue-tracking` | Filing issues and putting them on the AIND-behavior-fm project board with Status/Priority/Size |

## Structure & authoring convention (progressive disclosure)

Each skill is a folder:

```
skills/<name>/
  SKILL.md          # lean, stable core: hard rules, decision guides, common commands
  references/*.md   # deep dives loaded on demand (procedures, recipes, verified lessons)
```

Authoring rule: **new hard rules and commands go in `SKILL.md`; new lessons and
procedures go in a `references/` file** with a one-line pointer from `SKILL.md`.
This keeps the always-loaded core small while nothing is lost.

## Single source of truth

- **This pack is canonical for cross-cutting operational knowledge** (launching,
  scheduling, study conventions, reporting, the Claude Science workflow). The
  former `docs/*.md` playbooks in the dispatcher were absorbed here and are now
  pointer stubs — update the skill, not the stub.
- **Code-adjacent living docs stay canonical for code-coupled reference** and the
  skills defer to them: the wrapper's `code/TRAINING.md` +
  `code/POST_TRAINING_ANALYSIS.md` (update-with-the-code contract), and
  `code/beaker/README.md` / `code/hpc/README.md` in the dispatcher.
- `AGENTS.md` (both repos) stays the always-loaded terse guardrail layer; it
  points into this pack for detail.
- **A skill/`AGENTS.md` disagreement is a bug, not a precedence question.** Don't write
  "X wins on conflict" between a skill and the always-loaded summary: `AGENTS.md` is terse
  and has no update contract, so a stale line there can outrank a corrected skill.
  Deferring to a *code-adjacent living doc* — the wrapper's `TRAINING.md`,
  `code/beaker/README.md` — is fine, because those are updated with the code they describe.
  Until a disagreement is resolved, follow whichever side is more restrictive, and fix both
  copies in one PR.
- **`codebase-map` is a map, not a rulebook.** It answers "where is X" and "which skill
  next" and deliberately does not restate rules owned by `AGENTS.md` or another skill.
  Since it is the first skill loaded, a stale copy there is the first thing read.

## Maintaining the pack

**This repo is the source of truth. The copies imported into agent catalogs are read-only
mirrors, and editing a mirror does not reach anyone.** In Claude Science, `host.skills.edit`
touches only the platform copy — the repo is untouched, so installed Claude Code agents
never see the change, and the next re-import silently reverts it.

So every skill edit lands here, by PR:

1. Branch off `origin/main`, edit under `skills/<name>/`, open a PR
   (never squash-merge — `gh pr merge <n> --merge`, per AGENTS.md §9).
2. Bump `version` in `.claude-plugin/plugin.json`. **This is not bookkeeping — skipping it
   fails silently.** `claude plugin update` compares versions, not content, so an edit
   shipped without a bump makes it report *"already at the latest version"* and copy
   nothing. The repo is correct, the installed copy agents actually read is stale, and
   nothing warns you. Verify with `diff -rq` between the repo pack and
   `~/.claude/plugins/cache/<marketplace>/<plugin>/<version>/skills`.
3. Re-import the plugin in Claude Code (`/plugin` — updates are pull-based, not automatic).
4. If the skill's `description` changed, note it in the PR: descriptions are the triggering
   mechanism, so a description change alters *when* the skill loads, not just its content.

Authoring rule (progressive disclosure): **new hard rules and commands go in `SKILL.md`;
new lessons, mechanisms, and evidence go in a `references/` file** with a one-line pointer
from `SKILL.md`. `SKILL.md` is loaded in full every time the skill triggers, so its length
is a running cost paid by every agent; `references/` is free until read.

## Import

**Claude Code (whole pack via the repo's plugin marketplace):**

```
/plugin marketplace add AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher
/plugin install aind-behavior-foundation-model-skills@aind-behavior-foundation-model
```

Inside this repo (and the wrapper repo) the pack is auto-enabled via the
checked-in `.claude/settings.json`.

**Other Agent Skills tools (e.g. Claude Science):** import the `skills/` directory —
each subfolder is a standard SKILL.md skill (with its `references/`).
