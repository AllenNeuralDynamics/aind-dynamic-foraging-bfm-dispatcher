# Global AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

The four-principle backbone (Think Before Coding, Simplicity First, Surgical Changes, Goal-Driven Execution) is adapted from [multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills), which distills [Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876) on LLM coding pitfalls. HPC-specific rules and the commit-message convention below are local additions.

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 0. Load the Right Skill First

Operational detail lives in the skills pack (`aind-behavior-foundation-model-skills/`), not
in this file. **Load the relevant skill before you act, not after something breaks:**

- Any repo work → **codebase-map** and **git-session-isolation**
- Any Beaker job → **beaker-launch**
- Any HPC job → **hpc-launch**
- Any training config, run interpretation, W&B metrics, or post-training analysis → **wrapper-runtime**
- Any study or variant work → **study-conventions**
- Any analysis or figure work → **posthoc-reporting**
- Filing an issue, updating its status, or touching the project board → **issue-tracking**

**This file states rules; it does not enumerate the data a rule ranges over.** "Submit only
to hub clusters" belongs here and cannot go stale; the *list of clusters* belongs to
`beaker-launch`, because a list can. So when a rule below needs a name, a path, a number, or
a table to act on, that detail is in the named skill and this file points at it.

If a skill and this file ever disagree, that is a **bug, not a precedence question**: follow
whichever side is more restrictive and fix both in the same PR.

## 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them instead of picking silently.
- If a simpler approach exists, say so.
- If something is unclear, stop and ask.

## 2. Simplicity First

Write the minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No configurability that was not requested.
- No error handling for impossible scenarios.
- If the solution is overcomplicated, simplify it.

## 3. Surgical Changes

Touch only what is needed. Clean up only your own mess.

When editing existing code:
- Do not improve adjacent code, comments, or formatting unless required.
- Do not refactor unrelated code.
- Match existing style.
- If you find unrelated dead code, mention it but do not delete it.

When your changes create orphans:
- Remove imports, variables, or functions made unused by your change.
- Do not remove pre-existing dead code unless asked.

Test: Every changed line should trace directly to the request.

## 4. Goal-Driven Execution

Define success criteria and verify.

Transform tasks into verifiable goals:
- Add validation -> write failing tests for invalid inputs, then make them pass.
- Fix a bug -> write a reproducing test, then make it pass.
- Refactor -> ensure tests pass before and after.

For multi-step tasks, use a brief plan:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

These guidelines are working when diffs contain fewer unnecessary changes, solutions are simpler, and clarifications happen before implementation.

## 5. HPC Execution Safety

**Never run workload Python on the HPC login node.** This covers training, sweeps, data
generation, extraction, and analysis that imports loader/model code, accesses cluster-only
data, or starts workers. Those workloads go through `srun`/`sbatch`/`salloc` onto a compute
node.

Two narrow exceptions may run on the agent's local machine (not the HPC login node):

- A *submit-only launcher* that creates a sweep, submits, or probes capacity and returns —
  never importing loader/model/training code or starting a `multiprocessing` Pool.
- A pure offline report formatter that reads only committed frozen JSON/CSV artifacts and
  writes figures or report markers. It must not access W&B, raw data, `/allen`, loaders,
  models, or multiprocessing. These plotting steps stay local and must not require HPC.

Which launchers qualify, and the real incident behind this rule (an unguarded spawn-Pool
that forked into a 260 MB / 65k-error cascade): **hpc-launch**. Formatter/extractor boundaries
are defined by **posthoc-reporting**.

## 6. Semantic Commit Messages

[Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<optional scope>): <short imperative summary>

<optional body explaining what and why, wrapped at ~72 chars>

<optional footer, e.g. "Refs #123" or "BREAKING CHANGE: ...">
```

`<type>` ∈ `feat` `fix` `docs` `refactor` `perf` `test` `build` `ci` `chore` `revert`.

- Summary imperative, no trailing period, <= 72 chars; `<scope>` names the affected area.
- One logical change per commit; split unrelated changes.
- Breaking changes: `!` after the type/scope plus a `BREAKING CHANGE:` footer.
- The body explains motivation and non-obvious consequences — don't restate the diff.

## 7. Human-Facing Logs & Reporting

Anything a human reads (status updates, run reports, PR/commit notes) uses **Seattle time**
(`TZ=America/Los_Angeles`, stamped like `10:48 PT`), not UTC, and **links the W&B
project/run** so the reader can click through. Beaker experiments aren't W&B sweeps — link
the W&B run, plus the Beaker experiment id when relevant.

## 8. Study & Experiment Organization

One folder per study under `studies/<name>/`; variants as self-contained subfolders; one W&B
project per study, one group per variant. A study answers one scientific question — variants
are not separate studies. Layout, naming, group scheme, provenance `meta.*`, interventions:
**study-conventions**.

## 9. Merging Pull Requests

**Never squash-merge a PR.** Merge with a merge commit (`gh pr merge <n> --merge`) so the
branch's individual commits and their provenance survive on the target branch.

## 10. Beaker / AI Hub Launch & Scheduling

- **Submit ONLY to `hub` clusters** (`octo-hub-*`, `octo.hub-*`, `aihub-*`) — never to a
  non-hub cluster even if it shows idle GPUs, and there are **no exceptions**. A job sent to
  one silently never schedules rather than failing.
- **Check schedulable capacity before any large launch** (> 4 GPUs / > 4 concurrent tasks)
  and route to the backend with room. "Free" is not "schedulable": Beaker reports GPUs on
  cordoned nodes as free and `sinfo` counts `drain`/`down` nodes, so raw counts lie.
- **Scientific jobs submit immutable source refs.** Templates may carry readable branch
  values, but the launchers resolve every runtime ref to a full SHA before submitting. Pin
  manually when bypassing them.
- Heavy work never on the login node (§5).
- Which clusters are usable and why (memory limits, the S3 reachability rule), the capacity
  command, priority/preemption tiers and their slot budgets, GPU-bundle sizing, resumable
  launches, extend/re-score: **beaker-launch**, before any non-trivial launch.

## 11. Verify Mechanisms With Data Before Asserting

When explaining *why* infra/scheduling/quota behaves a certain way, pull the data first and
cite the field. Label observed fact ("verified: …") vs inference ("likely, unconfirmed: …");
don't present a hypothesis as a conclusion; isolate variables before attributing cause.
Worked examples: **beaker-launch**, `references/scheduling-lessons.md`.

## 12. Post-hoc Analysis & Reporting

Reports are code: committed, regenerable, one producer per artifact, provenance inside the
artifact. Numbers are frozen in a committed file and keyed by an immutable digest — never
read live from W&B on the reproducible path. File contracts, marker syntax, `_meta` schema,
Makefile and `.gitignore` policy, multi-agent rules: **posthoc-reporting**; mirror a
normalized study (e.g. `studies/01-gru-scaling-law/`).

## 13. Claude Science Workflow

The agent runs on the user's Mac (persistent brain); GitHub `origin` is the source of truth,
and every local checkout — Mac authoring clone, HPC pull-only runtime — is a disposable
cache of it. Load balancing: CPU jobs → HPC SLURM, GPU jobs → Beaker; both launchers live
here and only submit, so one repo drives both.

- Checkout paths, task-to-host table, W&B-from-sandbox access, credentials: **codebase-map**,
  `references/claude-science-workflow.md`.
- Committing and pushing from the sandbox, which git operations the sandbox blocks, and
  pairing the two repo SHAs on a deliverable: **git-session-isolation**.
- Launching from the sandbox (import quirk, dry-run, network grants), **verifying the image
  name before submitting** — a stale image fails with no logs — and telling a transient node
  failure from a code bug: **beaker-launch**, `references/sandbox-launch.md`. The live image
  table is `code/beaker/README.md`.
- The project is named `dynamic-foraging-bfm`. A `disrnn` token is not automatically the project name:
  in a code symbol it names the disRNN *architecture*, and under `studies/*/launch_record/`
  or `studies/*/analysis/` it names what actually ran. **Never global find-replace `disrnn`**
  — doing so renames a model family and falsifies the run record. Which tokens rename and
  which are frozen: `docs/adr/0007-project-identity-rename-boundary.md`.

## Agent skills

_Generated by `/setup-matt-pocock-skills`; read by `to-spec`, `to-tickets`, `wayfinder`,
`implement`, and `domain-modeling`. Edit `docs/agents/*.md` directly to adjust._

### Issue tracker

GitHub Issues on `AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher`, via the `gh` CLI.
See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: one `CONTEXT.md` at the repo root plus `docs/adr/`, both created lazily.
See `docs/agents/domain.md`.
