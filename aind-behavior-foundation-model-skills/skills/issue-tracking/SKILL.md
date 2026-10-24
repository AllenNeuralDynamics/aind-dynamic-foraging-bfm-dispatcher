---
name: issue-tracking
description: File GitHub issues for the AIND behavior foundation model repos and put them on the AIND-behavior-fm org project board (project 184) with Status, Priority, and Size set. Covers the always-file-an-issue-first rule, the cached board field/option IDs, the REST+GraphQL calls (no gh CLI in the sandbox), how to pick P0/P1/P2 and XS-XL, and linking the PR back. Use before starting any tracked work in aind-dynamic-foraging-bfm-dispatcher or aind-dynamic-foraging-bfm-wrapper, whenever opening an issue or PR, or when a board field needs updating.
---

# Issue & project-board tracking

**Every piece of tracked work starts with an issue; every issue is assigned to an owner and
goes on the board with Status, Priority, and Size set.** An issue nobody can find on the
board is invisible at planning time, an unassigned issue has no one it belongs to, and a PR
with no issue leaves the *why* only in a diff. File first, then work.

Default assignee is **`hanhou`** — assign at creation rather than later, so the issue is
never briefly ownerless. Override with `--assignee` when the work belongs to someone else.

Board: <https://github.com/orgs/AllenNeuralDynamics/projects/184/> ("AIND-behavior-fm").
Issues live on the repo the work touches: `aind-dynamic-foraging-bfm-studies` for a study,
variant, report or `studies/util` helper; `aind-dynamic-foraging-bfm-dispatcher` for launchers,
configs and the skills pack; `aind-dynamic-foraging-bfm-wrapper` for the training/analysis
payload. A cross-repo change gets one issue on the repo that owns the *decision*, with the
other repos' PRs cross-linked from it.

**Reference other repos' items in full.** A bare `#N` resolves against the repo it is written
in (and, in a chat, against the session's checkout), so `#7` written about the studies repo
opens dispatcher #7. On GitHub write `AllenNeuralDynamics/<repo>#N`; to a person, give the URL.

## The four steps

```bash
python scripts/board.py --repo aind-dynamic-foraging-bfm-dispatcher \
    --title "..." --body-file issue.md \
    --labels documentation priority:P1 \
    --status "In progress" --priority P1 --size M
```

That script does all four steps and prints the issue URL. Do them by hand only if you need
something it does not cover — the calls are below.

1. **Create the issue** (REST `POST /repos/{org}/{repo}/issues`) with `labels` and
   `assignees`. GitHub **silently drops** an assignee the repo cannot assign — it returns
   200 with an empty `assignees` array rather than erroring — so read the response back and
   say so if the login did not stick. The script does this check.
2. **Add it to the board** (GraphQL `addProjectV2ItemById`) — returns the *item* id, which
   is what the field mutations take. An issue node id is not an item id.
3. **Set Status, Priority, Size** (GraphQL `updateProjectV2ItemFieldValue`, once per field,
   `value: {singleSelectOptionId: ...}`).
4. **Read the item back** and confirm all three fields landed. The mutations return success
   for a valid-but-wrong option id, so a read-back is the only proof.

The board's `Assignees` column mirrors the issue's assignee, so there is no separate field
to set — assigning the issue in step 1 is enough.

## Choosing the field values

**Status** — where the work actually is, not where you hope it goes:
`Backlog` (filed, not scheduled) → `Ready` (scoped, someone can pick it up) →
`In progress` (being worked *now*) → `In review` (PR open) → `Done` (merged).
Set `In progress` when you file an issue you are about to start, and flip to `In review`
in the same session you open the PR.

**Priority** — what breaks if it waits:

| | Use for |
|---|---|
| `P0` | Blocks a run, a deliverable, or the manuscript path; or produces wrong published numbers |
| `P1` | Actively misleads or wastes time (stale guidance, a footgun, a missing guardrail) but nothing is blocked today |
| `P2` | Real improvement, no active harm — cleanups, nice-to-haves, deferred refactors |

**Size** — implementation effort, *not* importance. `XS` one-line/config; `S` one file or a
contained change; `M` several files or one subsystem, still one PR; `L` multi-PR or spans
both repos; `XL` needs decomposing into child issues before it can start.

## Issue body

Enough for someone with no context — including you in three months — to act:
**Context** (what prompted this) → **Root cause** if known, not just symptoms →
**Findings** (a table when there are several) → **Done when** (a checkbox list that is
literally the acceptance criteria) → **Notes** (constraints, gotchas, links).

State what you *verified* versus what you *suspect*, per AGENTS.md §11. A finding with a
file:line reference is actionable; "the docs seem stale" is not.

## Keep it short

**An issue is scanned later, not read.** Write the body so someone can act from it, and keep
every update comment to **~5 sentences or fewer** — a comment's job is to record a state
change, not to re-explain the work. The diff, the PR, and the "Done when" boxes already
carry the detail; a long comment buries the one line that mattered.

Practical limits: body sections stay tight (Context, Root cause, Findings table, Done when,
Notes — no narration between them); update comments say what changed and what is left, in a
sentence or two each. If an update needs more than that, it is usually a finding that belongs
in the body as a new row or checkbox, not a comment. Never restate the audit trail — reviewers
can read the commits.

Replies to PR review comments follow the same rule, point-to-point, and are written **after**
the fix is applied so each answers with what shipped rather than what is planned.

## Linking the PR

Put `Closes #<n>` in the PR body so the merge closes the issue, move the board item to
`In review`, and let the merge move it to `Done`. When a PR only partly addresses an
issue, write `Refs #<n>` instead and leave the issue open — an auto-closed issue with
unfinished "Done when" boxes is worse than an open one.

## Requesting a Copilot review (and re-requesting after a push)

Copilot is not an org user or team, so `gh pr edit --add-reviewer <anything>` fails with
`'...' not found` (gh 2.x resolves reviewers against the org's users and teams). Use the
REST endpoint with the bot login — it works for the first request **and** for a re-request
after a review has been posted:

```bash
gh api -X POST repos/AllenNeuralDynamics/<repo>/pulls/<n>/requested_reviewers \
    --input <(echo '{"reviewers":["copilot-pull-request-reviewer[bot]"]}')
```

The response's `requested_reviewers` is empty for the bot even on success; confirm with
the issue timeline (`.../issues/<n>/timeline`, a new `review_requested` event) or wait for
the review (`.../pulls/<n>/reviews`, author `copilot-pull-request-reviewer[bot]`). The
reviewer login `Copilot` also works for a first request but was not seen to register a
re-request; GraphQL `requestReviews` rejects the bot's node id. Verified 2026-10-23 on
dispatcher #176 (four rounds).

## Closing the loop — tick the boxes

**The "Done when" list is live state, not a plan you wrote once.** Update it whenever work
lands, and whenever the user asks where something stands. Two failure modes this prevents:
a closed issue whose boxes are all unticked (so nobody can tell what actually shipped from
what got dropped), and a months-old open issue whose boxes are stale (so its remaining work
has to be re-derived from the diff).

When to update, and to what:

| Moment | Do |
|---|---|
| A box's work is **merged** | Tick it: `- [ ]` → `- [x]` |
| Work is done but the PR is **open** | Leave unticked; comment naming the PR and which boxes it covers |
| A box turns out unnecessary or is deferred | Don't tick it — strike it (`~~...~~`) with a one-line why, or move it to a follow-up issue and say which |
| The user asks for status | Reconcile the boxes first, then answer from them |
| Last box ticked | Close the issue and move the board item to `Done` |

**Tick a box only when you have verified the thing, not when you believe you did it.** A
ticked box is a claim someone will rely on instead of re-checking. If you edited a file but
never confirmed the result, the box stays open — that is what the "PR open" row is for.

Ticking by hand means re-uploading the whole issue body, which risks clobbering edits made
in the browser. Use the script, which flips only the lines you name:

```bash
python scripts/board.py --existing 88 --check codebase-map --check wrapper-runtime
python scripts/board.py --existing 88 --check-all --status Done --close
```

It refuses a `--check` substring that matches no unticked box, so a typo fails loudly
instead of silently ticking nothing. Nothing else in the body is touched.

Labels available on the dispatcher: `bug`, `documentation`, `enhancement`, `evaluation`,
`extension`, `interpretability`, `baselines`, `training/pipeline`, `validation`, `blocked`,
`ready-for-agent`, `rename-migration`, `question`, `priority:P0/P1/P2`. The `priority:*`
label duplicates the board field on purpose — the label is visible in issue lists where
board fields are not.

## Copilot review — request it, and resolve every thread

**Every PR gets a Copilot review, and every Copilot comment gets resolved before merge.**
An unresolved Copilot thread is an unanswered reviewer. Automatic review is currently
enabled on these repos, so a new PR is reviewed without asking — the API below is for a
deliberate **re**-review after a fix push, which does not happen automatically.

Requesting a review is GraphQL-only. **REST fails**: `POST
/repos/{owner}/{repo}/pulls/{n}/requested_reviewers` with the Copilot bot returns
`422 Reviews may only be requested from collaborators`.

```python
# bot id in AllenNeuralDynamics: BOT_kgDOCnlnWA
gql("""mutation($pr:ID!,$b:[ID!]!){ requestReviews(input:{
  pullRequestId:$pr, botIds:$b, union:true}){
  pullRequest{ reviewRequests(first:5){ nodes{ requestedReviewer{
    __typename ... on Bot { login } } } } } } }""", pr=pr_node_id, b=["BOT_kgDOCnlnWA"])
```

`union: true` adds to the existing reviewers instead of replacing them.
`RequestReviewsInput` takes `pullRequestId`, `userIds`, `botIds`, `teamIds`, `union`. If the
bot id ever changes, read it off any PR that Copilot has touched — the `Bot` node in
`reviewRequests` or `latestReviews`:

```python
gql("""query($o:String!,$r:String!,$n:Int!){ repository(owner:$o,name:$r){
  pullRequest(number:$n){ id
    latestReviews(first:10){ nodes{ author{ __typename login
      ... on Bot { id } } } } } } }""", o=..., r=..., n=...)
```

Resolve with a reason — `resolveReviewThread` takes
`resolutionReason: ADDRESSED | WONT_FIX | INVALID | OUTDATED` (`INVALID` is the UI's "Incorrect"; `OUTDATED` is "Outdated"):
```python
gql("""mutation($id:ID!){ resolveReviewThread(input:{
  threadId:$id, resolutionReason:ADDRESSED}){ thread{ isResolved } } }""", id=thread_id)
```

Thread ids come from `repository.pullRequest(number: <n>).reviewThreads`.

| Reason | When |
|---|---|
| `ADDRESSED` | you changed the code — including when you fixed it *differently* than suggested |
| `WONT_FIX` | the finding is valid but you are deliberately not acting on it |
| `INVALID` | the finding is wrong |

For `WONT_FIX` and `INVALID`, reply in-thread with why before resolving
(`POST /repos/{owner}/{repo}/pulls/{n}/comments` with `in_reply_to: <comment_id>`). A
resolve with no explanation reads as dismissal.

Three things that cost time when learned the hard way:

- The REST `user.login` on a Copilot comment is **`Copilot`** — not
  `copilot-pull-request-reviewer`, which is the GraphQL `Bot` login. Filtering on the wrong
  one returns zero comments and looks like "no review".
- `PullRequestReviewThread` exposes **no readable `resolutionReason`**, so the reason cannot
  be read back; the mutation accepting the enum is the only confirmation.
- **Do not bulk-resolve.** Copilot re-reviews after a push and may add comments while you
  work; a blanket resolve marks those `ADDRESSED` when they are not. Resolve per finding
  you actually handled, and re-check for new comments after each push.

Copilot does not read replies — per GitHub's docs, comments on its review are visible to
humans but not to Copilot. The reply is for the human reviewer and for the record.

## Sandbox / auth notes

- **No `gh` CLI** in the Claude Science sandbox. Use REST + GraphQL directly with
  `GITHUB_TOKEN` (`urllib`/`requests` + `Authorization: Bearer`).
- **`AllenNeuralDynamics` enforces SAML SSO.** A classic PAT must be SSO-authorized for the
  org or every call returns 403 "Resource protected by organization SAML enforcement". This
  is a one-time authorization in the token's settings, not something to retry around.
- Setting a board field needs the `project` scope; creating issues needs `repo`.

## Cached IDs (verify before trusting)

Field and option ids are stable until someone recreates a field, so treat these as a cache
that saves a round-trip, not as truth. `scripts/board.py --discover` re-reads them; the
GraphQL query is in `references/board-api.md`.

| | ID |
|---|---|
| Project `AIND-behavior-fm` (#184) | `PVT_kwDOBa47bs4BIeG5` |
| Status field | `PVTSSF_lADOBa47bs4BIeG5zg4672s` |
| — Backlog / Ready / In progress / In review / Done | `f75ad846` / `61e4505c` / `47fc9ee4` / `df73e18b` / `98236657` |
| Priority field | `PVTSSF_lADOBa47bs4BIeG5zg467-8` |
| — P0 / P1 / P2 | `79628723` / `0a877460` / `da944a9c` |
| Size field | `PVTSSF_lADOBa47bs4BIeG5zg467_A` |
| — XS / S / M / L / XL | `6c6483d2` / `f784b110` / `7515a9f1` / `817d0097` / `db339eb2` |

The board also carries `Estimate` (number), `Start date`, `Target date`, `Parent issue`, and
`Sub-issues progress` — leave them unset unless the work is being scheduled.

## Reference

- `references/board-api.md` — the raw discovery query and each mutation, for when the
  script does not fit (bulk moves, sub-issues, date fields).
