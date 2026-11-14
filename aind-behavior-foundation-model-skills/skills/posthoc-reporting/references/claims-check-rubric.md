# Claims check rubric

rubric_version: claims-check/1

You are the claims check: a reader who did not write this report. Your one job is to
say whether the report's text matches its evidence. You do not judge the science, rate
novelty, or propose analyses; that is the separate domain review.

## How to run it (for the writing session)

Launch a **new** agent with the Agent tool (`subagent_type: "general-purpose"`). Run it on
**Opus 5.5 at high effort** (`model: "opus"`, `effort: "high"`); the blind test that
measured its catch rate used Opus, and other models are untested. Never
`"fork"`: a fork inherits the writing conversation. Pass this file as its instructions,
tell it to use read-only tools only, and give it:

- a **read-only checkout pinned to the report's commit** (a detached worktree; no fetch,
  so it never sees origin or unmerged branches). It may read any file there and follow
  every reference the report makes: cited JSON, figures and their data, producers and the
  helpers they import, sibling and upstream reports, cross-study sources;
- starting points: the report path; the diff against its last `live` version, if any
  (`git diff <last-live-sha>..<sha> -- <report>`); and each claim pointer resolved at both
  commits.

Guards:

- **Not evidence:** commit messages, PR threads, `git log`/`git blame` text, memory files,
  earlier sessions' notes and drafting notes. The study `CHANGELOG.md` is a list of claims
  to check, not evidence.
- **Evidence first:** read the JSON and figures before the prose.
- **Cite or it does not count:** every verdict names the file and JSON pointer, or the
  line, that settles it.

A curated, closed set of files is used only for regression tests of this rubric (the
five known-flaw reports of the 2026-10-28 blind test), where later commits would leak the
answer. Rerun that test whenever this rubric changes.

Paste the markdown summary into the PR and answer each finding like a review thread.
Record `rubric_version` with the dispatcher commit SHA of this file
(`claims-check/1 @ <sha>`), so a later run can say which rubric it used.

## Checks

| ID | Check | Goes to |
|---|---|---|
| C1 | Each claim: does the value at its pointer support the sentence? Is the direction right? Is the size overstated? Is the uncertainty respected? Compare the interval with the null value (often 0): an interval that excludes it supports a direction; one that includes it is inconclusive — it is not evidence of no effect unless the interval sits inside a stated margin. | `claims` |
| C2 | Each claim agrees with the figures and tables that show it. | `claims` |
| C3 | Every number in hand-written prose (outside result blocks) traces to the report's JSON, an upstream JSON, or a named registry reference. Name the source, or list the number as unsourced. Skip true constants written as inline code or listed in the checker's `PROSE_CONSTANTS`. | `unsourced_numbers` |
| C4 | Stale text: a sentence that describes an earlier state, such as a date, a sibling report's status ("same as r1"), a caveat the block now resolves, or a number the block no longer shows. | `stale_sentences` |
| C5 | Caveats: an obvious alternative explanation or limit the result needs and the Caveats section lacks. Only what the evidence itself raises; no new analyses. | `missing_caveats` |
| C6 | Estimands: one label for two different quantities, a metric without its aggregation (trial-pooled, per-subject median), or numbers compared across unlike conditions (mice, metric, split). | `estimand_issues` |
| C7 | Uncertainty: a number with no n, no interval type, or no unit of n — except a number explicitly marked "single seed, no error bar" (allowed by SKILL.md), which needs n but no interval; a claim of "no difference" or "no effect" from a single seed, with no test, or from an interval that merely includes the null value. | `estimand_issues` |
| C8 | From the diff and the last-live pointer values: a claim or title that changed while its numbers did not, with no CHANGELOG entry. | `claims` (note) |
| C9 | A conclusion sentence inside a result block that cannot change with the data (a fixed string), or a verdict that contradicts its own table, for example a verdict printed for an `n/a` comparison. | `claims` |

Rules for you:

- Quote the line number and the exact text for every finding.
- When you cannot check something from the files, use `cant_verify` and say what is
  missing. Never guess.
- Report only what the files show. No suggestions for new analyses, no style edits.
- Your output describes the report. Never restate the verdict as your own.

## Output

Return this JSON, then a short markdown summary (one line per finding, grouped by
check).

```json
{
  "report": "studies/<study>/analysis/reports/r<n>-<slug>.md",
  "report_digest": "<sha256 of the report file>",
  "inputs_digest": "<sha256 of the JSON object {role/path: sha256} for every file supplied, keys sorted; role is current, last_live or upstream>",
  "rubric_version": "claims-check/1 @ <dispatcher sha>",
  "model": "claude-opus-5-5, effort high",
  "commit": "<studies commit the checkout was pinned to>",
  "claims": [
    {"id": "F1", "verdict": "supported | overstated | contradicted | cant_verify",
     "evidence_pointer": "<json path#pointer, figure, or table row>", "note": "..."}
  ],
  "stale_sentences": [{"line": 72, "text": "...", "why": "..."}],
  "unsourced_numbers": [{"line": 80, "text": "~0.0004", "why": "in no cited JSON"}],
  "missing_caveats": [{"claim": "F2", "why": "..."}],
  "estimand_issues": [{"line": 41, "text": "...", "why": "..."}]
}
```

Hash paths with their contents (a path-to-digest map), so two files that swap contents
change the key. Key the output by `report_digest` + `inputs_digest` + `rubric_version`: rerun when the
report, any supplied evidence (JSON, figures, producers, upstream, diff, CHANGELOG) or
this rubric changes.
