# Seed files for `aind-dynamic-foraging-bfm-studies`

The files in this directory are the root of the studies repository as it will exist at
its first publish, drafted here so they can be reviewed before the extraction
(`docs/repo-split-plan.md` Step 3; dispatcher #173, part of #124). The assembly commit
copies them verbatim into the extracted repository at the paths below, next to the
`studies/` tree that `git filter-repo` carries over with its history.

| here | in the studies repo |
|---|---|
| `README.md` | `README.md` |
| `AGENTS.md` | `AGENTS.md` (also loaded as `CLAUDE.md` via an `@AGENTS.md` include) |
| `gitignore` | `.gitignore` |
| `gitattributes` | `.gitattributes` (git-LFS for new image / large-JSON objects only) |
| `requirements-dev.txt` | `requirements-dev.txt` |
| `github/workflows/ci.yml` | `.github/workflows/ci.yml` |
| `scripts/check_regeneration.py` | `scripts/check_regeneration.py` |
| `LICENSE`, `CODE_OF_CONDUCT.md` | same (copies of the dispatcher's) |

Dotfiles are stored without the leading dot so they stay visible and are not picked up by
tooling that walks this repository. `MIGRATION.md` and `migration/` are **not** here: they
are produced by the extraction itself (#174) and record what was actually done.

After the studies repository is published, this directory is history: edit the live files
there, not these.
