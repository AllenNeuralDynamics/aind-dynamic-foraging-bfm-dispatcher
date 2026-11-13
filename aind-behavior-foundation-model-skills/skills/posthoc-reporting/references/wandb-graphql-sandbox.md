# Reading W&B from the Claude Science sandbox (GraphQL, no SDK)

**On HPC, export `WANDB_API_KEY` before running any producer.** Several producers
(e.g. `02-gru-scaling-law-ignore/analysis/scaling.py`) use the GraphQL route below
and read `os.environ["WANDB_API_KEY"]` directly, so `make all` dies with
`KeyError: 'WANDB_API_KEY'` even though the machine is authenticated — HPC auth
lives in `~/.netrc`, which only the SDK consults:

```bash
export WANDB_API_KEY=$(grep -A2 'api.wandb.ai' ~/.netrc | grep password | awk '{print $2}')
```

**For a `NeuralDynamics` project, do not export it by hand.** The key is the `han-hou` one, not the
personal `.netrc` key. A producer calls `key_for(ENTITY)` from the studies repo's
`studies/util/wandb_key.py` (studies#157): it returns `~/.config/wandb/api_key.<entity>`, else
`$WANDB_API_KEY`, else `~/.netrc`. New producers use it; an existing frozen producer adopts it the
next time it regenerates (editing it earlier makes `make validate` refuse until the artifacts are
rebuilt).

In the Claude Science Mac sandbox, `wandb.Api()` fails outright: it tries to spawn a
`wandb-core` helper subprocess (`ServicePollForTokenError`) and write config under
`~/.config/wandb`, both blocked. Working route:

- Store the key as the `WANDB_API_KEY` credential; hit the GraphQL endpoint
  directly with `requests`, `auth=('api', KEY)` — no SDK.
- `https://api.wandb.ai` must be allowlisted (one-time `request_network_access`).
- Entity/login is `houhan` (`hanhou`); team entity `AIND-disRNN`.

```python
import requests, os
KEY = os.environ["WANDB_API_KEY"]
GQL = "https://api.wandb.ai/graphql"
q = """query($e:String!,$p:String!,$s:String!){
  project(name:$p,entityName:$e){ sweep(sweepName:$s){ id state
    runs(first:50){ edges{ node{ name displayName state summaryMetrics config }}}}}}"""
r = requests.post(GQL, auth=("api", KEY),
    json={"query": q, "variables": {"e":"AIND-disRNN","p":"hpc_test","s":"<sweep_id>"}},
    timeout=60).json()
# per-run full history: query run(name:$run){ history(samples:100000) }
```

Prefer running the producer on the HPC node when possible; use the GraphQL route
only for ad-hoc pulls from the sandbox.

Gotcha for run deletion (study wrap-up): the GraphQL `deleteRun` payload has
`clientMutationId` + `numDeleted`, not `success` — selecting `success` returns
HTTP 400.
