# Project agent memory

Research harness for LLM agent teams under a fixed token budget. `README.md` is the reference
for design decisions, config format, metrics and limitations; the study design is
`multi-agent-experimental-design.md`.

- Checks: `uv run pytest` (must stay fully offline: tests use the mock provider and the
  synthetic `toy_arithmetic` task, never real datasets or keys) and `uv run ruff check`.
- Never commit `runs/`, downloaded datasets or `.env`. Keys come from `--env-file` /
  `$WORKFORCE_ENV_FILE` / `./.env`; the OpenAI adapter accepts `OPENAI_API_KEY` or `OpenAI_API_Key`.
- Every model call goes through `src/workforce/gateway.py`. Its cache key hashes the full
  request, so editing prompts, procedures or generation params makes a resumed run re-pay
  for those calls. After a scorer or analysis fix, use `workforce summarize <run> --rebuild`
  (offline replay) instead of re-running.
- `src/workforce/protocol/mock_agent.py` parses the prompt shapes from `protocol/prompts.py`
  and the task prompts; keep it in sync when changing prompt markers.
- Solo iteration (`protocol/solo.py`) relies on prefix invariance: a step's request must not
  depend on the horizon or the checkpoints (fixed output ceiling, no budget in prompts), so one
  trajectory yields every checkpoint and continuing from `SoloState.at(c)` replays from cache.
- Test helpers: `tests/conftest.py` (`make_config`, `run_experiment`) and `tests/helpers.py`
  (`run_trial`, `agent_seed_map`).

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
