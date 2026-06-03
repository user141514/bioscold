# Ruflo — Claude Code Configuration

## Project Environment

- This project must default to the Conda environment `accfg_migrated`.
- Run Python, tests, training, evaluation, scripts, notebooks via `conda run -n accfg_migrated ...` or after activating `accfg_migrated`.
- Do not use `base`, `accfg`, `diffusion_model`, global Python, or any other environment unless explicitly overridden.
- Before long-running work, verify: `conda env list` and `conda run -n accfg_migrated python -V`. If `accfg_migrated` is missing or broken, stop and report.

## Rules

- Do what has been asked; nothing more, nothing less
- NEVER create files unless absolutely necessary — prefer editing existing files
- NEVER create documentation files unless explicitly requested
- NEVER save working files or tests to root — use `/src`, `/tests`, `/docs`, `/config`, `/scripts`
- ALWAYS read a file before editing it
- NEVER commit secrets, credentials, or .env files
- Keep files under 500 lines
- Validate input at system boundaries

## Agent Coordination

Use memory-as-bus pattern: subagents read inputs from memory keys, write outputs to memory keys, then complete. Lead verifies outputs in memory before spawning dependent agents. Parallelize only when work is genuinely independent. Every subagent brief must include degraded-mode instructions (read source files directly, write to memory keys). See `.claude/reference/agent-comms.md` for full patterns and anti-patterns.

## Swarm & Routing

Swarm: hierarchical-mesh, max 15 agents. Config in settings.json. Use swarm for 3+ file changes, new features, cross-module refactoring, API changes, security, performance. Skip for single-file edits, 1-2 line fixes, docs, config, questions.

## Memory & Learning

Before tasks: search patterns with `memory_search`. After success: store with `memory_store`. Key tools: `memory_store`, `memory_search`, `swarm_status`, `agent_spawn`, `hooks_route`, `aidefence_scan`.

### Verified Claude CLI -> Ruflo Tool Invocation

Verified on 2026-06-03: the reliable way to run Ruflo MCP tools through a real tool-using agent is Claude CLI print mode, not Ruflo `agent_execute`.

Use this pattern:

```powershell
$cfg = Get-Content -Raw 'C:\Users\Administrator\.claude\settings.json' | ConvertFrom-Json
foreach ($p in $cfg.env.PSObject.Properties) { Set-Item -Path "Env:$($p.Name)" -Value ([string]$p.Value) }
if ([string]::IsNullOrWhiteSpace($env:ANTHROPIC_API_KEY) -and -not [string]::IsNullOrWhiteSpace($env:ANTHROPIC_AUTH_TOKEN)) { $env:ANTHROPIC_API_KEY = $env:ANTHROPIC_AUTH_TOKEN }
claude --bare -p --verbose --output-format stream-json --model deepseek-v4-flash --permission-mode bypassPermissions --mcp-config .mcp.json --strict-mcp-config --allowedTools mcp__ruflo__task_create -- "Call the Ruflo task_create tool exactly once ..."
```

Evidence: `claude mcp get ruflo` reports connected; `stream-json` shows real `tool_use` / `tool_result` events for `mcp__ruflo__config_list` and `mcp__ruflo__task_create`; the created task was later visible via `ruflo task status`. Ruflo `agent_execute` only performs a direct model call and does not expose automatic tool-use. Also, Ruflo `memory_store` currently returns success but did not persist across separate processes in this project; prefer task/status or file-backed notes for durable verification.

Background workers: `audit` (after security), `optimize` (after performance), `testgaps` (after features), `map` (after 5+ file changes), `document` (after API changes).

## Agents

Core: `coder`, `reviewer`, `tester`, `planner`, `researcher`. Architecture: `system-architect`, `backend-dev`, `mobile-dev`. Security: `security-architect`, `security-auditor`. Performance: `performance-engineer`, `perf-analyzer`. GitHub: `pr-manager`, `code-review-swarm`, `issue-tracker`, `release-manager`. Any string works as custom agent type.

## Setup

```bash
claude mcp add claude-flow -- npx -y @claude-flow/cli@latest
npx @claude-flow/cli@latest daemon start
npx @claude-flow/cli@latest doctor --fix
```
