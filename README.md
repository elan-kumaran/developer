# Claude API Agent Experiments

An early-stage Python project for experimenting with the [Anthropic Claude API](https://docs.anthropic.com/), exploring a deliberate progression from a simple in-process tool registry all the way to a hub-and-spoke multi-agent system — and finally re-expressing that design on the higher-level Claude Agent SDK.

## The five scripts

| Script | What it does |
| --- | --- |
| `test.py` | Minimal smoke test: one `messages.create` call, prints the text blocks. |
| `agent.py` | A small agentic framework with a pluggable tool registry and a manual agentic loop. The model routes each request to the right registered tool (or answers directly). |
| `agent_mcp.py` | Same functionality as `agent.py`, but the tools are served over **MCP** (Model Context Protocol) instead of being registered in-process. One file plays both roles (server + client). |
| `agent_hub.py` | A **hub-and-spoke multi-agent** system: each spoke is its own Claude agent (own system prompt + own tools) exposed over MCP as a single `ask_<spoke>_agent(task)` delegation tool. The hub is an orchestrator loop whose only tools are those sub-agents. |
| `agent_hub_sdk.py` | The **same hub-and-spoke design** as `agent_hub.py`, but built on the **Claude Agent SDK** (`claude-agent-sdk`). Spokes are declarative `AgentDefinition`s and the SDK supplies the orchestration. |

The four `anthropic`-based scripts form a deliberate progression:

```
in-process tool registry   →   tools behind MCP   →   whole agents behind MCP
      (agent.py)                 (agent_mcp.py)            (agent_hub.py)
```

`agent_hub_sdk.py` then re-expresses that final hub-and-spoke step on a higher-level SDK. The same three tool implementations — `calculator`, `current_datetime`, `text_stats` — are reused across all of them.

## Setup

Requires **Python 3.12** (MCP needs 3.10+) and an Anthropic API key.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with your API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

`.env` and `.venv` are gitignored.

## Running

```bash
python test.py                        # smoke test

python agent.py "your question"       # registry agent on one query
python agent.py                       # built-in demo queries

python agent_mcp.py "your question"   # MCP agent (auto-spawns its own server)
python agent_mcp.py --server          # MCP server mode only (normally auto-spawned)

python agent_hub.py "your question"   # orchestrator (auto-spawns every spoke server)
python agent_hub.py                   # built-in demo query
python agent_hub.py --server math     # run one spoke as an MCP server (math|text|time)
```

> Run the `agent_mcp.py` / `agent_hub.py` scripts from this project's `.venv`, since they shell out to spawn their MCP server subprocesses and the spawned servers inherit that interpreter.

### `agent_hub_sdk.py` (separate toolchain)

This script uses the Claude Agent SDK, which spawns the Claude Code CLI as a subprocess — so it needs an extra dependency plus the CLI and Node 18+:

```bash
pip install -r requirements-agent-sdk.txt   # claude-agent-sdk
npm install -g @anthropic-ai/claude-code     # the CLI the SDK spawns (Node 18+)

python agent_hub_sdk.py "your question"      # orchestrator + AgentDefinition spokes
python agent_hub_sdk.py                       # built-in demo query
```

## Models

The `anthropic`-based scripts use `claude-haiku-4-5`; `agent_hub_sdk.py` uses the SDK's `"haiku"` alias (resolves to the latest Haiku).

## Notes

See [`CLAUDE.md`](./CLAUDE.md) for a deeper architectural walkthrough of each script.
