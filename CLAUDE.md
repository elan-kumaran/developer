# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Early-stage Python project for experimenting with the Anthropic Claude API. Five scripts:

- `test.py` — minimal smoke test: one `messages.create` call, prints the text blocks.
- `agent.py` — a small agentic framework with a pluggable tool registry and a manual agentic loop. The model routes each request to the right registered tool (or answers directly).
- `agent_mcp.py` — same functionality as `agent.py`, but the tools are served over **MCP** instead of being registered in-process. One file plays both roles (server + client).
- `agent_hub.py` — a **hub-and-spoke multi-agent** system: each spoke is its own Claude agent (own system prompt + own tools) exposed over MCP as a single `ask_<spoke>_agent(task)` delegation tool; the hub is an orchestrator Claude loop whose only tools are those sub-agents. One file plays all roles (orchestrator + each spoke server).
- `agent_hub_sdk.py` — the **same hub-and-spoke design as `agent_hub.py`**, but built on the **Claude Agent SDK** (`claude-agent-sdk`) instead of the raw `anthropic` API. Spokes are declarative `AgentDefinition`s and the SDK supplies the orchestration (delegation via its built-in Task tool). This is the **only** file that uses a different SDK + toolchain — see its own section and `requirements-agent-sdk.txt`.

The four `anthropic`-based scripts form a deliberate progression: in-process tool registry (`agent.py`) → tools behind MCP (`agent_mcp.py`) → whole agents behind MCP (`agent_hub.py`). `agent_hub_sdk.py` then re-expresses that final hub-and-spoke step on a higher-level SDK. The same three tool implementations (`calculator`, `current_datetime`, `text_stats`) are reused across all of them.

`claude_ai_arch/` is an empty placeholder directory. Not a git repository.

## Setup & Commands

Single virtualenv (`.venv`, **Python 3.12** — MCP requires 3.10+) and `.env` for the API key.

```bash
source .venv/bin/activate          # activate the virtualenv
pip install -r requirements.txt    # install dependencies (includes anthropic[mcp] → mcp)
python test.py                     # run the smoke test
python agent.py "your question"    # run the registry agent on one query
python agent.py                    # run the built-in demo queries
python agent_mcp.py "your question"  # run the MCP agent (auto-spawns its own server)
python agent_mcp.py --server         # MCP server mode only (normally auto-spawned)
python agent_hub.py "your question"  # run the orchestrator (auto-spawns every spoke server)
python agent_hub.py                  # run the built-in demo query
python agent_hub.py --server math    # run one spoke as an MCP server (normally auto-spawned; math|text|time)
```

`agent_hub_sdk.py` needs an extra dependency + the Claude Code CLI (see its architecture section):

```bash
pip install -r requirements-agent-sdk.txt   # claude-agent-sdk (separate from requirements.txt)
npm install -g @anthropic-ai/claude-code     # the CLI the SDK spawns (Node 18+)
python agent_hub_sdk.py "your question"      # run the orchestrator + its AgentDefinition spokes
python agent_hub_sdk.py                       # run the built-in demo query
```

Note: `agent_mcp.py`'s module docstring references a `.venv-mcp` and Python 3.9 — that is stale; there is one `.venv` on Python 3.12 and it runs MCP fine.

`.env` defines `ANTHROPIC_API_KEY` (loaded by `python-dotenv`). It is gitignored along with `.venv`.

To recreate the venv from scratch: `python3.12 -m venv .venv && .venv/bin/pip install "anthropic[mcp]" python-dotenv` (then `pip freeze > requirements.txt`).

## `agent.py` architecture

The whole framework is one file, organized around a tool registry:

- **`REGISTRY`** maps tool name → `{"schema", "fn"}`. The **`@tool(name, description, input_schema)`** decorator is the only registration point — it stores both the API tool definition and the Python implementation. Adding a tool requires nothing else; `tool_schemas()` and `run_tool()` read from the registry automatically.
- **`run_agent(query)`** is a model-agnostic manual agentic loop: call the API → if `stop_reason == "tool_use"`, execute every `tool_use` block via `run_tool()` and return **all** results in a single user message → repeat until `end_turn`. It also handles `refusal` and caps iterations with `max_turns`.
- Tools registered so far are all **client-side** and self-contained (no external services): `calculator` (safe arithmetic via `ast`, never `eval`), `current_datetime`, `text_stats`.
- `run_tool` wraps every call in try/except and returns `tool_result` blocks with `is_error: true` on failure, so a tool exception becomes recoverable model context instead of a crash.

## `agent_mcp.py` architecture

Mirrors `agent.py`'s behavior but moves the tools behind the Model Context Protocol. The single file runs in two modes selected by argv:

- **Server mode** (`--server`): a `FastMCP` stdio server registers the same three tool functions (`calculator`, `current_datetime`, `text_stats`). FastMCP derives each tool's input schema from type hints and its description from the docstring — so the tool's docstring *is* the model-facing description.
- **Client/agent mode** (default): `run_agent` spawns this same file as a subprocess in `--server` mode via `StdioServerParameters(command=sys.executable, args=[__file__, "--server"])`, connects with an MCP `ClientSession`, calls `list_tools()`, wraps each with `async_mcp_tool(tool, session)`, and hands them to the **SDK tool runner** (`client.beta.messages.tool_runner`). The tool runner drives the whole loop (call API → execute the chosen MCP tool over the session → feed results back → repeat) — there is no manual loop here, unlike `agent.py`.

This path is **async** (`AsyncAnthropic`); `main` wraps each query in `asyncio.run`. The tool functions are shared code, but only the server process ever executes them — the client discovers and invokes them purely through MCP.

## `agent_hub.py` architecture

Same MCP plumbing as `agent_mcp.py`, but the unit of delegation is a whole agent rather than a function. The single file runs in three modes selected by argv (no args → demo, `--server <spoke>` → one spoke, otherwise → orchestrator):

- **`TOOLSPECS`** holds the real tool implementations + API schemas (the work spokes actually do). **`SPOKES`** defines each spoke as `{system, tools, delegate_desc}` — its system prompt, which `TOOLSPECS` tools it owns, and the description of its delegation tool. Adding a spoke = one `SPOKES` entry.
- **Spoke / sub-agent half** (`--server <name>`): a `FastMCP` server exposes exactly one tool, `ask_<name>_agent(task)`, whose handler runs `_subagent_loop` — a self-contained manual agentic loop (Claude + that spoke's own tools, capped at 8 turns, errors fed back as `is_error` tool_results) returning the spoke's final text.
- **Hub / orchestrator half** (default): `run_orchestrator` spawns *one MCP server subprocess per spoke* (via `StdioServerParameters(command=sys.executable, args=[__file__, "--server", name])`), held open together with an `AsyncExitStack`, collects each server's single delegation tool through `async_mcp_tool`, and drives them with the SDK `tool_runner`. The hub's only tools are the spokes; `ORCH_SYSTEM` tells it to decompose the request, delegate independent subtasks, then synthesize.

Two levels of Claude here: the orchestrator loop reasons about *which spoke*, and each spoke's `_subagent_loop` reasons about *which real tool*. Fully **async**.

## `agent_hub_sdk.py` architecture

The same hub-and-spoke intent as `agent_hub.py`, but the orchestration is the **Claude Agent SDK**'s job, not hand-written. This is the project's one departure from the raw `anthropic` SDK — it imports from `claude_agent_sdk` instead.

- **Tools**: the three functions are SDK `@tool` coroutines returning the SDK content-block shape (`{"content": [{"type": "text", "text": ...}]}`), bundled into one in-process server via `create_sdk_mcp_server(name="tools", ...)`. They are addressed as `mcp__tools__<toolname>` (the SDK's `mcp__<server>__<tool>` naming).
- **Spokes** are declarative `AgentDefinition`s (no loop code): `description` is what the orchestrator reads to pick a spoke (the analog of `agent_hub.py`'s `delegate_desc`), `prompt` is the spoke's system prompt (analog of `system`), and `tools=[...]` locks each spoke to its single tool. `model="haiku"` (alias → latest Haiku).
- **Hub**: `run_orchestrator` builds `ClaudeAgentOptions(agents=SPOKES, mcp_servers={"tools": ...}, allowed_tools=[<the three tools>, "Task"], system_prompt=ORCH_SYSTEM, model=...)` and calls `query(prompt, options)`. The SDK drives the whole loop; the orchestrator delegates through the SDK's built-in **Task** tool (hence `"Task"` in `allowed_tools`), and `ORCH_SYSTEM` tells it to delegate rather than use tools directly. The final answer is read off the `ResultMessage.result`.

Key contrasts with `agent_hub.py`: delegation is the SDK's built-in Task tool (not a custom `ask_<spoke>_agent` MCP tool); spokes are in-process `AgentDefinition` objects (not stdio MCP subprocesses); and **we write no loop** — `query()` runs it. Unlike the `anthropic`-based files, this SDK spawns the **Claude Code CLI** as a subprocess to run the agent, so it needs the CLI + Node 18+ in addition to `pip install -r requirements-agent-sdk.txt`. It still reads `ANTHROPIC_API_KEY` from `.env`.

## Client-side vs server-side tools

The registry/`run_tool` path is for **client-side** tools only (we execute them). **Server-side** tools (e.g. web search `{"type": "web_search_20260209", "name": "web_search"}`) run on Anthropic's infrastructure — add them directly to the `tools` list in `run_agent`, *not* the registry, and the loop must then also handle `stop_reason == "pause_turn"` (re-send to resume). Note: web search is currently **not enabled for this org** (returns 400 `Web search is not enabled`); enable it in the Anthropic Console before using it.

## Key details

- `requirements.txt` covers the four `anthropic`-based scripts: its direct dependencies are `anthropic[mcp]` (the SDK plus its MCP extra, which pulls in `mcp`) and `python-dotenv`; everything else is transitive (it's a full `pip freeze`). `agent_hub_sdk.py`'s extra dependency lives in **`requirements-agent-sdk.txt`** (`claude-agent-sdk`), kept separate because it's a different toolchain that also needs the Claude Code CLI + Node 18+.
- `anthropic.Anthropic()` / `AsyncAnthropic()` read `ANTHROPIC_API_KEY` from the environment automatically — `load_dotenv()` must run first.
- `response.content` is a list of content blocks; filter on `block.type == "text"` to extract text and `block.type == "tool_use"` for tool calls.
- When appending an assistant turn to `messages`, append the whole `response.content` (not just text) so `tool_use` blocks are preserved for the next request.
- `agent_mcp.py` and `agent_hub.py` shell out to `sys.executable` to launch their MCP server subprocess(es), so always run them from this project's `.venv` (where `mcp` is installed) — the spawned servers inherit that interpreter.
- The `anthropic`-based scripts use the `claude-haiku-4-5` model (the `MODEL` constant in each agent; a string literal in `test.py`); `agent_hub_sdk.py` uses the SDK's `"haiku"` alias (its own `MODEL` constant), which resolves to the latest Haiku. When choosing models, prefer current Claude model IDs.
- `agent.py`'s `@tool` decorator emits **strict** schemas (`"strict": True` + `"additionalProperties": False`); `agent_mcp.py`/`agent_hub.py` instead let FastMCP / hand-written schemas describe the tools, so they are not strict. `agent_hub_sdk.py` uses the Claude Agent SDK's own `@tool` decorator — a different decorator entirely (`@tool(name, description, schema)` wrapping an `async` handler), not `agent.py`'s registry decorator.
