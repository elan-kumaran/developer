# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Early-stage Python project for experimenting with the Anthropic Claude API. Seven scripts:

- `test.py` — minimal smoke test: one `messages.create` call, prints the text blocks.
- `agent.py` — a small agentic framework with a pluggable tool registry and a manual agentic loop. The model routes each request to the right registered tool (or answers directly).
- `agent_mcp.py` — same functionality as `agent.py`, but the tools are served over **MCP** instead of being registered in-process. One file plays both roles (server + client).
- `agent_hub.py` — a **hub-and-spoke multi-agent** system: each spoke is its own Claude agent (own system prompt + own tools) exposed over MCP as a single `ask_<spoke>_agent(task)` delegation tool; the hub is an orchestrator Claude loop whose only tools are those sub-agents. One file plays all roles (orchestrator + each spoke server).
- `agent_hub_sdk.py` — the **same hub-and-spoke design as `agent_hub.py`**, but built on the **Claude Agent SDK** (`claude-agent-sdk`) instead of the raw `anthropic` API. Spokes are declarative `AgentDefinition`s and the SDK supplies the orchestration (delegation via its built-in Task tool). Uses a different SDK + toolchain — see its own section and `requirements-agent-sdk.txt`.
- `agent_a2a.py` — also on the **Claude Agent SDK**, but instead of local spokes it reaches *outward* two ways: (1) to a **remote (HTTP) MCP server** — public DeepWiki, no auth, pointed at by URL rather than spawned as a subprocess; and (2) to a **foreign-framework agent over the A2A (Agent2Agent) protocol** — the Analyst agent, built directly on the `a2a-sdk` (its own AgentCard + AgentExecutor, deterministic, no LLM). The Claude agent is the A2A *client*; the Analyst is the A2A *server*. Needs `a2a-sdk` — see its own section and `requirements-a2a.txt`.
- `agent_memory.py` — a standalone teaching demo (back on the raw `anthropic` SDK + a manual loop, like `agent.py`) that implements the **six memory types of agentic architectures** as six separate, inspectable stores wired into one agent: sensory, short-term/working, long-term, episodic, semantic, procedural. The volatile tiers (sensory, working) reset per session; the durable tiers (episodic/semantic/procedural, unified by a long-term facade) persist to a `.agent_memory/` directory. Not part of the progression above — see its own section. Uses only `requirements.txt`.

The four `anthropic`-based scripts form a deliberate progression: in-process tool registry (`agent.py`) → tools behind MCP (`agent_mcp.py`) → whole agents behind MCP (`agent_hub.py`). `agent_hub_sdk.py` then re-expresses that final hub-and-spoke step on a higher-level SDK. The same three tool implementations (`calculator`, `current_datetime`, `text_stats`) are reused across the first four; `agent_a2a.py` instead demonstrates *outward* interop (remote MCP + cross-framework A2A) and reuses only the text-analysis idea (in its own deterministic `analyze_text`). `agent_hub_sdk.py` and `agent_a2a.py` are the two files on the Claude Agent SDK + Claude Code CLI toolchain.

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
python agent_memory.py               # the two-session six-memory-types demo (no extra deps)
python agent_memory.py "your message"  # one turn against the persisted long-term memory
python agent_memory.py --inspect     # print what's in .agent_memory/ ; --forget wipes it
```

`agent_hub_sdk.py` needs an extra dependency + the Claude Code CLI (see its architecture section):

```bash
pip install -r requirements-agent-sdk.txt   # claude-agent-sdk (separate from requirements.txt)
npm install -g @anthropic-ai/claude-code     # the CLI the SDK spawns (Node 18+)
python agent_hub_sdk.py "your question"      # run the orchestrator + its AgentDefinition spokes
python agent_hub_sdk.py                       # run the built-in demo query
```

`agent_a2a.py` needs the same CLI + Node 18+, plus the A2A SDK, and network access to `https://mcp.deepwiki.com`:

```bash
pip install -r requirements-a2a.txt          # claude-agent-sdk + a2a-sdk + uvicorn
npm install -g @anthropic-ai/claude-code      # the CLI the SDK spawns (Node 18+)
python agent_a2a.py "your question"           # spawn the Analyst A2A agent, then run the hub
python agent_a2a.py                            # run the built-in demo query
python agent_a2a.py --peer                     # run ONLY the Analyst A2A server (normally auto-spawned)
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

## `agent_a2a.py` architecture

Also on the Claude Agent SDK, but where `agent_hub_sdk.py` delegates *inward* to local `AgentDefinition` spokes, this file interoperates *outward* with two independent systems. The file is organized in three parts and runs in two modes selected by argv (`--peer` → the Analyst server only; otherwise → the orchestrator, which auto-spawns the peer):

- **Part 1 — the Analyst agent, built on the A2A SDK (a *different framework*).** No Claude, no LLM: `AnalystAgentExecutor(AgentExecutor)` wraps a deterministic `analyze_text` (word/char/sentence counts + top keywords). It follows the `a2a-sdk` server contract — subclass `AgentExecutor`, drive a `TaskUpdater` through `TASK_STATE_WORKING → TASK_STATE_COMPLETED`, emit the answer as an artifact. `build_peer_app()` assembles a plain **Starlette** app from `create_agent_card_routes(card)` (serves `/.well-known/agent-card.json`) + `create_jsonrpc_routes(handler, "/")`, wired through a `DefaultRequestHandler` + `InMemoryTaskStore`; `run_peer()` serves it with **uvicorn** on `127.0.0.1:9100`.
- **Part 2 — the A2A client bridge (where agent-to-agent actually happens).** `ask_peer_over_a2a(task_text)` resolves the Analyst's card via `A2ACardResolver`, opens a client with `create_client(agent=card, client_config=ClientConfig(...))`, sends a `SendMessageRequest(message=new_text_message(...))`, and stitches text out of the streamed `StreamResponse` oneof (`task` / `message` / `status_update` / `artifact_update`). This is wrapped in a single Claude SDK `@tool`, `ask_analyst_agent`, so "delegate to the Analyst" is one tool call from the orchestrator's view. That tool is bundled into an in-process `create_sdk_mcp_server(name="bridge", ...)` → `mcp__bridge__ask_analyst_agent`.
- **Part 3 — the hub.** `run_orchestrator` calls `query()` with `ClaudeAgentOptions(mcp_servers={"bridge": <sdk server>, "deepwiki": {"type": "http", "url": "https://mcp.deepwiki.com/mcp"}}, allowed_tools=[ASK_ANALYST, DW_STRUCTURE, DW_CONTENTS, DW_ASK], ...)`. `run_with_peer` spawns `sys.executable __file__ --peer` as a subprocess, polls the agent-card endpoint via `_wait_for_peer()` until it's live, runs the hub, then tears the subprocess down in a `finally`.

Two things here differ from every other file in the repo: the MCP server is **remote HTTP** (a `{"type": "http", "url": ...}` config the SDK connects to, *not* a stdio subprocess we spawn), and its tools are named `mcp__deepwiki__<tool>` (`read_wiki_structure` / `read_wiki_contents` / `ask_question`) — remote MCP tools use the same `mcp__<server>__<tool>` naming as in-process ones, so they must be listed in `allowed_tools` by that name or the non-interactive CLI denies them. The Analyst is deterministic on purpose (no second API key needed); swapping its `AgentExecutor` for a LangGraph/CrewAI/OpenAI agent would leave the rest of the file unchanged — that's the payoff of a shared protocol. Fully **async**; still reads `ANTHROPIC_API_KEY` from `.env`.

## `agent_memory.py` architecture

A standalone demo of the **six memory types** used in agentic architectures, each a distinct class, all wired into one manual `anthropic` agentic loop (same loop shape as `agent.py`). It is *not* part of the tool-registry → MCP → hub progression; it is orthogonal, about memory rather than delegation. The six, grouped by volatility:

- **Volatile (per-session, RAM only):** `SensoryMemory` — a raw-input buffer with a TTL that decays almost immediately (input is `perceive()`d, `read()` once to promote it, then cleared). `WorkingMemory` — a bounded sliding window (`deque(maxlen=…)`) of clean conversation turns plus a task `scratchpad` that is wiped at the start of each turn; `as_messages()` is what feeds the API `messages`.
- **Durable (persisted to `.agent_memory/`):** `LongTermMemory` is a *facade*, not a fourth content type — it owns the directory and the `load_all()/save_all()` lifecycle that backs the next three. `EpisodicMemory` — append-only `(timestamp, input, response)` episodes with keyword-overlap `recall()`. `SemanticMemory` — key→value facts. `ProceduralMemory` — a list of learned behavioral rules. Each persists to its own JSON file.
- **How memory becomes behavior:** `MemoryAgent.turn()` runs sensory→working, then `_system_prompt()` assembles the system prompt from `BASE_SYSTEM` + procedural rules + semantic facts + episodes auto-recalled for the current input. Three client-side tools let the model *write* memory: `remember_fact` (semantic), `learn_rule` (procedural), `recall_memory` (read). After each turn it records an episode and calls `save_all()`. A dimmed dashboard of all six stores prints to stderr per turn.

The persistence proof is `run_demo()` (the no-arg default): session 1 teaches facts + a rule; session 2 constructs a **new** `MemoryAgent` (empty sensory/working) against the same directory and still knows the user and applies the learned rule, because the durable tiers reloaded from disk. `--inspect` dumps the store; `--forget` wipes it. `.agent_memory/` is gitignored.

## Client-side vs server-side tools

The registry/`run_tool` path is for **client-side** tools only (we execute them). **Server-side** tools (e.g. web search `{"type": "web_search_20260209", "name": "web_search"}`) run on Anthropic's infrastructure — add them directly to the `tools` list in `run_agent`, *not* the registry, and the loop must then also handle `stop_reason == "pause_turn"` (re-send to resume). Note: web search is currently **not enabled for this org** (returns 400 `Web search is not enabled`); enable it in the Anthropic Console before using it.

## Key details

- `requirements.txt` covers the four `anthropic`-based scripts: its direct dependencies are `anthropic[mcp]` (the SDK plus its MCP extra, which pulls in `mcp`) and `python-dotenv`; everything else is transitive (it's a full `pip freeze`). `agent_hub_sdk.py`'s extra dependency lives in **`requirements-agent-sdk.txt`** (`claude-agent-sdk`), kept separate because it's a different toolchain that also needs the Claude Code CLI + Node 18+. `agent_a2a.py`'s dependencies live in **`requirements-a2a.txt`** (`claude-agent-sdk` + `a2a-sdk` + `uvicorn`) — same CLI/Node toolchain as `agent_hub_sdk.py`, plus the A2A stack (`a2a-sdk` pulls in `starlette`; `uvicorn` serves the Analyst).
- `anthropic.Anthropic()` / `AsyncAnthropic()` read `ANTHROPIC_API_KEY` from the environment automatically — `load_dotenv()` must run first.
- `response.content` is a list of content blocks; filter on `block.type == "text"` to extract text and `block.type == "tool_use"` for tool calls.
- When appending an assistant turn to `messages`, append the whole `response.content` (not just text) so `tool_use` blocks are preserved for the next request.
- `agent_mcp.py` and `agent_hub.py` shell out to `sys.executable` to launch their MCP server subprocess(es), so always run them from this project's `.venv` (where `mcp` is installed) — the spawned servers inherit that interpreter. `agent_a2a.py` does the same to spawn its Analyst A2A server (`sys.executable __file__ --peer`), so run it from `.venv` too (where `a2a-sdk` + `uvicorn` live).
- The `anthropic`-based scripts use the `claude-haiku-4-5` model (the `MODEL` constant in each agent; a string literal in `test.py`); `agent_hub_sdk.py` and `agent_a2a.py` use the SDK's `"haiku"` alias (each its own `MODEL` constant), which resolves to the latest Haiku. When choosing models, prefer current Claude model IDs.
- `agent.py`'s `@tool` decorator emits **strict** schemas (`"strict": True` + `"additionalProperties": False`); `agent_mcp.py`/`agent_hub.py` instead let FastMCP / hand-written schemas describe the tools, so they are not strict. `agent_hub_sdk.py` uses the Claude Agent SDK's own `@tool` decorator — a different decorator entirely (`@tool(name, description, schema)` wrapping an `async` handler), not `agent.py`'s registry decorator.
