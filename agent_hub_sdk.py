"""The same hub-and-spoke design as agent_hub.py, but built on the **Claude Agent
SDK** (`claude-agent-sdk`) instead of hand-rolling the orchestration on the raw
`anthropic` Messages API + MCP.

The contrast with agent_hub.py:
  - There, each spoke is its *own* Claude loop we wrote (`_subagent_loop`), exposed
    over stdio MCP, and the hub is a `tool_runner` loop we drive by hand.
  - Here, each spoke is an **`AgentDefinition`** — a declarative subagent (its own
    description, system prompt, model, and restricted tool set). The SDK provides
    the orchestrator's delegation machinery: a built-in **Task** tool the main
    agent uses to hand a subtask to the right spoke. We write zero loop code.

    Hub (the SDK's main agent, ORCH_SYSTEM)
      ├── Task → math   subagent  (calculator only)
      ├── Task → text   subagent  (text_stats only)
      └── Task → time   subagent  (current_datetime only)

Custom tools are defined in-process with the SDK's `@tool` decorator and served
through a single `create_sdk_mcp_server` ("tools"); each spoke is then locked to
just its one tool via `AgentDefinition.tools`.

Prerequisites (this is a DIFFERENT dependency from the rest of the repo):
  - pip install claude-agent-sdk
  - The Claude Code CLI must be installed (Node 18+):  npm install -g @anthropic-ai/claude-code
  - ANTHROPIC_API_KEY in the environment (loaded from .env below)

Run it:
  python agent_hub_sdk.py "your question"
  python agent_hub_sdk.py                 # runs the built-in demo query
"""

from __future__ import annotations

import ast
import asyncio
import operator
import sys
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    HookContext,
    HookMatcher,
    ResultMessage,
    TextBlock,
    create_sdk_mcp_server,
    query,
    tool,
)

load_dotenv()  # ANTHROPIC_API_KEY → environment, consumed by the spawned Claude CLI

# Alias the SDK resolves to the latest Haiku; matches the repo's claude-haiku-4-5 choice.
MODEL = "haiku"

# --- Custom in-process tools (the real work the spokes do) ------------------
# Same three tools as agent_hub.py, but written as SDK @tool coroutines. Each
# returns the SDK's content-block shape: {"content": [{"type": "text", ...}]}.

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


def _text(s: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": s}]}


@tool("calculator", "Evaluate an arithmetic expression (+ - * / ** %, parens, decimals).",
      {"expression": str})
async def calculator(args: dict[str, Any]) -> dict[str, Any]:
    return _text(str(_eval_node(ast.parse(args["expression"], mode="eval").body)))


@tool("current_datetime", "Current date/time. Pass an IANA timezone like 'Asia/Kolkata'; defaults to UTC.",
      {"timezone_name": str})
async def current_datetime(args: dict[str, Any]) -> dict[str, Any]:
    name = args.get("timezone_name") or "UTC"
    tz = timezone.utc if name.upper() == "UTC" else ZoneInfo(name)
    return _text(datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z"))


@tool("text_stats", "Count words, characters, and sentences in some text.", {"text": str})
async def text_stats(args: dict[str, Any]) -> dict[str, Any]:
    text = args["text"]
    words = len(text.split())
    chars = len(text)
    sentences = sum(text.count(c) for c in ".!?") or (1 if text.strip() else 0)
    return _text(f"words={words}, characters={chars}, sentences={sentences}")


# One in-process MCP server holds all three; per-spoke `tools` lists do the locking.
TOOLS_SERVER = create_sdk_mcp_server(name="tools", version="1.0.0",
                                     tools=[calculator, current_datetime, text_stats])

# SDK tool-name format is mcp__<server>__<tool>; "tools" is the server name above.
CALC = "mcp__tools__calculator"
TIME = "mcp__tools__current_datetime"
STATS = "mcp__tools__text_stats"

# --- Spokes as declarative subagents ----------------------------------------
# `description` is what the orchestrator reads to decide which spoke to delegate to
# (the analog of agent_hub.py's delegate_desc). `prompt` is the spoke's system
# prompt (the analog of its `system`). `tools` restricts the spoke to its one tool.

SPOKES = {
    "math": AgentDefinition(
        description="Delegate math/computation subtasks here. Pass a self-contained "
        "natural-language task; returns the computed answer.",
        prompt="You are a meticulous math specialist. Use the calculator tool for every "
        "computation — never do arithmetic in your head. Return just the answer.",
        tools=[CALC],
        model=MODEL,
    ),
    "text": AgentDefinition(
        description="Delegate text-analysis subtasks (word/char/sentence counts). Pass "
        "the text and what to measure.",
        prompt="You are a text-analysis specialist. Use the text_stats tool to measure "
        "any text the user mentions. Report the figures plainly.",
        tools=[STATS],
        model=MODEL,
    ),
    "time": AgentDefinition(
        description="Delegate date/time lookups here. State which timezone is wanted.",
        prompt="You are a timekeeping specialist. Use the current_datetime tool to answer "
        "any 'what time is it' question for the requested timezone.",
        tools=[TIME],
        model=MODEL,
    ),
}

ORCH_SYSTEM = (
    "You are an orchestrator. Do not use any tool directly. Break the user's request into "
    "subtasks and delegate each, via the Task tool, to the most appropriate subagent "
    "(math, text, or time). Delegate independent subtasks before combining their results "
    "into one final answer."
)


# --- Subagent lifecycle hooks -----------------------------------------------
# These two events fire from the SDK's built-in **Task** tool — the same tool the
# orchestrator uses to delegate. There is no separate code path to trigger them:
# every `Task → <spoke>` delegation fires SubagentStart when the spoke is spawned
# and SubagentStop when it returns. So the demo query (which fans out to math,
# text, AND time) triggers each hook three times. The callbacks below just observe
# — they print to STDERR so the run's stdout answer stays clean — and return {}
# (no `decision`/`hookSpecificOutput`, so they neither block nor inject context).
#
# Hook callback signature (claude_agent_sdk.HookCallback):
#   async (input_data, tool_use_id, context) -> HookJSONOutput
# input_data is the event's TypedDict (a plain dict at runtime). For these two:
#   SubagentStart: hook_event_name, agent_id, agent_type, session_id, ...
#   SubagentStop:  + stop_hook_active, agent_transcript_path


async def on_subagent_start(
    input_data: dict[str, Any], tool_use_id: str | None, context: HookContext
) -> dict[str, Any]:
    """Fires when the orchestrator's Task tool spawns a spoke subagent."""
    print(
        f"\033[2m[hook] SubagentStart  → spawned '{input_data.get('agent_type')}' "
        f"(agent_id={input_data.get('agent_id')})\033[0m",
        file=sys.stderr,
    )
    return {}  # observe only — no decision, no injected context


async def on_subagent_stop(
    input_data: dict[str, Any], tool_use_id: str | None, context: HookContext
) -> dict[str, Any]:
    """Fires when a spoke subagent finishes and control returns to the hub."""
    print(
        f"\033[2m[hook] SubagentStop   ← finished '{input_data.get('agent_type')}' "
        f"(agent_id={input_data.get('agent_id')})\033[0m",
        file=sys.stderr,
    )
    return {}


# --- Hub (orchestrator) -----------------------------------------------------


async def run_orchestrator(prompt: str) -> str:
    """Run the SDK's main agent with the three spokes registered as subagents.

    The SDK drives the whole loop: the orchestrator delegates via Task, each spoke
    runs as its own agent with only its tool, results flow back, repeat. We just
    register the pieces and read the final answer off the ResultMessage.
    """
    options = ClaudeAgentOptions(
        agents=SPOKES,
        mcp_servers={"tools": TOOLS_SERVER},
        # Enable the spoke tools in the session + let the hub spawn subagents.
        allowed_tools=[CALC, TIME, STATS, "Task"],
        system_prompt=ORCH_SYSTEM,
        model=MODEL,
        # Observe the Task-driven spoke lifecycle. A bare HookMatcher (no `matcher`)
        # runs on every occurrence of the event; SubagentStart/Stop carry no tool
        # name to match on anyway. query() plumbs these through even for a plain
        # string prompt (the SDK always runs in streaming mode internally).
        hooks={
            "SubagentStart": [HookMatcher(hooks=[on_subagent_start])],
            "SubagentStop": [HookMatcher(hooks=[on_subagent_stop])],
        },
    )

    final = ""
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            # Live view of the orchestrator's own narration as it delegates.
            for block in message.content:
                if isinstance(block, TextBlock):
                    final = block.text
        elif isinstance(message, ResultMessage):
            # The authoritative final answer once the whole run completes.
            if message.result:
                final = message.result
    return final.strip()


# The default query, shown pre-filled in the prompt so the user can accept it
# (just press Enter) or edit it inline before submitting.
DEFAULT_QUERY = (
    "Compute 12.5% of 840 divided by 3, tell me the current time in Asia/Kolkata, "
    "and count the words in 'The quick brown fox jumps.' — all in one answer."
)


def _input_with_prefill(prompt: str, text: str) -> str:
    """Like input(), but starts with `text` already in the editable line buffer."""
    import readline  # POSIX-only; present on macOS/Linux where this script runs

    def _hook() -> None:
        readline.insert_text(text)
        readline.redisplay()

    readline.set_pre_input_hook(_hook)
    try:
        return input(prompt)
    finally:
        readline.set_pre_input_hook()  # clear it so later input() calls are clean


def main() -> None:
    # One-shot mode: a query on the command line bypasses the prompt entirely.
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # Interactive: show the default query pre-filled and editable.
        query = _input_with_prefill("\033[1mQuery:\033[0m ", DEFAULT_QUERY).strip()
        if not query:
            print("No query entered — nothing to do.")
            return

    print(f"\n\033[1mQ:\033[0m {query}")
    print(f"\033[1mA:\033[0m {asyncio.run(run_orchestrator(query))}")


if __name__ == "__main__":
    main()
