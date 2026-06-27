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


def main() -> None:
    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        queries = [
            "Compute 12.5% of 840 divided by 3, tell me the current time in Asia/Kolkata, "
            "and count the words in 'The quick brown fox jumps.' — all in one answer.",
        ]

    for q in queries:
        print(f"\n\033[1mQ:\033[0m {q}")
        print(f"\033[1mA:\033[0m {asyncio.run(run_orchestrator(q))}")


if __name__ == "__main__":
    main()
