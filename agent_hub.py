"""A *true* hub-and-spoke agentic system, built on the same MCP plumbing as agent_mcp.py.

The difference from agent_mcp.py: there the spokes are plain *functions* and only
Claude reasons. Here each spoke is its own **Claude agent** — its own system prompt,
its own reasoning loop, its own real tools — exposed over MCP as a single delegation
tool `ask_<spoke>_agent(task)`. The **hub** is an orchestrator Claude loop whose only
"tools" are those sub-agents; it decomposes the request and delegates.

    Hub (orchestrator Claude)
      ├── ask_math_agent   → math spoke   (Claude loop + calculator)
      ├── ask_text_agent   → text spoke   (Claude loop + text_stats)
      └── ask_time_agent   → time spoke   (Claude loop + current_datetime)

Modes (one file plays every role, like agent_mcp.py):
  - python agent_hub.py "..."          run the orchestrator (spawns every spoke server)
  - python agent_hub.py --server math  run one spoke as an MCP stdio server (auto-spawned)
  - python agent_hub.py                run the built-in demo queries

Requires Python 3.10+ and `anthropic[mcp]`. Run with the MCP venv (see CLAUDE.md).
"""

from __future__ import annotations

import ast
import asyncio
import operator
import sys
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 16000

# --- Tool implementations (the real work the spokes do) ---------------------

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


def calculator(expression: str) -> str:
    return str(_eval_node(ast.parse(expression, mode="eval").body))


def current_datetime(timezone_name: str = "UTC") -> str:
    tz = timezone.utc if timezone_name.upper() == "UTC" else ZoneInfo(timezone_name)
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")


def text_stats(text: str) -> str:
    words = len(text.split())
    chars = len(text)
    sentences = sum(text.count(c) for c in ".!?") or (1 if text.strip() else 0)
    return f"words={words}, characters={chars}, sentences={sentences}"


# A tool's API schema lives next to its implementation so a spoke can hand it to Claude.
TOOLSPECS = {
    "calculator": {
        "fn": calculator,
        "schema": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression (+ - * / ** %, parens, decimals).",
            "input_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    "current_datetime": {
        "fn": current_datetime,
        "schema": {
            "name": "current_datetime",
            "description": "Current date/time. Pass an IANA timezone like 'Asia/Kolkata'; defaults to UTC.",
            "input_schema": {
                "type": "object",
                "properties": {"timezone_name": {"type": "string"}},
            },
        },
    },
    "text_stats": {
        "fn": text_stats,
        "schema": {
            "name": "text_stats",
            "description": "Count words, characters, and sentences in some text.",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
}

# --- Spoke definitions: each is an agent (system prompt + which tools it owns) ----

SPOKES = {
    "math": {
        "system": "You are a meticulous math specialist. Use the calculator tool for "
        "every computation — never do arithmetic in your head. Return just the answer.",
        "tools": ["calculator"],
        "delegate_desc": "Delegate a math/computation subtask to the math specialist agent. "
        "Pass a self-contained natural-language task; it returns the computed answer.",
    },
    "text": {
        "system": "You are a text-analysis specialist. Use the text_stats tool to measure "
        "any text the user mentions. Report the figures plainly.",
        "tools": ["text_stats"],
        "delegate_desc": "Delegate a text-analysis subtask (word/char/sentence counts) to the "
        "text specialist agent. Pass the text and what to measure.",
    },
    "time": {
        "system": "You are a timekeeping specialist. Use the current_datetime tool to answer "
        "any 'what time is it' question for the requested timezone.",
        "tools": ["current_datetime"],
        "delegate_desc": "Delegate a date/time lookup to the time specialist agent. "
        "State which timezone is wanted.",
    },
}

ORCH_SYSTEM = (
    "You are an orchestrator. You have no tools of your own except specialist sub-agents. "
    "Break the user's request into subtasks and delegate each to the most appropriate "
    "ask_<spoke>_agent tool, then synthesize their results into one final answer. "
    "Delegate independent subtasks before combining them."
)


# --- Spoke (sub-agent) half: a self-contained Claude loop behind one MCP tool ----


async def _subagent_loop(spoke_name: str, task: str) -> str:
    """Run the spoke's own agentic loop: Claude + that spoke's real tools."""
    from anthropic import AsyncAnthropic

    spoke = SPOKES[spoke_name]
    fns = {name: TOOLSPECS[name]["fn"] for name in spoke["tools"]}
    tools = [TOOLSPECS[name]["schema"] for name in spoke["tools"]]

    client = AsyncAnthropic()
    messages = [{"role": "user", "content": task}]

    for _ in range(8):  # cap turns, same guard rail as agent.py
        resp = await client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=spoke["system"],
            tools=tools,
            messages=messages,
        )
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text").strip()

        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            try:
                out = fns[block.name](**block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
            except Exception as exc:  # a tool error becomes recoverable model context
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(exc),
                        "is_error": True,
                    }
                )
        messages.append({"role": "user", "content": results})

    return "Sub-agent reached its turn limit without finishing."


def run_server(spoke_name: str) -> None:
    """Expose one spoke as an MCP server with a single delegation tool."""
    from mcp.server.fastmcp import FastMCP

    spoke = SPOKES[spoke_name]
    server = FastMCP(f"agent-{spoke_name}", log_level="WARNING")

    @server.tool(name=f"ask_{spoke_name}_agent", description=spoke["delegate_desc"])
    async def delegate(task: str) -> str:  # noqa: D401 - description set above
        return await _subagent_loop(spoke_name, task)

    server.run()  # stdio transport


# --- Hub (orchestrator) half ------------------------------------------------


async def run_orchestrator(query: str) -> str:
    """Spawn every spoke server, expose their delegation tools to the hub Claude loop."""
    from anthropic import AsyncAnthropic
    from anthropic.lib.tools.mcp import async_mcp_tool
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    client = AsyncAnthropic()

    # One MCP session per spoke; AsyncExitStack keeps them all open for the run.
    async with AsyncExitStack() as stack:
        tools = []
        for name in SPOKES:
            params = StdioServerParameters(command=sys.executable, args=[__file__, "--server", name])
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            for t in (await session.list_tools()).tools:
                tools.append(async_mcp_tool(t, session))

        runner = client.beta.messages.tool_runner(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=ORCH_SYSTEM,
            messages=[{"role": "user", "content": query}],
            tools=tools,
        )

        final = None
        async for message in runner:
            final = message

    if final is None:
        return ""
    return "".join(b.text for b in final.content if b.type == "text").strip()


def main() -> None:
    if "--server" in sys.argv:
        i = sys.argv.index("--server")
        spoke_name = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
        if spoke_name not in SPOKES:
            sys.exit(f"unknown spoke {spoke_name!r}; choose from {list(SPOKES)}")
        run_server(spoke_name)
        return

    args = [a for a in sys.argv[1:] if a != "--server"]
    if args:
        queries = [" ".join(args)]
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
