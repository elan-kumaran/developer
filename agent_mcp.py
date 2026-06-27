"""Same functionality as agent.py, but the tools are served over MCP.

This one file is both halves of the system:

  - `python agent_mcp.py --server`   runs an MCP server (stdio) exposing the tools
  - `python agent_mcp.py "..."`      runs the agent: it spawns the server above,
                                     discovers its tools over MCP, and lets Claude
                                     call them via the SDK tool runner
  - `python agent_mcp.py`            runs the built-in demo queries

Requires Python 3.10+ and `anthropic[mcp]`. Run it with the dedicated venv:
    .venv-mcp/bin/python agent_mcp.py "your question"

The project's default .venv is Python 3.9 and cannot run MCP — see CLAUDE.md.
"""

from __future__ import annotations

import ast
import asyncio
import operator
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()  # loads ANTHROPIC_API_KEY from .env (used by the client half)

MODEL = "claude-haiku-4-5"

# --- Tool implementations (registered on the MCP server below) --------------

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
    """Evaluate an arithmetic expression and return the numeric result.

    Call this whenever the user asks to compute, calculate, or solve a math
    expression — supports + - * / ** %, parentheses, and decimals.
    """
    return str(_eval_node(ast.parse(expression, mode="eval").body))


def current_datetime(timezone_name: str = "UTC") -> str:
    """Get the current date and time.

    Call this when the user asks what time or date it is now. Pass an IANA
    timezone name like 'Asia/Kolkata'; defaults to UTC.
    """
    tz = timezone.utc if timezone_name.upper() == "UTC" else ZoneInfo(timezone_name)
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")


def text_stats(text: str) -> str:
    """Count words, characters, and sentences in a piece of text.

    Call this when the user asks how many words/characters/sentences some
    text has.
    """
    words = len(text.split())
    chars = len(text)
    sentences = sum(text.count(c) for c in ".!?") or (1 if text.strip() else 0)
    return f"words={words}, characters={chars}, sentences={sentences}"


# --- MCP server half --------------------------------------------------------


def run_server() -> None:
    """Expose the tools over an MCP stdio server. FastMCP derives each tool's
    schema from the type hints and its description from the docstring."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("agent-tools", log_level="WARNING")
    server.tool()(calculator)
    server.tool()(current_datetime)
    server.tool()(text_stats)
    server.run()  # stdio transport


# --- Agent (client) half ----------------------------------------------------


async def run_agent(query: str) -> str:
    """Spawn the MCP server, discover its tools, and run the agent loop."""
    from anthropic import AsyncAnthropic
    from anthropic.lib.tools.mcp import async_mcp_tool
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    client = AsyncAnthropic()
    params = StdioServerParameters(command=sys.executable, args=[__file__, "--server"])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools

            # The tool runner drives the loop: it calls the API, executes any MCP
            # tool the model picks (via the MCP session), feeds results back, and
            # repeats until the model stops calling tools.
            runner = client.beta.messages.tool_runner(
                model=MODEL,
                max_tokens=16000,
                messages=[{"role": "user", "content": query}],
                tools=[async_mcp_tool(t, session) for t in tools],
            )

            final = None
            async for message in runner:
                final = message

    if final is None:
        return ""
    return "".join(b.text for b in final.content if b.type == "text").strip()


def main() -> None:
    if "--server" in sys.argv:
        run_server()
        return

    args = [a for a in sys.argv[1:] if a != "--server"]
    if args:
        queries = [" ".join(args)]
    else:
        queries = [
            "What is 12.5% of 840, then divided by 3?",
            "What is the current time in Asia/Kolkata?",
            "How many words and sentences are in: The quick brown fox. It jumps!",
        ]

    for q in queries:
        print(f"\n\033[1mQ:\033[0m {q}")
        print(f"\033[1mA:\033[0m {asyncio.run(run_agent(q))}")


if __name__ == "__main__":
    main()
