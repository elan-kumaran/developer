"""A tiny agentic framework for Claude with a pluggable tool registry.

The model decides which tool (if any) to use for each request. Adding a new
tool is a single @tool(...) decorator — its schema is registered and the agent
loop will route to it automatically.

Run it:
    python agent.py "what is 12.5% of 840, then divided by 3?"
    python agent.py "what time is it right now?"
    python agent.py "how many words are in: the quick brown fox jumps"
    python agent.py                 # runs a few demo queries
"""

from __future__ import annotations

import ast
import operator
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import anthropic

load_dotenv()  # loads ANTHROPIC_API_KEY from the .env file into the environment

client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY from the environment

MODEL = "claude-haiku-4-5"

# --- Tool registry ----------------------------------------------------------

# name -> {"schema": <tool definition dict>, "fn": <callable(**input) -> str>}
REGISTRY: dict[str, dict] = {}


def tool(name: str, description: str, input_schema: dict):
    """Decorator: register a client-side tool with its API schema."""

    def decorator(fn):
        REGISTRY[name] = {
            "schema": {
                "name": name,
                "description": description,
                "strict": True,
                "input_schema": {**input_schema, "additionalProperties": False},
            },
            "fn": fn,
        }
        return fn

    return decorator


def tool_schemas() -> list[dict]:
    return [entry["schema"] for entry in REGISTRY.values()]


def run_tool(name: str, tool_input: dict) -> dict:
    """Execute a registered tool, returning a tool_result content block."""
    try:
        entry = REGISTRY.get(name)
        if entry is None:
            raise ValueError(f"unknown tool: {name}")
        return {"content": str(entry["fn"](**tool_input)), "is_error": False}
    except Exception as e:  # surface errors so the model can recover
        return {"content": f"Error: {e}", "is_error": True}


# --- Tools ------------------------------------------------------------------

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


@tool(
    name="calculator",
    description=(
        "Evaluate an arithmetic expression and return the numeric result. "
        "Call this whenever the user asks to compute, calculate, or solve a "
        "math expression — supports + - * / ** %, parentheses, and decimals."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The arithmetic expression, e.g. '2 + 2 * (3 - 1)'",
            }
        },
        "required": ["expression"],
    },
)
def calculator(expression: str) -> str:
    """Evaluate arithmetic safely (no eval())."""
    return _eval_node(ast.parse(expression, mode="eval").body)


@tool(
    name="current_datetime",
    description=(
        "Get the current date and time. Call this when the user asks what time "
        "or date it is now. Optionally pass an IANA timezone like 'Asia/Kolkata'; "
        "defaults to UTC."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone name, e.g. 'America/New_York'. Defaults to UTC.",
            }
        },
        "required": [],
    },
)
def current_datetime(timezone: str | None = None) -> str:
    tz = ZoneInfo(timezone) if timezone else timezone_utc()
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")


def timezone_utc():
    return timezone.utc


@tool(
    name="text_stats",
    description=(
        "Count words, characters, and sentences in a piece of text. Call this "
        "when the user asks how many words/characters/sentences some text has."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to analyze."}
        },
        "required": ["text"],
    },
)
def text_stats(text: str) -> str:
    words = len(text.split())
    chars = len(text)
    sentences = sum(text.count(c) for c in ".!?") or (1 if text.strip() else 0)
    return f"words={words}, characters={chars}, sentences={sentences}"


# --- Agentic loop -----------------------------------------------------------


def run_agent(query: str, max_turns: int = 10) -> str:
    """Send a query, run the tool loop, and return the final text answer."""
    messages = [{"role": "user", "content": query}]
    response = None

    for _ in range(max_turns):
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            tools=tool_schemas(),
            messages=messages,
        )

        if response.stop_reason == "refusal":
            return "[Claude declined this request for safety reasons.]"

        # Append the assistant turn (preserves tool_use blocks for the next call).
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break  # end_turn — Claude is done

        # Execute every tool call, return all results in one user message.
        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                **run_tool(block.name, block.input),
            }
            for block in response.content
            if block.type == "tool_use"
        ]
        messages.append({"role": "user", "content": tool_results})

    return "".join(b.text for b in response.content if b.type == "text").strip()


def main():
    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        queries = [
            "What is 12.5% of 840, then divided by 3?",
            "What is the current time in Asia/Kolkata?",
            "How many words and sentences are in: The quick brown fox. It jumps!",
        ]

    for q in queries:
        print(f"\n\033[1mQ:\033[0m {q}")
        print(f"\033[1mA:\033[0m {run_agent(q)}")


if __name__ == "__main__":
    main()
