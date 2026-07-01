"""A Claude Agent SDK orchestrator that reaches out in two directions at once:

  1. **A public, remote MCP server** — DeepWiki (https://mcp.deepwiki.com), which
     answers questions about public GitHub repositories. Unlike every other file
     in this repo, the MCP server here is NOT a stdio subprocess we spawn; it is a
     remote **HTTP** MCP endpoint we simply point the SDK at. No auth, no server
     code of our own.

  2. **Another agent built on a DIFFERENT framework, over the A2A protocol** — the
     Analyst agent below is NOT a Claude agent. It is a plain agent built directly
     on the **A2A SDK** (`a2a-sdk`): its own AgentCard, AgentExecutor, and JSON-RPC
     server. The two agents interoperate purely through the open **Agent2Agent
     (A2A)** protocol — the Claude agent is an A2A *client*, the Analyst is an A2A
     *server*. Neither knows the other's internals; they only exchange A2A messages.

        Claude orchestrator (Claude Agent SDK, ORCH_SYSTEM)
          ├── mcp__deepwiki__*      → REMOTE HTTP MCP server (repo knowledge)
          └── ask_analyst_agent     → A2A client bridge ──(A2A/JSON-RPC)──▶ Analyst
                                                                            agent
                                                                        (a2a-sdk,
                                                                         no LLM)

So this file demonstrates two flavours of interop the rest of the repo doesn't:
MCP to a *remote* server, and A2A to a *foreign-framework* agent. The bridge that
joins them is a single Claude SDK `@tool` (`ask_analyst_agent`) whose body is an
A2A client call — that is where "agent-to-agent" actually happens.

The Analyst is deterministic (rule-based text/data analysis) on purpose: it needs
no API key of its own, so the whole demo runs with just ANTHROPIC_API_KEY. Swap
its AgentExecutor for a LangGraph/CrewAI/OpenAI agent and nothing else changes —
that is the point of speaking a shared protocol.

Prerequisites (a superset of agent_hub_sdk.py's):
  - pip install -r requirements-a2a.txt   (claude-agent-sdk + a2a-sdk + uvicorn)
  - The Claude Code CLI (Node 18+):  npm install -g @anthropic-ai/claude-code
  - ANTHROPIC_API_KEY in the environment (loaded from .env below)
  - Network access to https://mcp.deepwiki.com for the remote-MCP half.

Run it:
  python agent_a2a.py "your question"   # spawns the Analyst, then runs the hub
  python agent_a2a.py                   # runs the built-in demo query
  python agent_a2a.py --peer            # run ONLY the Analyst A2A server
                                        # (normally auto-spawned; for debugging)
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import sys
from collections import Counter
from typing import Any

import httpx
import uvicorn
from dotenv import load_dotenv
from starlette.applications import Starlette

# --- Claude Agent SDK (the orchestrator + the A2A-bridge tool) --------------
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    create_sdk_mcp_server,
    query,
    tool,
)

# --- A2A SDK (the "different framework" the Analyst agent is built on) -------
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import (
    get_message_text,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.types.a2a_pb2 import Role, SendMessageRequest, TaskState

load_dotenv()  # ANTHROPIC_API_KEY → environment, consumed by the spawned Claude CLI

# Alias the SDK resolves to the latest Haiku; matches the repo's claude-haiku-4-5 choice.
MODEL = "haiku"

# Where the Analyst A2A agent listens. The orchestrator auto-spawns it (see main).
PEER_HOST, PEER_PORT = "127.0.0.1", 9100
PEER_URL = f"http://{PEER_HOST}:{PEER_PORT}"

# The public, remote MCP server. HTTP transport, no auth. Its tools are exposed to
# the SDK as mcp__deepwiki__<tool>: read_wiki_structure, read_wiki_contents,
# ask_question. (All MCP tools in Claude Code are named mcp__<server>__<tool>,
# remote ones included — the server key below, "deepwiki", is the <server> part.)
DEEPWIKI_MCP: dict[str, Any] = {"type": "http", "url": "https://mcp.deepwiki.com/mcp"}


def _text(s: str) -> dict[str, Any]:
    """Wrap a string in the SDK @tool content-block return shape."""
    return {"content": [{"type": "text", "text": s}]}


# ============================================================================
# PART 1 — The Analyst agent, built on the A2A SDK (a DIFFERENT framework).
# ============================================================================
# No Claude, no LLM: a deterministic text/data analyst. It receives an A2A task
# (natural-language instruction + the text to analyse) and returns figures. The
# a2a-sdk contract is: subclass AgentExecutor, drive a TaskUpdater through the
# WORKING → COMPLETED states, and emit the answer as an artifact. That is a whole
# different agent-authoring model from the Claude SDK's AgentDefinition/@tool.

_WORD_RE = re.compile(r"\b\w+\b")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "is",
    "are", "was", "were", "be", "it", "its", "this", "that", "with", "as", "by",
    "at", "from", "into", "you", "your",
}


def analyze_text(text: str) -> str:
    """The Analyst's one skill: word/char/sentence counts + top keywords."""
    words = _WORD_RE.findall(text)
    sentences = sum(text.count(c) for c in ".!?") or (1 if text.strip() else 0)
    keywords = Counter(
        w.lower() for w in words if w.lower() not in _STOPWORDS and len(w) > 2
    )
    top = ", ".join(f"{w}×{n}" for w, n in keywords.most_common(5)) or "(none)"
    return (
        f"words={len(words)}, characters={len(text)}, sentences={sentences}; "
        f"top keywords: {top}"
    )


class AnalystAgentExecutor(AgentExecutor):
    """Bridges the deterministic analyzer into the A2A server lifecycle."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Get-or-create the A2A task for this request.
        task = context.current_task or new_task_from_user_message(context.message)
        if context.current_task is None:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(
            event_queue=event_queue, task_id=task.id, context_id=task.context_id
        )
        await updater.update_status(
            state=TaskState.TASK_STATE_WORKING,
            message=new_text_message("Analyst agent working…"),
        )

        instruction = get_message_text(context.message)
        result = analyze_text(instruction) if instruction.strip() else "No text supplied."

        # The answer is delivered as an artifact, then the task is marked done.
        await updater.add_artifact(parts=[new_text_part(text=result, media_type="text/plain")])
        await updater.update_status(
            state=TaskState.TASK_STATE_COMPLETED,
            message=new_text_message("Analyst agent finished."),
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("The Analyst agent does not support cancellation.")


def build_peer_app() -> Starlette:
    """Assemble the Analyst's A2A server: agent card routes + JSON-RPC routes."""
    skill = AgentSkill(
        id="analyze_text",
        name="Text & data analysis",
        description="Count words, characters, and sentences in text and surface its "
        "most frequent keywords.",
        tags=["analysis", "text", "statistics"],
        examples=["Count the words in this paragraph", "What are the top keywords here?"],
        input_modes=["text/plain"],
        output_modes=["text/plain"],
    )
    card = AgentCard(
        name="Analyst Agent",
        description="A standalone A2A agent (built on a2a-sdk, no LLM) that performs "
        "deterministic text and data analysis.",
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[
            AgentInterface(protocol_binding="JSONRPC", url=PEER_URL),
        ],
        skills=[skill],
    )
    handler = DefaultRequestHandler(
        agent_executor=AnalystAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    # create_agent_card_routes serves /.well-known/agent-card.json; the JSON-RPC
    # routes at "/" handle message/send. A plain Starlette app ties them together.
    routes = [*create_agent_card_routes(card), *create_jsonrpc_routes(handler, "/")]
    return Starlette(routes=routes)


def run_peer() -> None:
    """--peer mode: run only the Analyst A2A server (normally auto-spawned)."""
    print(f"[analyst] A2A server listening on {PEER_URL}", file=sys.stderr)
    uvicorn.run(build_peer_app(), host=PEER_HOST, port=PEER_PORT, log_level="warning")


# ============================================================================
# PART 2 — The A2A client bridge: how the Claude agent talks to the Analyst.
# ============================================================================
# This is the actual "agent-to-agent" hop. It resolves the Analyst's agent card
# from its well-known URL, opens an A2A client, sends the subtask as an A2A text
# message, and stitches the returned artifact/message parts back into a string.


async def ask_peer_over_a2a(task_text: str) -> str:
    """Send `task_text` to the Analyst agent over A2A and return its answer."""
    async with httpx.AsyncClient(timeout=30.0) as hc:
        card = await A2ACardResolver(httpx_client=hc, base_url=PEER_URL).get_agent_card()
        client = await create_client(
            agent=card, client_config=ClientConfig(streaming=False, httpx_client=hc)
        )
        request = SendMessageRequest(message=new_text_message(task_text, role=Role.ROLE_USER))

        parts: list[str] = []
        # send_message yields StreamResponse objects; each is a oneof over
        # task / message / status_update / artifact_update. We collect any text.
        async for resp in client.send_message(request):
            payload = resp.WhichOneof("payload")
            if payload == "artifact_update":
                parts += [p.text for p in resp.artifact_update.artifact.parts if p.text]
            elif payload == "task":
                parts += [p.text for a in resp.task.artifacts for p in a.parts if p.text]
            elif payload == "message":
                parts += [p.text for p in resp.message.parts if p.text]
        await client.close()
        return "\n".join(parts).strip() or "(the Analyst agent returned no text)"


# The Claude SDK @tool the orchestrator calls. Its body IS the A2A client hop, so
# from the orchestrator's point of view "delegating to the Analyst" is one tool call.
@tool(
    "ask_analyst_agent",
    "Delegate a text/data-analysis subtask to the external Analyst agent, a separate "
    "agent that communicates over the A2A protocol. Pass a self-contained instruction "
    "that INCLUDES the full text to analyse; returns the Analyst's findings.",
    {"task": str},
)
async def ask_analyst_agent(args: dict[str, Any]) -> dict[str, Any]:
    print(f"\033[36m[a2a] → Analyst  task={args['task'][:80]!r}\033[0m", file=sys.stderr)
    answer = await ask_peer_over_a2a(args["task"])
    print(f"\033[36m[a2a] ← Analyst  answer={answer[:80]!r}\033[0m", file=sys.stderr)
    return _text(answer)


# One in-process MCP server exposes the A2A bridge tool to the orchestrator.
BRIDGE_SERVER = create_sdk_mcp_server(name="bridge", version="1.0.0", tools=[ask_analyst_agent])
ASK_ANALYST = "mcp__bridge__ask_analyst_agent"

# DeepWiki's remote tools, named the Claude Code way: mcp__<server>__<tool>.
DW_STRUCTURE = "mcp__deepwiki__read_wiki_structure"
DW_CONTENTS = "mcp__deepwiki__read_wiki_contents"
DW_ASK = "mcp__deepwiki__ask_question"

ORCH_SYSTEM = (
    "You are an orchestrator coordinating two external collaborators:\n"
    "  1. DeepWiki — a remote MCP server that answers questions about PUBLIC GitHub "
    "repositories. Use its tools (ask_question / read_wiki_structure / read_wiki_contents) "
    "to gather facts or summaries about a repo the user names.\n"
    "  2. The Analyst agent — reachable ONLY through the ask_analyst_agent tool (it is a "
    "separate agent speaking the A2A protocol). It performs text/data analysis such as "
    "word, character, and sentence counts and keyword frequencies.\n"
    "Do the analysis-flavoured work by delegating to the Analyst — do NOT count words "
    "yourself. When you delegate, pass the Analyst a self-contained instruction that "
    "includes the exact text to analyse. Combine the repo facts and the Analyst's figures "
    "into one clear final answer."
)


# ============================================================================
# PART 3 — The hub: run the Claude orchestrator with both collaborators wired in.
# ============================================================================


async def run_orchestrator(prompt: str) -> str:
    """Run the Claude Agent SDK main agent with the remote MCP + A2A bridge tools."""
    options = ClaudeAgentOptions(
        mcp_servers={"bridge": BRIDGE_SERVER, "deepwiki": DEEPWIKI_MCP},
        allowed_tools=[ASK_ANALYST, DW_STRUCTURE, DW_CONTENTS, DW_ASK],
        system_prompt=ORCH_SYSTEM,
        model=MODEL,
    )

    final = ""
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    final = block.text
        elif isinstance(message, ResultMessage):
            if message.result:
                final = message.result
    return final.strip()


async def _wait_for_peer(timeout: float = 20.0) -> bool:
    """Poll the Analyst's agent-card endpoint until it is serving (or time out)."""
    url = f"{PEER_URL}/.well-known/agent-card.json"
    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient(timeout=2.0) as hc:
        while asyncio.get_event_loop().time() < deadline:
            try:
                if (await hc.get(url)).status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.3)
    return False


async def run_with_peer(prompt: str) -> str:
    """Spawn the Analyst A2A server as a subprocess, run the hub, then tear it down."""
    peer = subprocess.Popen([sys.executable, __file__, "--peer"])
    try:
        if not await _wait_for_peer():
            return "The Analyst A2A agent did not start in time — aborting."
        print(f"\033[2m[hub] Analyst agent ready at {PEER_URL}\033[0m", file=sys.stderr)
        return await run_orchestrator(prompt)
    finally:
        peer.terminate()
        try:
            peer.wait(timeout=5)
        except subprocess.TimeoutExpired:
            peer.kill()


# The default query exercises BOTH paths: DeepWiki (remote MCP) fetches a repo
# summary, then the Analyst (A2A) measures that summary.
DEFAULT_QUERY = (
    "Using DeepWiki, briefly summarize what the 'modelcontextprotocol/python-sdk' "
    "GitHub repository is for, then have the Analyst agent report the word, sentence, "
    "and keyword statistics of that summary. Give me both the summary and the figures."
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
    # --peer: run only the Analyst A2A server (this is what run_with_peer spawns).
    if len(sys.argv) > 1 and sys.argv[1] == "--peer":
        run_peer()
        return

    # One-shot mode: a query on the command line bypasses the prompt entirely.
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = _input_with_prefill("\033[1mQuery:\033[0m ", DEFAULT_QUERY).strip()
        if not user_query:
            print("No query entered — nothing to do.")
            return

    print(f"\n\033[1mQ:\033[0m {user_query}")
    print(f"\033[1mA:\033[0m {asyncio.run(run_with_peer(user_query))}")


if __name__ == "__main__":
    main()
