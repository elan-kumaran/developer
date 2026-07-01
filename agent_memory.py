"""A Claude agent that demonstrates the SIX types of memory used in agentic
architectures — each as a real, separate, inspectable implementation wired into
one working agentic loop (raw `anthropic` SDK + a manual tool loop, in the style
of agent.py).

The taxonomy (the cognitive-architecture framing most agent literature uses):

  VOLATILE (reset every session — they live only in RAM):
    1. Sensory memory      — a raw input buffer that holds the just-arrived,
                             unprocessed input for a moment before it is encoded
                             into working memory. Decays almost immediately (TTL).
    2. Short-term / Working memory — the active scratchpad for the CURRENT task:
                             a sliding window of recent conversation turns plus
                             intermediate notes. This is what becomes the `messages`
                             list and drives the context window.

  DURABLE (persisted to disk — they survive across sessions/process restarts):
    3. Long-term memory    — the persistence substrate itself. Not a fourth "kind"
                             of content so much as the durable store that BACKS the
                             next three. Implemented here as the facade that loads/
                             saves episodic + semantic + procedural to a directory.
    4. Episodic memory     — a log of past experiences: timestamped (input, response)
                             episodes, retrievable by recency or relevance.
    5. Semantic memory     — durable facts/knowledge (about the user, the world),
                             stored as key→value and injected as "known facts".
    6. Procedural memory   — learned skills/rules: standing behavioral guidelines the
                             agent has acquired, injected into the system prompt so
                             they shape HOW it acts (the analog of a learned skill).

How they interact on every turn:
    raw input ─▶ SENSORY (buffer) ─▶ WORKING (window) ─▶ Claude
                                                          │
    system prompt = base + PROCEDURAL rules + SEMANTIC facts + recalled EPISODES
                                                          │
    Claude may call tools to WRITE memory: remember_fact (semantic),
    learn_rule (procedural), recall_memory (read episodic+semantic on demand).
                                                          │
    after the turn ─▶ the (input, response) is recorded as an EPISODE, and all
    durable stores are saved to disk (LONG-TERM persistence).

The practical proof of long-term memory is the built-in demo: it runs TWO
sessions. Session 1 teaches the agent facts + a rule. Session 2 spins up a
BRAND-NEW agent object (empty sensory + working memory) pointed at the same
memory directory — and it still knows who you are, because episodic/semantic/
procedural were reloaded from disk.

Run it:
  python agent_memory.py                 # the two-session demo (best first run)
  python agent_memory.py "your message"  # one turn against the persisted memory
  python agent_memory.py --inspect       # print what's currently in long-term memory
  python agent_memory.py --forget        # wipe the long-term memory directory

Needs ANTHROPIC_API_KEY in .env (same as the other anthropic-based scripts);
no extra dependencies beyond requirements.txt.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()  # ANTHROPIC_API_KEY → environment

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5"

# Where the durable (long-term) memories live between runs.
MEMORY_DIR = Path(__file__).with_name(".agent_memory")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "is",
    "are", "was", "were", "be", "it", "its", "this", "that", "with", "as", "by",
    "at", "from", "into", "you", "your", "i", "me", "my", "do", "you're", "what",
    "how", "who", "when", "please", "can", "could", "would", "about",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _keywords(text: str) -> set[str]:
    """Lowercased content words — the crude 'embedding' used for relevance recall."""
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split()
            if w not in _STOPWORDS and len(w) > 2}


# ============================================================================
# 1. SENSORY MEMORY — a transient raw-input buffer that decays almost at once.
# ============================================================================
@dataclass
class SensoryMemory:
    """Holds the most recent raw input for `ttl_seconds`, then forgets it.

    In a perception→cognition pipeline this is the buffer that briefly retains
    unprocessed input until it is 'attended to' and encoded into working memory.
    Here we perceive() the raw user text, read() it once to promote it, then it
    is gone — nothing sensory is ever persisted.
    """

    ttl_seconds: float = 3.0
    _slot: tuple[float, str] | None = None

    def perceive(self, raw: str) -> None:
        self._slot = (time.monotonic(), raw)

    def read(self) -> str | None:
        if self._slot is None:
            return None
        stored_at, value = self._slot
        if time.monotonic() - stored_at > self.ttl_seconds:
            self._slot = None  # decayed
            return None
        return value

    def clear(self) -> None:
        self._slot = None

    def snapshot(self) -> str:
        val = self.read()
        return f"buffer={val!r}" if val else "buffer=(empty/decayed)"


# ============================================================================
# 2. SHORT-TERM / WORKING MEMORY — sliding window of turns + task scratchpad.
# ============================================================================
@dataclass
class WorkingMemory:
    """The active context for the current task.

    `turns` is a bounded sliding window of clean conversation turns (what we send
    to the model as `messages`). `scratchpad` holds intermediate notes made WHILE
    solving the current task (e.g. 'called recall_memory'); it is wiped at the
    start of each new task, modelling working memory's task-scoped volatility.
    """

    max_turns: int = 8
    turns: deque[dict[str, str]] = field(default_factory=lambda: deque(maxlen=8))
    scratchpad: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.turns = deque(self.turns, maxlen=self.max_turns)

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})

    def note(self, thought: str) -> None:
        self.scratchpad.append(thought)

    def start_task(self) -> None:
        self.scratchpad.clear()

    def as_messages(self) -> list[dict[str, str]]:
        return list(self.turns)

    def snapshot(self) -> str:
        return (f"{len(self.turns)}/{self.max_turns} turns in window; "
                f"scratchpad={self.scratchpad or '[]'}")


# ============================================================================
# 4. EPISODIC MEMORY — a durable, retrievable log of past experiences.
# ============================================================================
@dataclass
class Episode:
    timestamp: str
    user_input: str
    agent_response: str

    def to_dict(self) -> dict[str, str]:
        return {"timestamp": self.timestamp, "user_input": self.user_input,
                "agent_response": self.agent_response}


class EpisodicMemory:
    """Append-only episodes, recall-able by recency or keyword relevance."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.episodes: list[Episode] = []

    def record(self, user_input: str, agent_response: str) -> None:
        self.episodes.append(Episode(_now(), user_input, agent_response))

    def recall(self, query: str, k: int = 2) -> list[Episode]:
        """Return the k episodes whose content best overlaps the query."""
        q = _keywords(query)
        if not q:
            return self.recent(k)
        scored = ((len(q & _keywords(e.user_input + " " + e.agent_response)), e)
                  for e in self.episodes)
        hits = sorted((s for s in scored if s[0] > 0), key=lambda s: s[0], reverse=True)
        return [e for _, e in hits[:k]]

    def recent(self, k: int = 2) -> list[Episode]:
        return self.episodes[-k:]

    def load(self) -> None:
        if self.path.exists():
            self.episodes = [Episode(**d) for d in json.loads(self.path.read_text())]

    def save(self) -> None:
        self.path.write_text(json.dumps([e.to_dict() for e in self.episodes], indent=2))

    def snapshot(self) -> str:
        return f"{len(self.episodes)} episode(s) recorded"


# ============================================================================
# 5. SEMANTIC MEMORY — durable facts/knowledge as key→value.
# ============================================================================
class SemanticMemory:
    """What the agent KNOWS: stable facts about the user/world."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.facts: dict[str, str] = {}

    def upsert(self, key: str, value: str) -> None:
        self.facts[key.strip()] = value.strip()

    def as_prompt_block(self) -> str:
        if not self.facts:
            return ""
        lines = "\n".join(f"  - {k}: {v}" for k, v in self.facts.items())
        return f"Known facts (semantic memory):\n{lines}"

    def load(self) -> None:
        if self.path.exists():
            self.facts = json.loads(self.path.read_text())

    def save(self) -> None:
        self.path.write_text(json.dumps(self.facts, indent=2))

    def snapshot(self) -> str:
        return f"{len(self.facts)} fact(s): {list(self.facts.keys()) or '[]'}"


# ============================================================================
# 6. PROCEDURAL MEMORY — learned skills/rules that shape HOW the agent acts.
# ============================================================================
class ProceduralMemory:
    """Standing behavioral rules the agent has learned; injected into the system
    prompt so they change its behavior on every subsequent turn."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rules: list[str] = []

    def learn(self, rule: str) -> None:
        rule = rule.strip()
        if rule and rule not in self.rules:
            self.rules.append(rule)

    def as_prompt_block(self) -> str:
        if not self.rules:
            return ""
        lines = "\n".join(f"  - {r}" for r in self.rules)
        return f"Operating rules you have learned (procedural memory):\n{lines}"

    def load(self) -> None:
        if self.path.exists():
            self.rules = json.loads(self.path.read_text())

    def save(self) -> None:
        self.path.write_text(json.dumps(self.rules, indent=2))

    def snapshot(self) -> str:
        return f"{len(self.rules)} rule(s) learned"


# ============================================================================
# 3. LONG-TERM MEMORY — the durable substrate that backs #4, #5, #6.
# ============================================================================
class LongTermMemory:
    """Facade over the three persistent stores. 'Long-term memory' in an agent is
    not a separate content type — it is the durable layer that episodic, semantic,
    and procedural memory are written to and reloaded from. This class owns the
    directory and the load_all/save_all lifecycle that makes memory outlive a run.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.episodic = EpisodicMemory(directory / "episodic.json")
        self.semantic = SemanticMemory(directory / "semantic.json")
        self.procedural = ProceduralMemory(directory / "procedural.json")
        self.load_all()

    def load_all(self) -> None:
        self.episodic.load()
        self.semantic.load()
        self.procedural.load()

    def save_all(self) -> None:
        self.episodic.save()
        self.semantic.save()
        self.procedural.save()

    def snapshot(self) -> str:
        return f"persisted at {self.directory.name}/ (episodic+semantic+procedural)"


# ============================================================================
# Memory-writing tools the model can call (client-side, agent.py-style schemas).
# ============================================================================
TOOL_SCHEMAS = [
    {
        "name": "remember_fact",
        "description": "Store a durable fact in SEMANTIC memory when the user shares "
        "stable information about themselves or the world (name, role, preferences, "
        "project details). Use a short snake_case key.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Short identifier, e.g. 'user_name'."},
                "value": {"type": "string", "description": "The fact to remember."},
            },
            "required": ["key", "value"],
            "additionalProperties": False,
        },
    },
    {
        "name": "learn_rule",
        "description": "Store a standing behavioral instruction in PROCEDURAL memory when "
        "the user tells you HOW to behave from now on (formatting, tone, workflow). Phrase "
        "it as an imperative rule.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rule": {"type": "string", "description": "The rule, e.g. 'Always include a runnable example with code.'"},
            },
            "required": ["rule"],
            "additionalProperties": False,
        },
    },
    {
        "name": "recall_memory",
        "description": "Search your EPISODIC and SEMANTIC memory for anything relevant to a "
        "query. Use when you need past context that isn't already in the conversation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look up."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]

BASE_SYSTEM = (
    "You are a helpful personal assistant with a persistent memory. "
    "You have three memory tools: remember_fact (save durable facts to semantic memory), "
    "learn_rule (save standing behavioral instructions to procedural memory), and "
    "recall_memory (search your past episodes and facts). Proactively call remember_fact "
    "when the user shares stable personal information, and learn_rule when the user tells "
    "you how to behave going forward. Follow every learned rule below, and use the known "
    "facts to personalize your answers."
)


# ============================================================================
# The agent: ties all six memories into one manual agentic loop.
# ============================================================================
class MemoryAgent:
    def __init__(self, memory_dir: Path = MEMORY_DIR, verbose: bool = True) -> None:
        self.verbose = verbose
        # Volatile tiers: fresh every time an agent object is created (i.e. per session).
        self.sensory = SensoryMemory()
        self.working = WorkingMemory()
        # Durable tier: reloaded from disk, so it carries over between sessions.
        self.ltm = LongTermMemory(memory_dir)

    # -- prompt assembly: this is where memory becomes context ---------------
    def _system_prompt(self, current_input: str) -> str:
        blocks = [BASE_SYSTEM]
        if rules := self.ltm.procedural.as_prompt_block():
            blocks.append(rules)
        if facts := self.ltm.semantic.as_prompt_block():
            blocks.append(facts)
        # Auto-recall episodes relevant to the current input (episodic retrieval).
        relevant = self.ltm.episodic.recall(current_input, k=2)
        if relevant:
            recalled = "\n".join(
                f"  - [{e.timestamp}] you said: {e.user_input!r} → you answered: "
                f"{e.agent_response[:120]!r}" for e in relevant)
            blocks.append(f"Relevant past interactions (episodic memory):\n{recalled}")
        return "\n\n".join(blocks)

    # -- tool dispatch: the model writing to / reading from memory -----------
    def _run_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        self.working.note(f"tool:{name}({tool_input})")
        if name == "remember_fact":
            self.ltm.semantic.upsert(tool_input["key"], tool_input["value"])
            return f"Stored semantic fact {tool_input['key']!r}."
        if name == "learn_rule":
            self.ltm.procedural.learn(tool_input["rule"])
            return f"Learned procedural rule: {tool_input['rule']!r}."
        if name == "recall_memory":
            q = tool_input["query"]
            eps = self.ltm.episodic.recall(q, k=3)
            facts = self.ltm.semantic.as_prompt_block() or "(no facts)"
            ep_txt = "\n".join(f"- {e.user_input} → {e.agent_response[:120]}" for e in eps) or "(no episodes)"
            return f"Episodes:\n{ep_txt}\n{facts}"
        return f"Error: unknown tool {name}"

    def turn(self, user_input: str, max_turns: int = 6) -> str:
        """Process one user message through all six memory systems."""
        self.working.start_task()  # wipe the task scratchpad (working memory)

        # 1. SENSORY: buffer the raw input, then read it once to 'attend' to it.
        self.sensory.perceive(user_input)
        attended = self.sensory.read() or user_input

        # 2. WORKING: encode the attended input as a conversation turn.
        self.working.add_turn("user", attended)
        self.sensory.clear()  # promoted into working memory; sensory buffer emptied

        # Build the request: system prompt (procedural+semantic+episodic) + window.
        system = self._system_prompt(attended)
        messages = self.working.as_messages()

        response = None
        for _ in range(max_turns):
            response = client.messages.create(
                model=MODEL, max_tokens=1500, system=system,
                tools=TOOL_SCHEMAS, messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                break
            tool_results = [
                {"type": "tool_result", "tool_use_id": b.id,
                 "content": self._run_tool(b.name, b.input)}
                for b in response.content if b.type == "tool_use"
            ]
            messages.append({"role": "user", "content": tool_results})

        answer = "".join(b.text for b in response.content if b.type == "text").strip()

        # 2. WORKING: record the clean assistant turn in the window.
        self.working.add_turn("assistant", answer)
        # 4. EPISODIC: log this experience.
        self.ltm.episodic.record(user_input, answer)
        # 3. LONG-TERM: persist episodic + semantic + procedural to disk.
        self.ltm.save_all()

        if self.verbose:
            self._print_dashboard()
        return answer

    def _print_dashboard(self) -> None:
        """Show the live state of all six memory systems (to stderr, dimmed)."""
        rows = [
            ("1 sensory   ", self.sensory.snapshot()),
            ("2 working   ", self.working.snapshot()),
            ("3 long-term ", self.ltm.snapshot()),
            ("4 episodic  ", self.ltm.episodic.snapshot()),
            ("5 semantic  ", self.ltm.semantic.snapshot()),
            ("6 procedural", self.ltm.procedural.snapshot()),
        ]
        print("\033[2m  ┌─ memory ─────────────────────────────────────────────", file=sys.stderr)
        for name, snap in rows:
            print(f"\033[2m  │ {name} │ {snap}", file=sys.stderr)
        print("\033[2m  └──────────────────────────────────────────────────────\033[0m", file=sys.stderr)


# ============================================================================
# Entry points: the two-session demo, single-turn, inspect, forget.
# ============================================================================
def _print_qa(user: str, answer: str) -> None:
    print(f"\n\033[1mUser:\033[0m  {user}")
    print(f"\033[1mAgent:\033[0m {answer}")


def run_demo() -> None:
    """Two sessions proving volatile memory resets while long-term memory persists."""
    print("\033[1m═══ SESSION 1 — a fresh agent learns about you ═══\033[0m")
    print("\033[2m(starting from a clean slate — wiping any prior memory dir)\033[0m")
    if MEMORY_DIR.exists():
        shutil.rmtree(MEMORY_DIR)

    agent1 = MemoryAgent()
    for msg in [
        "Hi! I'm Elan, a staff engineer at HCLTech working on Claude agents. "
        "Please remember that about me.",
        "Going forward, whenever you give me code, always include a one-line command "
        "to run it. That's a standing rule.",
        "Give me a Python snippet that reverses a string.",
    ]:
        _print_qa(msg, agent1.turn(msg))

    print("\n\033[1m═══ SESSION 2 — a BRAND-NEW agent, same memory directory ═══\033[0m")
    print("\033[2m(agent2 has empty sensory + working memory, but reloads long-term "
          "memory from disk)\033[0m")
    agent2 = MemoryAgent()  # fresh volatile memory; durable memory reloaded from disk
    for msg in [
        "Do you remember who I am and where I work?",
        "Show me how to read a text file in Python.",  # should auto-apply the learned rule
    ]:
        _print_qa(msg, agent2.turn(msg))

    print("\n\033[2mLong-term memory now lives in "
          f"{MEMORY_DIR}/ — run `python agent_memory.py --inspect` to see it, "
          "or just run the script again and session 2 will already know you.\033[0m")


def inspect_memory() -> None:
    ltm = LongTermMemory(MEMORY_DIR)
    print(f"\033[1mLong-term memory in {MEMORY_DIR}/\033[0m")
    print(f"\n\033[1mSemantic facts:\033[0m")
    for k, v in (ltm.semantic.facts or {"(none)": ""}).items():
        print(f"  - {k}: {v}")
    print(f"\n\033[1mProcedural rules:\033[0m")
    for r in ltm.procedural.rules or ["(none)"]:
        print(f"  - {r}")
    print(f"\n\033[1mEpisodes ({len(ltm.episodic.episodes)}):\033[0m")
    for e in ltm.episodic.episodes:
        print(f"  - [{e.timestamp}] {e.user_input[:70]!r} → {e.agent_response[:70]!r}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--inspect":
        inspect_memory()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--forget":
        if MEMORY_DIR.exists():
            shutil.rmtree(MEMORY_DIR)
        print(f"Wiped {MEMORY_DIR}/ — long-term memory forgotten.")
        return
    if len(sys.argv) > 1:
        # Single turn against whatever is already in long-term memory.
        agent = MemoryAgent()
        _print_qa(sys.argv[1], agent.turn(" ".join(sys.argv[1:])))
        return
    run_demo()


if __name__ == "__main__":
    main()
