"""Neutral data shapes for the systems under test.

Your project writes a thin adapter that turns its real output (a LangGraph
state, a RAG chain response, an API JSON body) into these types. The
assertions only ever see these shapes, so they work across projects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedDoc:
    id: str
    text: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RagResult:
    answer: str
    retrieved: list[RetrievedDoc] = field(default_factory=list)

    @property
    def retrieved_ids(self) -> list[str]:
        return [d.id for d in self.retrieved]

    @property
    def context(self) -> str:
        return "\n\n".join(f"[{d.id}] {d.text}" for d in self.retrieved)


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    error: str | None = None


@dataclass
class AgentTrace:
    """What an agent did: the ordered tool calls and its final answer."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    final_answer: str = ""
    steps: int | None = None          # LLM turns; defaults to len(tool_calls) + 1
    terminated: bool = True           # False if it hit a recursion/step limit
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_names(self) -> list[str]:
        return [c.name for c in self.tool_calls]

    @property
    def step_count(self) -> int:
        return self.steps if self.steps is not None else len(self.tool_calls) + 1

    @classmethod
    def from_langchain_messages(cls, messages: list[Any], **kwargs: Any) -> "AgentTrace":
        """Build a trace from LangChain/LangGraph messages (duck-typed, no import).

        Works with ``result["messages"]`` from a LangGraph agent: AIMessages with
        ``tool_calls`` become ToolCalls, ToolMessages fill in their outputs, and
        the last AI message without tool calls is the final answer.
        """
        calls: list[ToolCall] = []
        by_id: dict[str, ToolCall] = {}
        final = ""
        ai_turns = 0
        for m in messages:
            mtype = getattr(m, "type", None) or (m.get("type") if isinstance(m, dict) else None)
            if mtype == "ai":
                ai_turns += 1
                tcs = getattr(m, "tool_calls", None) or (m.get("tool_calls") if isinstance(m, dict) else None) or []
                for tc in tcs:
                    call = ToolCall(name=tc["name"], args=tc.get("args", {}))
                    calls.append(call)
                    if tc.get("id"):
                        by_id[tc["id"]] = call
                if not tcs:
                    content = getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")
                    final = content if isinstance(content, str) else str(content)
            elif mtype == "tool":
                tid = getattr(m, "tool_call_id", None) if not isinstance(m, dict) else m.get("tool_call_id")
                content = getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")
                status = getattr(m, "status", None) if not isinstance(m, dict) else m.get("status")
                if tid in by_id:
                    by_id[tid].output = content
                    if status == "error":
                        by_id[tid].error = str(content)
        return cls(tool_calls=calls, final_answer=final, steps=ai_turns or None, **kwargs)
