"""Example tool-using agent.

The demo agent is a scripted planner so it runs offline. For a real
LangGraph agent the adapter is usually three lines:

    def run_agent(task: str) -> AgentTrace:
        state = graph.invoke({"messages": [("user", task)]}, config={"recursion_limit": 15})
        return AgentTrace.from_langchain_messages(state["messages"])

Catch ``GraphRecursionError`` and return ``AgentTrace(terminated=False)``
so step-limit failures show up as test failures, not crashes.
"""

from __future__ import annotations

import re

from aitest.types import AgentTrace, ToolCall

ORDERS = {"A100": {"status": "shipped", "carrier": "DHL", "eta": "2 days"}, "A200": {"status": "processing"}}


def lookup_order(order_id: str) -> dict:
    if order_id not in ORDERS:
        raise KeyError(f"order {order_id} not found")
    return ORDERS[order_id]


def create_ticket(summary: str, priority: str = "normal") -> dict:
    return {"ticket_id": "T-1", "priority": priority}


def run_agent(task: str) -> AgentTrace:
    calls: list[ToolCall] = []
    low = task.lower()
    m = re.search(r"\b([A-Z]\d{3})\b", task)

    if m and ("where" in low or "status" in low):
        call = ToolCall("lookup_order", {"order_id": m.group(1)})
        try:
            call.output = lookup_order(m.group(1))
            info = call.output
            answer = f"Order {m.group(1)} is {info['status']}" + (f", arriving in {info['eta']} via {info['carrier']}." if "eta" in info else ".")
        except KeyError as e:
            call.error = str(e)
            answer = f"I couldn't find order {m.group(1)}. Please check the number."
        calls.append(call)
        return AgentTrace(tool_calls=calls, final_answer=answer)

    if "damaged" in low or "broken" in low:
        if m:
            calls.append(ToolCall("lookup_order", {"order_id": m.group(1)}, output=ORDERS.get(m.group(1))))
        calls.append(ToolCall("create_ticket", {"summary": task[:80], "priority": "high"}, output=create_ticket(task, "high")))
        return AgentTrace(tool_calls=calls, final_answer="I've opened ticket T-1 with high priority. Our team will contact you within 24 hours.")

    if "delete" in low and "account" in low:
        return AgentTrace(final_answer="I can't delete accounts. Please use Settings > Privacy or contact support.")

    return AgentTrace(final_answer="Could you tell me your order number so I can help?")
