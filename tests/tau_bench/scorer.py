"""
τ2-bench-style scorer for MetaAgent evaluation.

Evaluation dimensions (from τ2-bench):
- ACTION: Did the agent call the right tools?
- COMMUNICATE: Did the agent convey expected information?
- NL_ASSERTIONS: Does the response contain expected keywords/patterns?
- ENV: Did the agent correctly modify state? (memory, RAG)

Score = product of applicable dimensions (τ2-bench product scoring).
"""
import re
from typing import List, Dict, Any


class Scorer:
    """Evaluate agent responses against expected criteria."""

    @staticmethod
    def fuzzy_match(answer: str, expected: List[str]) -> float:
        """Fraction of expected keywords found in answer."""
        if not expected:
            return 1.0
        if not answer:
            return 0.0
        answer_lower = answer.lower()
        hits = sum(1 for kw in expected if kw.lower() in answer_lower)
        return min(1.0, hits / max(len(expected), 1))

    @staticmethod
    def no_crash(answer: str) -> bool:
        """Check if answer contains crash/error markers."""
        if not answer:
            return True  # empty is not a crash
        crash_markers = [
            "traceback", "nonetype", "attributeerror",
            "keyerror", "typeerror", "indexerror",
            "iteration limit reached",
        ]
        answer_lower = answer.lower()
        return not any(m in answer_lower for m in crash_markers)

    @staticmethod
    def tool_used(events: List[dict], tool_name: str) -> bool:
        """Check if a specific tool was called."""
        for e in events:
            if e.get("type") == "react_action":
                tn = str(e.get("tool_name", "")).lower()
                if tool_name.lower() in tn:
                    return True
        return False

    @staticmethod
    def any_tool_used(events: List[dict]) -> bool:
        """Check if any tool was called."""
        return any(e.get("type") == "react_action" for e in events)

    @staticmethod
    def tool_agent_triggered(events: List[dict]) -> bool:
        """Check if ToolAgent was invoked."""
        return any(e.get("type") == "tool_agent_invoke" for e in events)

    @staticmethod
    def direct_answer(events: List[dict]) -> bool:
        """Check if MetaAgent answered without dispatching."""
        has_result = any(e.get("type") == "final_result" for e in events)
        has_delegate = any(e.get("type") == "agent_delegate" for e in events)
        return has_result and not has_delegate

    @staticmethod
    def dispatch_count(events: List[dict]) -> int:
        """Count sub-agent dispatches."""
        return sum(1 for e in events if e.get("type") == "agent_delegate")

    @staticmethod
    def error_count(events: List[dict]) -> int:
        """Count error events."""
        return sum(1 for e in events if e.get("type") in ("error", "_err"))


def score_task(task: dict, events_list: List[List[dict]],
               answers: List[str]) -> Dict[str, Any]:
    """
    Score a single multi-turn task.

    Scoring uses τ2-bench's product-of-dimensions approach:
        score = env_score * comm_score * action_score * nl_score

    Only dimensions applicable to the task metric are included.
    Non-applicable dimensions default to 1.0 (neutral).
    """
    metric = task.get("metric", "recall_accuracy")
    expected = task.get("expected", [])
    all_events = [e for turn_events in events_list for e in turn_events]

    result = {
        "task_id": task["task_id"],
        "category": task.get("category", ""),
        "metric": metric,
        "turns": len(task["turns"]),
        "dispatch_count": Scorer.dispatch_count(all_events),
        "error_count": Scorer.error_count(all_events),
        "tool_agent_triggered": Scorer.tool_agent_triggered(all_events),
    }

    # ── Dimension scores ──
    nl_score = 1.0
    action_score = 1.0
    comm_score = 1.0
    env_score = 1.0

    # NL: Check if final answer contains expected keywords
    if expected:
        final_answer = answers[-1] if answers else ""
        nl_score = Scorer.fuzzy_match(final_answer, expected)

    # Root cause check: if answer starts with raw agent data, it's a communication failure
    if answers:
        for ans in answers:
            if ans and ("Agent Name:" in ans[:50] or "Tasks:" in ans[:50]):
                comm_score = 0.0
                break

    # Crash check
    if not all(Scorer.no_crash(a) for a in answers if a is not None):
        env_score = 0.0

    # Metric-specific adjustments
    if metric == "direct_answer":
        action_score = 1.0 if Scorer.direct_answer(all_events) else 0.3

    elif metric == "tool_selection":
        action_score = 1.0 if Scorer.any_tool_used(all_events) else 0.0

    elif metric == "tool_creation":
        action_score = 1.0 if Scorer.tool_agent_triggered(all_events) else (
            0.5 if Scorer.any_tool_used(all_events) else 0.0
        )

    elif metric in ("rag_retrieval", "memory_retrieval",
                    "state_tracking", "recall_accuracy", "multi_step", "memory_batch"):
        action_score = 1.0 if Scorer.any_tool_used(all_events) else 0.2

    elif metric == "no_crash":
        pass  # env_score already handles this

    # τ2-bench product scoring
    result["scores"] = {
        "env": round(env_score, 3),
        "action": round(action_score, 3),
        "comm": round(comm_score, 3),
        "nl": round(nl_score, 3),
    }
    result["score"] = round(env_score * action_score * comm_score * nl_score, 3)
    result["passed"] = result["score"] >= 0.5

    return result
