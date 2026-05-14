"""
agent/state.py
--------------
AgentState — the shared state object that flows through all LangGraph nodes.
"""

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Shared state for the Sidekick browser operator agent.

    Fields
    ------
    messages : list
        Full conversation history (user + assistant turns).
        Uses LangGraph's `add_messages` reducer to append incrementally.

    task : str
        The original natural-language task provided by the user.

    plan : list[str]
        Ordered list of browser action steps produced by the Planner node.

    current_step : int
        Index of the step currently being executed.

    step_results : list[dict]
        Accumulated results from each executed step.
        Each entry: {"step": str, "result": str, "screenshot": str | None}

    screenshots : list[str]
        Paths to screenshots taken during execution.

    final_answer : str
        The summarized result returned to the user after task completion.

    error : str | None
        Any error message encountered during execution.

    browser_context : dict | None
        Internal metadata about the current browser state
        (current URL, page title, etc.). Updated after each tool call.
    """

    messages: Annotated[list, add_messages]
    task: str
    plan: list[str]
    current_step: int
    step_results: list[dict[str, Any]]
    screenshots: list[str]
    final_answer: str
    error: str | None
    browser_context: dict[str, Any] | None
