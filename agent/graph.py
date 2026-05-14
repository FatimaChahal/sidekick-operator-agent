"""
agent/graph.py
--------------
LangGraph StateGraph definition for the Sidekick browser operator agent.

Graph flow:
    START → planner → executor (loop) → summarizer → END

The executor loops back to itself until all steps are done,
then transitions to the summarizer.
"""

import asyncio
from functools import partial
from typing import Literal

from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.nodes import planner_node, executor_node, summarizer_node
from tools.browser_tools import BrowserToolkit


# ─────────────────────────────────────────────────────────────────
# Routing Logic
# ─────────────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> Literal["executor", "summarizer"]:
    """
    Conditional edge: decides whether to keep executing steps
    or move to the summarizer.

    Routes to "executor" if there are remaining steps.
    Routes to "summarizer" once all steps are done.
    """
    current = state.get("current_step", 0)
    plan = state.get("plan", [])
    max_steps = 15  # Safety cap

    if current < len(plan) and current < max_steps:
        return "executor"
    return "summarizer"


# ─────────────────────────────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────────────────────────────

def build_graph(toolkit: BrowserToolkit) -> StateGraph:
    """
    Build and compile the LangGraph StateGraph.

    Args:
        toolkit: A started BrowserToolkit instance

    Returns:
        Compiled LangGraph app
    """
    graph = StateGraph(AgentState)

    # Bind the toolkit to node functions (partial application)
    planner = partial(planner_node, toolkit=toolkit)
    executor = partial(_sync_executor, toolkit=toolkit)

    # Add nodes
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("summarizer", summarizer_node)

    # Add edges
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")

    # Conditional loop: executor → executor (more steps) or → summarizer (done)
    graph.add_conditional_edges(
        "executor",
        should_continue,
        {
            "executor": "executor",
            "summarizer": "summarizer",
        },
    )
    graph.add_edge("summarizer", END)

    return graph.compile()


def _sync_executor(state: AgentState, toolkit: BrowserToolkit) -> AgentState:
    """
    Synchronous wrapper for the async executor_node.
    LangGraph nodes must be sync; we bridge with asyncio.
    """
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # In Gradio / Jupyter context — create a new event loop in a thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, executor_node(state, toolkit))
            return future.result()
    else:
        return loop.run_until_complete(executor_node(state, toolkit))


# ─────────────────────────────────────────────────────────────────
# High-level runner
# ─────────────────────────────────────────────────────────────────

async def run_agent(task: str, toolkit: BrowserToolkit) -> AgentState:
    """
    Run the full agent pipeline for a given task.

    Args:
        task: Natural language task for the agent
        toolkit: A started BrowserToolkit instance

    Returns:
        Final AgentState with results
    """
    graph = build_graph(toolkit)

    initial_state: AgentState = {
        "messages": [],
        "task": task,
        "plan": [],
        "current_step": 0,
        "step_results": [],
        "screenshots": [],
        "final_answer": "",
        "error": None,
        "browser_context": None,
    }

    final_state = graph.invoke(initial_state)
    return final_state
