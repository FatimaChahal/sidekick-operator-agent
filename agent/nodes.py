"""
agent/nodes.py
--------------
LangGraph node functions for the Sidekick browser operator agent.

Three nodes:
1. planner_node   — Takes the task, produces a step-by-step plan
2. executor_node  — Executes the current step using a browser tool
3. summarizer_node — Synthesizes all step results into a final answer
"""

import json
import os
import re
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.state import AgentState
from tools.browser_tools import BrowserToolkit


# ─────────────────────────────────────────────────────────────────
# LLM Factory
# ─────────────────────────────────────────────────────────────────

def get_llm():
    """Instantiate the LLM based on LLM_PROVIDER env variable."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.1,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}. Use 'groq' or 'openai'.")


# ─────────────────────────────────────────────────────────────────
# Node 1: Planner
# ─────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a browser automation planner. 
Given a natural language task, produce a precise, ordered list of browser actions to accomplish it.

Available tools:
{tools}

Rules:
- Each step must use ONE tool and be a single action
- Use specific selectors when possible (CSS, text-based)
- For web searches, prefer google_search over manual navigation
- Always end with extract_content to gather results
- Maximum {max_steps} steps
- Be concise but precise

Respond with ONLY a JSON array of steps, no explanation:
[
  {{"step": 1, "tool": "tool_name", "params": {{"param1": "value1"}}, "description": "What this does"}},
  ...
]
"""

def planner_node(state: AgentState, toolkit: BrowserToolkit) -> AgentState:
    """
    Planner node: converts the user task into an ordered list of browser steps.
    """
    llm = get_llm()
    max_steps = int(os.getenv("MAX_STEPS", "15"))

    system_msg = SystemMessage(content=PLANNER_SYSTEM.format(
        tools=toolkit.get_tool_descriptions(),
        max_steps=max_steps,
    ))
    human_msg = HumanMessage(content=f"Task: {state['task']}")

    response = llm.invoke([system_msg, human_msg])
    raw = response.content.strip()

    # Parse JSON plan
    try:
        # Strip markdown code fences if present
        clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        plan_data = json.loads(clean)
        plan_steps = [
            f"Step {s['step']}: [{s['tool']}] {s['description']} — params: {json.dumps(s['params'])}"
            for s in plan_data
        ]
        # Store structured plan for executor
        state["plan"] = [json.dumps(s) for s in plan_data]
    except (json.JSONDecodeError, KeyError):
        # Fallback: treat as plain text steps
        plan_steps = [line.strip() for line in raw.split("\n") if line.strip()]
        state["plan"] = plan_steps

    state["current_step"] = 0
    state["step_results"] = []
    state["screenshots"] = []
    state["error"] = None

    state["messages"].append(
        AIMessage(content=f"📋 **Plan created** ({len(state['plan'])} steps):\n" +
                  "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan_steps)))
    )
    return state


# ─────────────────────────────────────────────────────────────────
# Node 2: Executor
# ─────────────────────────────────────────────────────────────────

EXECUTOR_SYSTEM = """You are a browser action executor. 
You have completed some steps and need to decide the next browser tool call.

Current task: {task}
Steps completed so far: {completed}
Current step to execute: {current_step}

Based on what you've seen so far, determine the exact tool call for this step.

Respond with ONLY valid JSON:
{{"tool": "tool_name", "params": {{"param1": "value1"}}}}
"""

async def executor_node(state: AgentState, toolkit: BrowserToolkit) -> AgentState:
    """
    Executor node: runs the current step's browser tool, updates state.
    Continues until all steps are done or max steps reached.
    """
    llm = get_llm()
    max_steps = int(os.getenv("MAX_STEPS", "15"))
    idx = state["current_step"]
    plan = state["plan"]

    if idx >= len(plan) or idx >= max_steps:
        return state  # Done — move to summarizer

    current_step_raw = plan[idx]

    # Try to parse pre-structured step from planner
    try:
        step_data = json.loads(current_step_raw)
        tool_name = step_data["tool"]
        params = step_data.get("params", {})
    except (json.JSONDecodeError, KeyError):
        # Ask LLM to determine the tool call
        completed_summary = "\n".join(
            [f"- {r['step']}: {r['result'][:200]}" for r in state.get("step_results", [])]
        )
        system_msg = SystemMessage(content=EXECUTOR_SYSTEM.format(
            task=state["task"],
            completed=completed_summary or "None yet",
            current_step=current_step_raw,
        ))
        response = llm.invoke([system_msg])
        raw = response.content.strip()
        clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        tool_data = json.loads(clean)
        tool_name = tool_data["tool"]
        params = tool_data.get("params", {})

    # Execute the tool
    print(f"[Executor] Step {idx+1}: {tool_name}({params})")
    result = await toolkit.execute_tool(tool_name, params)

    # Record result
    step_result = {
        "step": current_step_raw,
        "tool": tool_name,
        "params": params,
        "result": result.get("result", ""),
        "success": result.get("success", False),
        "screenshot": result.get("screenshot"),
    }
    state["step_results"].append(step_result)

    if result.get("screenshot"):
        state["screenshots"].append(result["screenshot"])

    # Update messages with progress
    status = "✅" if result["success"] else "❌"
    state["messages"].append(
        AIMessage(content=f"{status} **Step {idx+1}** `{tool_name}`: {result['result'][:300]}")
    )

    state["current_step"] = idx + 1
    return state


# ─────────────────────────────────────────────────────────────────
# Node 3: Summarizer
# ─────────────────────────────────────────────────────────────────

SUMMARIZER_SYSTEM = """You are a helpful AI assistant summarizing the results of a browser automation task.

Original task: {task}

Steps executed:
{steps_summary}

Based on the above browser actions and their results, provide a clear, well-structured answer to the user's original task.
- Be concise but complete
- Use markdown formatting (lists, bold, headers) where helpful
- If something went wrong, explain what happened
- If you found data (links, text, results), present it clearly
"""

def summarizer_node(state: AgentState) -> AgentState:
    """
    Summarizer node: takes all step results and produces a final answer.
    """
    llm = get_llm()

    steps_summary = "\n\n".join([
        f"Step {i+1} [{r['tool']}]: {'✅' if r['success'] else '❌'}\n"
        f"Params: {json.dumps(r['params'])}\n"
        f"Result: {r['result'][:1000]}"
        for i, r in enumerate(state.get("step_results", []))
    ])

    system_msg = SystemMessage(content=SUMMARIZER_SYSTEM.format(
        task=state["task"],
        steps_summary=steps_summary or "No steps were executed.",
    ))

    response = llm.invoke([system_msg])
    final_answer = response.content.strip()

    state["final_answer"] = final_answer
    n_shots = len(state.get("screenshots", []))
    state["messages"].append(
        AIMessage(content=f"🎯 **Task Complete!**\n\n{final_answer}"
                  + (f"\n\n📸 *{n_shots} screenshot(s) saved to `{os.getenv('SCREENSHOT_DIR', 'screenshots')}/`*" if n_shots else ""))
    )
    return state
