"""
ui/gradio_app.py
----------------
Gradio-based chat interface for the Sidekick browser operator agent.

Features:
- Chat interface with message history
- Real-time step-by-step progress display
- Screenshot gallery from the browser session
- Task examples for quick start
- Browser status indicator
"""

import asyncio
import os
import threading
from pathlib import Path
from typing import Iterator

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from tools.browser_tools import BrowserToolkit
from agent.graph import build_graph
from agent.state import AgentState


# ─────────────────────────────────────────────────────────────────
# Global Browser Toolkit (shared across requests)
# ─────────────────────────────────────────────────────────────────

_toolkit: BrowserToolkit | None = None
_toolkit_lock = threading.Lock()


async def get_or_create_toolkit() -> BrowserToolkit:
    """Lazily initialize the browser toolkit (singleton)."""
    global _toolkit
    with _toolkit_lock:
        if _toolkit is None:
            _toolkit = BrowserToolkit()
            await _toolkit.start()
    return _toolkit


async def reset_browser():
    """Close and restart the browser session."""
    global _toolkit
    with _toolkit_lock:
        if _toolkit:
            await _toolkit.close()
        _toolkit = BrowserToolkit()
        await _toolkit.start()
    return "🔄 Browser restarted."


# ─────────────────────────────────────────────────────────────────
# Agent Runner
# ─────────────────────────────────────────────────────────────────

def run_agent_sync(task: str, history: list) -> tuple[list, list]:
    """
    Run the agent synchronously (for Gradio compatibility).
    Returns (updated_history, screenshot_paths).
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        toolkit = loop.run_until_complete(get_or_create_toolkit())
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

        # Build chat history from agent messages
        new_history = list(history)
        new_history.append({"role": "user", "content": task})

        # Combine all agent messages into one assistant turn
        agent_messages = [
            msg.content for msg in final_state["messages"]
            if hasattr(msg, "content")
        ]
        combined = "\n\n".join(agent_messages)
        new_history.append({"role": "assistant", "content": combined})

        screenshots = final_state.get("screenshots", [])
        return new_history, screenshots

    except Exception as e:
        error_msg = f"❌ Agent error: {str(e)}\n\nPlease check your API key in `.env` and try again."
        new_history = list(history)
        new_history.append({"role": "user", "content": task})
        new_history.append({"role": "assistant", "content": error_msg})
        return new_history, []
    finally:
        loop.close()


# ─────────────────────────────────────────────────────────────────
# Gradio Interface
# ─────────────────────────────────────────────────────────────────

EXAMPLE_TASKS = [
    "Search for 'LangGraph tutorial' on Google and summarize the top 3 results",
    "Go to github.com/trending and list the top 5 trending Python repositories",
    "Navigate to news.ycombinator.com and summarize the top 5 stories",
    "Search Wikipedia for 'Reinforcement Learning' and extract the introduction",
    "Go to arxiv.org and find the latest papers on 'multimodal AI'",
]

CSS = """
.gradio-container {
    font-family: 'IBM Plex Mono', monospace !important;
}
.status-bar {
    background: #0f1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
    color: #58a6ff;
    font-family: monospace;
    font-size: 13px;
}
.example-btn {
    font-size: 12px !important;
}
footer { display: none !important; }
"""

HEADER = """
# 🤖 Sidekick — Browser Operator Agent
**LangGraph + Playwright + Groq** | Your AI that works *inside* your browser

---
"""


def build_gradio_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""

    with gr.Blocks(title="Sidekick — Browser Operator Agent") as app:

        gr.Markdown(HEADER)

        with gr.Row():
            # Left: Chat panel
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Sidekick Agent",
                    height=520,
                    type="messages",
                    show_label=True,
                    bubble_full_width=False,
                    avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=sidekick"),
                    render_markdown=True,
                )

                with gr.Row():
                    task_input = gr.Textbox(
                        placeholder="Describe what you want Sidekick to do in your browser...",
                        show_label=False,
                        scale=5,
                        lines=2,
                    )
                    run_btn = gr.Button("▶ Run", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("🗑 Clear Chat", variant="secondary", size="sm")
                    reset_btn = gr.Button("🔄 Reset Browser", variant="secondary", size="sm")
                    status_box = gr.Textbox(
                        value="🟢 Browser ready",
                        show_label=False,
                        interactive=False,
                        scale=2,
                        elem_classes=["status-bar"],
                    )

                # Example tasks
                gr.Markdown("**💡 Example tasks:**")
                with gr.Row():
                    for i, ex in enumerate(EXAMPLE_TASKS[:3]):
                        gr.Button(
                            f"Try: {ex[:45]}...",
                            size="sm",
                            elem_classes=["example-btn"],
                        ).click(
                            fn=lambda e=ex: e,
                            outputs=task_input,
                        )

            # Right: Screenshots panel
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Browser Screenshots")
                screenshot_gallery = gr.Gallery(
                    label="Step-by-step screenshots",
                    show_label=False,
                    columns=2,
                    height=580,
                    object_fit="contain",
                )

        # ── State
        history_state = gr.State([])

        # ── Event handlers

        def on_run(task: str, history: list):
            if not task.strip():
                return history, [], "⚠️ Please enter a task."
            status = f"⏳ Running: {task[:60]}..."
            yield history, [], status

            new_history, screenshots = run_agent_sync(task, history)
            final_status = f"✅ Done — {len(screenshots)} screenshot(s) captured"
            yield new_history, screenshots, final_status

        def on_clear():
            return [], [], "🟢 Browser ready"

        def on_reset():
            loop = asyncio.new_event_loop()
            msg = loop.run_until_complete(reset_browser())
            loop.close()
            return msg

        run_btn.click(
            fn=on_run,
            inputs=[task_input, history_state],
            outputs=[chatbot, screenshot_gallery, status_box],
        ).then(
            fn=lambda h: h,
            inputs=[chatbot],
            outputs=[history_state],
        )

        task_input.submit(
            fn=on_run,
            inputs=[task_input, history_state],
            outputs=[chatbot, screenshot_gallery, status_box],
        ).then(
            fn=lambda h: h,
            inputs=[chatbot],
            outputs=[history_state],
        )

        clear_btn.click(
            fn=on_clear,
            outputs=[chatbot, screenshot_gallery, status_box],
        ).then(fn=lambda: [], outputs=[history_state])

        reset_btn.click(
            fn=on_reset,
            outputs=[status_box],
        )

    return app
