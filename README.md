# 🤖 Sidekick — Your Browser Operator Agent

> Build your own version of OpenAI's Operator — an AI agent that works *with you* inside your browser, powered by **LangGraph** + **Playwright** + **Gradio**.

## 🎬 Demo

![Sidekick en action](assets/demo_github_trending.png)
*Sidekick navigue autonomement sur GitHub Trending*

---

## 🧠 What is Sidekick?

Sidekick is a local AI browser automation agent that:
- Understands **natural language instructions** ("Search for AI jobs in Paris and save the top 5")
- Uses **LangGraph** to plan and execute multi-step browser tasks
- Controls a real **Chromium browser** via Playwright
- Presents a clean **Gradio chat interface** to interact with you
- Captures **screenshots** at each step so you can follow along

Inspired by OpenAI Operator, but 100% local and open-source.

---

## 🏗️ Architecture

```
User (Gradio Chat)
       │
       ▼
  LangGraph Agent
  ┌────────────────────────────────┐
  │  Planner Node                  │
  │    → Breaks task into steps    │
  │                                │
  │  Tool Executor Node            │
  │    → navigate_to               │
  │    → click_element             │
  │    → type_text                 │
  │    → extract_content           │
  │    → take_screenshot           │
  │    → scroll_page               │
  │    → wait_for_element          │
  │                                │
  │  Summarizer Node               │
  │    → Returns result to user    │
  └────────────────────────────────┘
       │
       ▼
  Playwright (Chromium Browser)
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/FatimaChahal/sidekick-operator-agent.git
cd sidekick-operator-agent

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your API key (Groq recommended — free & fast)
```

### 3. Run

```bash
python main.py
```

Open http://localhost:7860 in your browser.

---

## 🔧 Example Tasks

- `"Go to wikipedia.org and search for LangGraph, then summarize the intro"`
- `"Search for 'AI engineer jobs France' on LinkedIn and list 5 results"`
- `"Navigate to github.com/trending and tell me the top 3 Python repos today"`
- `"Go to hacker news and summarize the top 5 stories"`
- `"Search 'postdoc AI France' on Google Scholar and extract titles"`

---

## 📁 Project Structure

```
sidekick-operator-agent/
├── main.py                  # Entry point
├── requirements.txt
├── .env.example
├── agent/
│   ├── __init__.py
│   ├── graph.py             # LangGraph workflow definition
│   ├── nodes.py             # Planner, Executor, Summarizer nodes
│   └── state.py             # AgentState definition
├── tools/
│   ├── __init__.py
│   └── browser_tools.py     # Playwright browser tool functions
├── ui/
│   ├── __init__.py
│   └── gradio_app.py        # Gradio chat interface
└── tests/
    └── test_tools.py        # Unit tests
```

---

## 🔑 API Keys

| Provider | Used for | Free tier |
|----------|----------|-----------|
| [Groq](https://console.groq.com) | LLM (recommended) | ✅ Yes |
| [OpenAI](https://platform.openai.com) | LLM (alternative) | ⚠️ Paid |

Set `LLM_PROVIDER=groq` or `LLM_PROVIDER=openai` in `.env`.

---

## 🛡️ Safety Note

This agent can interact with real websites. Use responsibly:
- Do not use on sites requiring login without understanding the implications
- Actions are reversible in most cases (reading, searching)
- For destructive actions (form submissions, purchases), confirm manually

---

## 🤝 Built With

- [LangGraph](https://github.com/langchain-ai/langgraph) — Agent orchestration
- [Playwright](https://playwright.dev/python/) — Browser automation
- [Gradio](https://gradio.app) — Chat interface
- [LangChain](https://python.langchain.com) — LLM abstraction
- [Groq](https://groq.com) — Fast LLM inference

---

## 👩‍💻 Author

**Fatima Chahal** — AI Engineer & Postdoctoral Researcher  
🔗 [GitHub](https://github.com/FatimaChahal) | [LinkedIn](https://linkedin.com/in/fatima-chahal)

---

*Part of an open-source AI Engineering portfolio.*
