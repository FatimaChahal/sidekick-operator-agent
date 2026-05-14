"""
main.py — Sidekick Browser Operator Agent
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def check_environment():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    key = os.getenv("GROQ_API_KEY", "") if provider == "groq" else os.getenv("OPENAI_API_KEY", "")
    if not key or "your_" in key:
        print(f"❌ Clé API manquante dans .env (provider: {provider})")
        sys.exit(1)
    Path(os.getenv("SCREENSHOT_DIR", "screenshots")).mkdir(parents=True, exist_ok=True)
    print(f"✅ Environment OK — LLM provider: {provider.upper()}")
    print(f"✅ Screenshots → {Path(os.getenv('SCREENSHOT_DIR','screenshots')).resolve()}")

def print_banner():
    print(r"""
  ___  _     _      _    _    _
 / __\(_)  _| | ___| | _(_)__| | __
 \__ \| | / _` |/ _ \ |/ / |/ _` |/ /
 ___) | || (_| |  __/   <| | (_| |<
|____/|_| \__,_|\___|_|\_\_|\__,_|\_\

 🤖 Sidekick — Browser Operator Agent (Flask UI)
 LangGraph + Playwright + Groq
    """)

def main():
    print_banner()
    check_environment()
    from ui.flask_app import start
    port = int(os.getenv("FLASK_PORT", "5000"))
    print(f"   Browser: {'headless' if os.getenv('BROWSER_HEADLESS','false').lower()=='true' else 'visible'}")
    print("\nCtrl+C pour arrêter.\n")
    start(port=port)

if __name__ == "__main__":
    main()