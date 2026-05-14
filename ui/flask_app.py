import os
import asyncio
from flask import Flask, request, jsonify, render_template_string
from tools.browser_tools import BrowserToolkit
from agent.graph import build_graph
from agent.state import AgentState

app = Flask(__name__)
_toolkit = None

async def get_toolkit():
    global _toolkit
    if _toolkit is None:
        _toolkit = BrowserToolkit()
        await _toolkit.start()
    return _toolkit

def run_agent_sync(task):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        toolkit = loop.run_until_complete(get_toolkit())
        graph = build_graph(toolkit)
        state = {
            "messages": [], "task": task, "plan": [],
            "current_step": 0, "step_results": [], "screenshots": [],
            "final_answer": "", "error": None, "browser_context": None,
        }
        final = graph.invoke(state)
        steps = [{"tool": r["tool"], "result": r["result"][:300], "success": r["success"]} for r in final.get("step_results", [])]
        return {"answer": final.get("final_answer", ""), "steps": steps}
    except Exception as e:
        return {"answer": f"Erreur : {e}", "steps": []}
    finally:
        loop.close()

HTML = open(os.path.join(os.path.dirname(__file__), "index.html")).read() if os.path.exists(os.path.join(os.path.dirname(__file__), "index.html")) else "<h1>Sidekick</h1>"

@app.route("/")
def index():
    return render_template_string("""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Sidekick</title>
<style>*{box-sizing:border-box;margin:0;padding:0}body{font-family:monospace;background:#0d1117;color:#c9d1d9;padding:2rem}h1{color:#58a6ff;margin-bottom:.3rem}p{color:#8b949e;font-size:.85rem;margin-bottom:2rem}.box{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:1.2rem;margin-bottom:1rem}textarea{width:100%;background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:.8rem;font-family:inherit;font-size:.95rem;min-height:80px}button{background:#238636;color:#fff;border:none;border-radius:6px;padding:.7rem 1.6rem;cursor:pointer;margin-top:.6rem}button:disabled{background:#21262d;color:#8b949e}.ex{background:#21262d;border:1px solid #30363d;border-radius:4px;padding:.3rem .7rem;font-size:.78rem;cursor:pointer;color:#8b949e;margin:.3rem .3rem 0 0;display:inline-block}.ok{color:#3fb950}.err{color:#f85149}.tag{background:#1f6feb;color:#fff;border-radius:3px;padding:.1rem .4rem;font-size:.75rem;margin-right:.4rem}.answer{white-space:pre-wrap;line-height:1.6;font-size:.9rem;margin-top:.6rem}#spin{display:none;color:#58a6ff;margin-top:.5rem}li{padding:.4rem 0;border-bottom:1px solid #21262d;list-style:none}</style></head>
<body>
<h1>🤖 Sidekick — Browser Operator Agent</h1>
<p>LangGraph + Playwright + Groq</p>
<div class="box">
<textarea id="task" placeholder="Ex: Cherche LangGraph sur Google et résume les 3 premiers résultats..."></textarea>
<button id="btn" onclick="run()">▶ Lancer</button>
<div id="spin">⏳ Agent en cours...</div>
<div style="margin-top:.8rem">
<span class="ex" onclick="document.getElementById('task').value=this.innerText">GitHub trending Python repos</span>
<span class="ex" onclick="document.getElementById('task').value=this.innerText">Recherche Wikipedia sur LangGraph</span>
<span class="ex" onclick="document.getElementById('task').value=this.innerText">Top 5 Hacker News stories</span>
</div></div>
<div class="box" id="res" style="display:none">
<strong style="color:#58a6ff">📋 Étapes</strong>
<ul id="steps" style="margin-top:.6rem"></ul>
<hr style="border-color:#30363d;margin:1rem 0">
<strong style="color:#58a6ff">🎯 Réponse</strong>
<div class="answer" id="answer"></div>
</div>
<script>
async function run(){
  const task=document.getElementById('task').value.trim();
  if(!task)return;
  document.getElementById('btn').disabled=true;
  document.getElementById('spin').style.display='block';
  document.getElementById('res').style.display='none';
  try{
    const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task})});
    const d=await r.json();
    document.getElementById('steps').innerHTML=d.steps.map(s=>'<li><span class="'+(s.success?'ok':'err')+'">'+(s.success?'✅':'❌')+'</span> <span class="tag">'+s.tool+'</span>'+s.result.substring(0,120)+'</li>').join('');
    document.getElementById('answer').textContent=d.answer;
    document.getElementById('res').style.display='block';
  }catch(e){document.getElementById('answer').textContent='Erreur: '+e.message;document.getElementById('res').style.display='block';}
  finally{document.getElementById('btn').disabled=false;document.getElementById('spin').style.display='none';}
}
</script></body></html>""")

@app.route("/run", methods=["POST"])
def run_route():
    task = request.json.get("task", "").strip()
    if not task:
        return jsonify({"error": "vide"}), 400
    return jsonify(run_agent_sync(task))

def start(port=5000):
    print(f"🚀 Sidekick → http://localhost:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False)
