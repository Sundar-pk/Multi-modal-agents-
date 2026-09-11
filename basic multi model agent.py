"""
basic_multi_model_agent.py

A BASIC starting point — not the polished reference file, just the core
mechanic: three nodes, three different internal models, one shared state,
wired together with LangGraph. Deliberately minimal so you can extend it
live in front of the room (add a node, add a conditional edge, add error
handling — whatever your session needs).

Run: python basic_multi_model_agent.py
"""

import json
import requests
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------
# 1. Internal model endpoints — same model, different base_url/key.
#    Fill these in with your real internal gateway details.
# ---------------------------------------------------------------------
MODELS = {
    "mistral": {"base_url": "https://internal-llm-gw/mistral/v1", "model": "mistral-large-internal", "api_key": "REPLACE_ME"},
    "qwen":    {"base_url": "https://internal-llm-gw/qwen/v1",    "model": "qwen2.5-72b-internal",    "api_key": "REPLACE_ME"},
    "gpt-oss": {"base_url": "https://internal-llm-gw/gpt-oss/v1", "model": "gpt-oss-120b-internal",   "api_key": "REPLACE_ME"},
}


def call_llm(which: str, prompt: str) -> str:
    """One function every node calls — swap `which` to swap the model."""
    cfg = MODELS[which]
    resp = requests.post(
        f"{cfg['base_url']}/chat/completions",
        headers={"Authorization": f"Bearer {cfg['api_key']}"},
        json={"model": cfg["model"], "messages": [{"role": "user", "content": prompt}]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------
# 2. Shared state — the contract between nodes
# ---------------------------------------------------------------------
class AgentState(TypedDict):
    application_text: str
    extracted: str
    risk_notes: str
    decision: str


# ---------------------------------------------------------------------
# 3. Nodes — each one a plain function, each backed by a different model
# ---------------------------------------------------------------------
def intake_node(state: AgentState) -> dict:
    prompt = f"Summarize the key facts from this loan application in 2-3 bullet points:\n{state['application_text']}"
    return {"extracted": call_llm("mistral", prompt)}


def risk_node(state: AgentState) -> dict:
    prompt = f"Given these application facts, assess credit risk in one paragraph:\n{state['extracted']}"
    return {"risk_notes": call_llm("qwen", prompt)}


def decision_node(state: AgentState) -> dict:
    prompt = f"Based on this risk assessment, recommend APPROVE, REJECT, or MANUAL_REVIEW with a one-line reason:\n{state['risk_notes']}"
    return {"decision": call_llm("gpt-oss", prompt)}


# ---------------------------------------------------------------------
# 4. Wire the graph — linear for now, add branches as you go
# ---------------------------------------------------------------------
graph = StateGraph(AgentState)
graph.add_node("intake", intake_node)
graph.add_node("risk", risk_node)
graph.add_node("decision", decision_node)

graph.add_edge(START, "intake")
graph.add_edge("intake", "risk")
graph.add_edge("risk", "decision")
graph.add_edge("decision", END)

app = graph.compile()


# ---------------------------------------------------------------------
# 5. Run it
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sample = "Applicant requests $45,000 to expand a retail business. Monthly income $6,200, existing debt $1,100."
    result = app.invoke({"application_text": sample})
    print(json.dumps(result, indent=2))
