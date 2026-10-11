 """
agent.py - Gemini decides which predefined tool(s) to run, we run them locally,
and Gemini turns the results into a plain-language answer.

Flow for one question:
  1. Send question + tool list (names, descriptions, parameters) to Gemini.
  2. Gemini replies with function call(s)  ->  we run them with tools.run_tool().
  3. Send the results back  ->  Gemini may call more tools or write the answer.
  4. Check every number in the answer against the tool results (faithfulness check).
  5. Save a trace (steps, tools, args, outputs, LLM calls, tokens, latency) to traces/<run_id>.json.

Env vars: GEMINI_API_KEY, GEMINI_MODEL (default gemini-3.8-flash), MOCK_LLM=true (no API, keyword router).
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path

from tools import PlanningSession, run_tool, tool_declarations

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MAX_TOOL_ROUNDS = 6
TRACE_DIR = Path("traces")

SYSTEM_PROMPT = """You are the planning assistant for a manufacturer with several plants.
You answer questions about allocating customer orders to plants, balancing landed cost, carbon and delivery dates.

RULES
1. Every number you state (cost, carbon, days, counts, percentages, dates) MUST come from a tool result in this
   conversation. Never estimate, extrapolate, convert currencies or invent figures. If a number is not in a tool
   result, say it is not available.
2. Always call a tool before answering a question about the plan, an order, or a scenario. For "what if" questions,
   use the simulate_* tools: they run on a copy, so the real data and plan never change. Say so when relevant.
3. You never create allocations yourself. The optimiser (tools) creates them; you explain them.
4. If the request is ambiguous (which plant? how much capacity? which dates?), ask ONE short clarifying question
   instead of guessing - unless a tool reports the assumption it used, in which case state that assumption.
5. If a request asks to break a rule (put an order in a plant that cannot make it, exceed capacity), explain that
   it is refused and why, using the tool result.
6. If a tool returns an error or an infeasible result, say so plainly and give the closest option the tool reports.
7. For a management brief: call get_brief_facts first, then write at most one page with headings
   (Summary, Cost & carbon, Delivery risks, Trade-off, Recommendation). Use only those numbers.
8. Style: plain business English, short paragraphs or bullets, money with thousands separators,
   carbon in tonnes (t) when large. Mention order IDs and plant names exactly as the tools give them.
"""


# =============================================================================
# Faithfulness check: every number in the answer must appear in a tool result
# =============================================================================
_NUM = re.compile(r"(?<![A-Za-z0-9\-])(-?\d[\d,]*\.?\d*)\s*(%|k|K|M|m\b|t\b)?")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _collect_numbers(obj, out: list):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)
    elif isinstance(obj, str):
        for m in re.findall(r"-?\d+\.?\d*", _DATE.sub(" ", obj.replace(",", ""))):
            out.append(float(m))


def check_numbers(answer: str, tool_outputs: list, question: str = "") -> list[str]:
    """Return numbers in the answer that do not match any tool output (within rounding)."""
    known: list[float] = []
    for t in tool_outputs:
        _collect_numbers(t, known)
    _collect_numbers(question, known)
    text = _DATE.sub(" ", answer)
    unverified = []
    for raw, suffix in _NUM.findall(text):
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        mult = {"k": 1e3, "K": 1e3, "M": 1e6}.get(suffix or "", 1)
        cands = {v * mult, abs(v * mult)}
        if suffix == "t":                      # tonnes may be reported as kg in tools
            cands |= {v * 1000}
        ok = any(abs(c - k) <= max(0.006 * abs(c), 0.51) or abs(abs(c) - abs(k)) <= max(0.006 * abs(c), 0.51)
                 for c in cands for k in known)
        if not ok and v not in (0, 1):
            unverified.append(f"{raw}{suffix or ''}")
    return unverified


# =============================================================================
# Gemini loop
# =============================================================================
def _gemini_answer(question, session, api_key, model, history, steps, usage):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    tools = [types.Tool(function_declarations=[types.FunctionDeclaration(**d) for d in tool_declarations()])]
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=tools, temperature=0.1)

    contents = []
    for turn in history[-6:]:                      # short memory of the chat (text only)
        contents.append(types.Content(role=turn["role"], parts=[types.Part(text=turn["text"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

    for _ in range(MAX_TOOL_ROUNDS):
        t0 = time.time()
        resp = client.models.generate_content(model=model, contents=contents, config=config)
        usage["llm_calls"] += 1
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            usage["tokens"] += (getattr(um, "total_token_count", 0) or 0)
        steps.append({"agent": "llm", "action": "generate", "ms": int((time.time() - t0) * 1000)})

        calls = resp.function_calls or []
        if not calls:
            return resp.text or "(no answer)"
        contents.append(resp.candidates[0].content)          # keep the model's function-call turn
        parts = []
        for call in calls:
            args = dict(call.args or {})
            t1 = time.time()
            result = run_tool(session, call.name, args)
            steps.append({"agent": "tool", "tool": call.name, "input": args, "output": result,
                          "ms": int((time.time() - t1) * 1000)})
            parts.append(types.Part.from_function_response(name=call.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=parts))
    return "I could not finish within the tool-call limit. Please ask a narrower question."


# =============================================================================
# MOCK mode: keyword router (no API key needed) - for testing the pipeline
# =============================================================================
def _mock_answer(question, session, steps, usage):
    q = question.lower()
    m_order = re.search(r"\b(?:o-?|order\s*)(\d{1,4})\b", q)
    m_plant = re.search(r"plant\s+([a-z])\b", q)
    m_pct = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
    if "brief" in q:
        name, args = "get_brief_facts", {}
    elif m_plant and m_pct and ("lose" in q or "loss" in q or "capacity" in q):
        name, args = "simulate_capacity_loss", {"plant": m_plant.group(1).upper(), "loss_pct": float(m_pct.group(1))}
    elif "carbon" in q and m_pct and ("cut" in q or "reduce" in q):
        name, args = "carbon_cut_cost", {"cut_pct": float(m_pct.group(1))}
    elif "why" in q and m_order:
        name, args = "explain_order", {"order_id": m_order.group(1)}
    elif "cheapest" in q:
        name, args = "cheapest_on_time_plan", {}
    elif "late" in q:
        name, args = "list_late_orders", {}
    elif "weight" in q:
        name, args = "compare_weight_settings", {}
    elif "utili" in q or "capacity" in q:
        name, args = "plant_utilisation", {}
    else:
        name, args = "get_plan_summary", {}
    t1 = time.time()
    result = run_tool(session, name, args)
    steps.append({"agent": "tool", "tool": name, "input": args, "output": result, "ms": int((time.time() - t1) * 1000)})
    return f"(MOCK mode - raw tool result, no LLM)\n\nTool `{name}` returned:\n```json\n{json.dumps(result, indent=1)[:3500]}\n```"


# =============================================================================
# Public entry point
# =============================================================================
def ask(question: str, session: PlanningSession, *, api_key: str | None = None, model: str | None = None,
        history: list | None = None, mock: bool | None = None) -> dict:
    run_id = "run-" + uuid.uuid4().hex[:8]
    t0 = time.time()
    steps, usage = [], {"llm_calls": 0, "tokens": 0}
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    mock = mock if mock is not None else (os.getenv("MOCK_LLM", "false").lower() == "true" or not api_key)
    try:
        if mock:
            answer = _mock_answer(question, session, steps, usage)
        else:
            answer = _gemini_answer(question, session, api_key, model or DEFAULT_MODEL, history or [], steps, usage)
        error = None
    except Exception as e:
        answer, error = f"Sorry - the assistant failed: {type(e).__name__}: {e}", str(e)
    tool_outputs = [s["output"] for s in steps if s["agent"] == "tool"]
    unverified = [] if mock else check_numbers(answer, tool_outputs, question)
    trace = {"run_id": run_id, "request": question, "mode": "mock" if mock else "gemini",
             "model": None if mock else (model or DEFAULT_MODEL),
             "steps": steps, "llm_calls": usage["llm_calls"], "tokens": usage["tokens"],
             "latency_ms": int((time.time() - t0) * 1000), "tools_used": [s["tool"] for s in steps if s["agent"] == "tool"],
             "faithfulness": {"unverified_numbers": unverified, "ok": not unverified}, "error": error,
             "answer": answer}
    TRACE_DIR.mkdir(exist_ok=True)
    (TRACE_DIR / f"{run_id}.json").write_text(json.dumps(trace, indent=2, default=str))
    return trace


if __name__ == "__main__":       # quick terminal test:  python agent.py "Why was order 77 built in Plant C?"
    import sys
    s = PlanningSession("data")
    out = ask(" ".join(sys.argv[1:]) or "Summarise the plan", s)
    print(out["answer"])
    print("\ntools:", out["tools_used"], "| llm calls:", out["llm_calls"], "| unverified:", out["faithfulness"])
