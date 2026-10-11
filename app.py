"""
app.py - Streamlit UI for the multi-plant order allocation assistant.

Run:   streamlit run app.py
Needs: allocator.py, tools.py, agent.py and a data/ folder with the 4 CSVs.
The Gemini key can be typed in the sidebar or put in a .env file as GEMINI_API_KEY=...
"""
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agent import ask, DEFAULT_MODEL
from tools import PlanningSession, run_tool
from allocator import summary_table

st.set_page_config(page_title="Order Allocation Assistant", page_icon="🏭", layout="wide")

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Gemini API key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
    model = st.text_input("Gemini model", value=DEFAULT_MODEL)
    mock = st.toggle("Mock mode (no LLM, raw tool output)", value=not bool(api_key))
    st.divider()
    data_dir = st.text_input("Data folder", value="data")
    start = st.text_input("Planning start (Monday)", value="2026-10-12")
    cost_w = st.slider("Weight on cost", 0.0, 1.0, 0.7, 0.05,
                       help="Carbon weight = 1 - cost weight. Default 0.7 / 0.3.")
    st.caption(f"Carbon weight: **{1 - cost_w:.2f}**")

    try:
        events_df = pd.read_csv(Path(data_dir) / "capacity_events.csv")
        ev_options = events_df["event_id"].astype(str).tolist()
    except Exception:
        ev_options = []
    base_events = st.multiselect("Capacity events applied to the base plan", ev_options, default=[],
                                 help="What-if questions always run on a copy; this only changes the base plan.")
    if st.button("Reload data / clear cache"):
        st.cache_resource.clear()
        st.rerun()


@st.cache_resource(show_spinner="Loading data and solving the base plan...")
def get_session(data_dir, start, cost_w, events):
    s = PlanningSession(data_dir, planning_start=start, cost_weight=cost_w,
                        carbon_weight=round(1 - cost_w, 4), base_event_ids=list(events))
    s.base_plan()          # solve once, cached
    return s


try:
    session = get_session(data_dir, start, cost_w, tuple(base_events))
except Exception as e:
    st.error(f"Could not load data from '{data_dir}': {e}")
    st.stop()

base = session.base_plan()
if not base.get("kpis"):
    st.error(f"Base plan is not valid: {base.get('message')}")
    st.stop()
k = base["kpis"]

# ----------------------------------------------------------------------------- header KPIs
st.title("🏭 Multi-Plant Order Allocation Assistant")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total landed cost", f"{k['total_cost']:,.0f}")
c2.metric("Total carbon", f"{k['total_carbon_t']:,.1f} t")
c3.metric("Orders assigned", f"{k['orders_assigned']} / {k['orders_total']}")
c4.metric("Late orders", k["late_orders"])
c5.metric("Plan valid", "✅" if base["plan_valid"] else "❌")

tab_chat, tab_plan, tab_trade, tab_trace = st.tabs(["💬 Ask", "📋 Plan", "📈 Trade-off", "🧾 Traces"])

# ----------------------------------------------------------------------------- chat
EXAMPLES = [
    "Allocate this week's orders and summarise the plan.",
    "What does a 10% carbon cut cost?",
    "What is the cheapest plan that is on time?",
    "Plant B loses 30% capacity. Re-plan.",
    "Why was order 77 built in Plant C?",
    "Which orders are late and why?",
    "What if a new order of 400 units of P1 to Germany is due on 2026-11-10?",
    "What if energy prices at Plant A rise by 25%?",
    "Write the brief for the leadership team.",
]

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.expander("Example questions", expanded=not st.session_state.messages):
        cols = st.columns(3)
        for i, q in enumerate(EXAMPLES):
            if cols[i % 3].button(q, key=f"ex{i}", use_container_width=True):
                st.session_state.pending = q

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["text"])
            if m.get("trace"):
                t = m["trace"]
                if t["faithfulness"]["unverified_numbers"]:
                    st.warning("Numbers not found in any tool result: " + ", ".join(t["faithfulness"]["unverified_numbers"]))
                with st.expander(f"Steps · tools: {', '.join(t['tools_used']) or 'none'} · "
                                 f"{t['llm_calls']} LLM calls · {t['tokens']} tokens · {t['latency_ms']} ms"):
                    for step in t["steps"]:
                        if step["agent"] == "tool":
                            st.markdown(f"**🔧 {step['tool']}** `{json.dumps(step['input'])}` · {step['ms']} ms")
                            st.json(step["output"], expanded=False)
                        else:
                            st.caption(f"LLM call · {step['ms']} ms")
                    st.caption(f"run_id: {t['run_id']}")

    question = st.chat_input("Ask about the plan, an order, or a what-if scenario...")
    question = question or st.session_state.pop("pending", None)
    if question:
        st.session_state.messages.append({"role": "user", "text": question})
        history = [{"role": "user" if m["role"] == "user" else "model", "text": m["text"]}
                   for m in st.session_state.messages[:-1]]
        with st.spinner("Thinking - running the optimiser..."):
            trace = ask(question, session, api_key=api_key or None, model=model, history=history, mock=mock)
        st.session_state.messages.append({"role": "assistant", "text": trace["answer"], "trace": trace})
        st.rerun()

    if st.session_state.messages and st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ----------------------------------------------------------------------------- plan table
with tab_plan:
    df = summary_table(base)
    f1, f2, f3, f4 = st.columns(4)
    plant_f = f1.multiselect("Plant", sorted(df["Assigned Plant"].unique()))
    prod_f = f2.multiselect("Product", sorted(df["Product"].unique()))
    dest_f = f3.multiselect("Destination", sorted(df["Destination"].unique()))
    late_only = f4.checkbox("Late / not allocated only")
    view = df
    if plant_f: view = view[view["Assigned Plant"].isin(plant_f)]
    if prod_f: view = view[view["Product"].isin(prod_f)]
    if dest_f: view = view[view["Destination"].isin(dest_f)]
    if late_only: view = view[~view["Delivery"].eq("On time")]
    st.dataframe(view, use_container_width=True, hide_index=True,
                 column_config={"Total Cost": st.column_config.NumberColumn(format="%d"),
                                "Carbon (kg)": st.column_config.NumberColumn(format="%d")})
    st.download_button("Download allocation_summary.csv", df.to_csv(index=False), "allocation_summary.csv", "text/csv")

    u1, u2 = st.columns(2)
    with u1:
        st.subheader("Plant utilisation")
        st.bar_chart(pd.Series(k["plant_utilisation"], name="utilisation"))
    with u2:
        st.subheader("Notices")
        notes = [n for n in base["notifications"]]
        st.caption(f"{sum(1 for n in notes if n.get('requires_user_choice'))} orders need a delivery decision")
        for n in notes[:30]:
            (st.warning if n.get("requires_user_choice") else st.info)(n["message"])

    with st.expander("Assumptions and limitations"):
        for a in base["assumptions"] + base["limitations"]:
            st.markdown(f"- {a}")

# ----------------------------------------------------------------------------- trade-off chart (R6 / R11)
with tab_trade:
    st.subheader("Cost vs carbon vs delivery")
    st.caption("Each point minimises cost while forcing carbon down by the given %. Labels show late orders.")
    levels = st.multiselect("Carbon cut levels (%)", [0, 5, 10, 15, 20, 25, 30, 40, 50], default=[0, 5, 10, 20, 30])
    if st.button("Compute trade-off", type="primary"):
        with st.spinner("Solving one plan per cut level..."):
            st.session_state.frontier = run_tool(session, "tradeoff_frontier", {"cut_levels": sorted(levels)})
    fr = st.session_state.get("frontier")
    if fr and "frontier" in fr:
        fdf = pd.DataFrame(fr["frontier"])
        ok = fdf[fdf["feasible"]]
        fig = px.scatter(ok, x="total_carbon_t", y="total_cost", size=ok["late_orders"] + 1,
                         color="late_orders", text=ok["carbon_cut_pct"].astype(str) + "% cut · " +
                         ok["late_orders"].astype(str) + " late",
                         labels={"total_carbon_t": "Total carbon (t)", "total_cost": "Total landed cost",
                                 "late_orders": "Late orders"}, color_continuous_scale="OrRd")
        fig.update_traces(textposition="top center", mode="markers+text+lines")
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fdf, hide_index=True, use_container_width=True)
        if (~fdf["feasible"]).any():
            st.warning("Some cut levels are infeasible with the available capacity.")

# ----------------------------------------------------------------------------- traces (R12)
with tab_trace:
    files = sorted(Path("traces").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        st.info("No runs yet. Ask a question first.")
    else:
        pick = st.selectbox("Run", [f.stem for f in files])
        st.json(json.loads((Path("traces") / f"{pick}.json").read_text()), expanded=2)
