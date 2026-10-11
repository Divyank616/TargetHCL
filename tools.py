"""
tools.py - predefined, deterministic functions the LLM can call.

Every number the assistant reports comes from one of these functions, which call the
PuLP optimiser in allocator.py. The LLM only chooses WHICH function to run and turns
the result into plain language.

SAFETY: the original CSV files are never written. Every what-if works on a deep copy of
the in-memory data (copy.deepcopy), so "what if Plant B loses 30%?" never changes the
real dataset or the base plan.

Usage:
    from tools import PlanningSession, TOOL_SPECS, run_tool
    s = PlanningSession("data", planning_start="2026-10-12")
    run_tool(s, "simulate_capacity_loss", {"plant": "B", "loss_pct": 30})
"""
from __future__ import annotations

import copy
from dataclasses import replace
from datetime import date, timedelta

import pandas as pd

from allocator import (Config, InputData, load_data, validate_data, solve_allocation,
                       compare_allocations, cheapest_plant_baseline, build_options,
                       resolve_horizon, effective_capacity, delivery_decisions,
                       add_order_and_reoptimize, MissingOrderInfo, summary_table)

MAX_LIST = 15   # keep tool outputs short so the LLM gets facts, not 150-row dumps


# =============================================================================
# Session: loads data once, caches the base plan, hands out copies for what-ifs
# =============================================================================
class PlanningSession:
    def __init__(self, data_dir="data", planning_start="2026-10-12", cost_weight=0.7,
                 carbon_weight=0.3, base_event_ids=()):
        self.cfg = Config(planning_start=date.fromisoformat(planning_start) if planning_start else None,
                          cost_weight=cost_weight, carbon_weight=carbon_weight)
        raw = load_data(data_dir)
        full = validate_data(raw, self.cfg)
        self.all_events = full.events.copy()               # every event in capacity_events.csv
        self.base_event_ids = list(base_event_ids)
        # the "base" world = original data + only the events the user chose to apply
        self.data = copy.deepcopy(full)
        self.data.events = full.events[full.events["event_id"].isin(self.base_event_ids)].reset_index(drop=True)
        self._base = None

    # -- helpers -------------------------------------------------------------
    def working_copy(self) -> InputData:
        """Deep copy for a what-if. Changing it never touches self.data or the CSV files."""
        return copy.deepcopy(self.data)

    def cfg_with(self, **kw) -> Config:
        return replace(self.cfg, **kw)

    def base_plan(self) -> dict:
        if self._base is None:
            self._base = solve_allocation(self.data, self.cfg)
        return self._base

    def solve(self, data=None, cfg=None, **kw) -> dict:
        return solve_allocation(data or self.data, cfg or self.cfg, **kw)

    def base_plants(self) -> dict:
        return {a["order_id"]: a["plant"] for a in self.base_plan()["allocation"] if a["plant"]}


# =============================================================================
# Compact views (what the LLM sees)
# =============================================================================
def _kpi_view(r: dict) -> dict:
    if not r.get("kpis"):
        return {"status": r.get("status"), "plan_valid": False, "message": r.get("message")}
    k = r["kpis"]
    late = [a for a in r["allocation"] if a.get("late_days")]
    unassigned = [a for a in r["allocation"] if a["status"] != "ASSIGNED"]
    return {
        "status": r["status"], "plan_valid": r["plan_valid"],
        "total_cost": k["total_cost"], "total_carbon_t": k["total_carbon_t"],
        "cost_breakdown": k["cost_breakdown"],
        "orders_assigned": k["orders_assigned"], "orders_total": k["orders_total"],
        "late_orders": k["late_orders"], "total_late_days": k["total_late_days"],
        "late_order_list": [{"order_id": a["order_id"], "plant": a["plant"], "late_days": a["late_days"]}
                            for a in late][:MAX_LIST],
        "unallocated_orders": [{"order_id": a["order_id"], "reason": a.get("reason")} for a in unassigned][:MAX_LIST],
        "plant_utilisation": k["plant_utilisation"], "orders_per_plant": k["plant_order_share"],
        "weights_used": k["weights_used"], "events_applied": r.get("events_applied", []),
        "capacity_checks_passed": r["validation"]["capacity_ok"],
        "eligibility_checks_passed": r["validation"]["eligibility_ok"],
    }


def _comparison_view(base: dict, new: dict) -> dict:
    if not new.get("kpis"):
        return {"status": new.get("status"), "plan_valid": False,
                "message": new.get("message", "No valid plan exists for this scenario.")}
    c = compare_allocations(base, new)
    moved = c["orders_moved"]
    return {
        "before": c["baseline"], "after": c["revised"], "change": c["delta"],
        "orders_moved_count": len(moved), "orders_moved": moved[:MAX_LIST],
        "orders_kept_same_plant": c["orders_same_plant"],
        "newly_late_orders": c["newly_late"][:MAX_LIST],
        "newly_unallocated_orders": c["newly_unassigned"][:MAX_LIST],
        "warnings": c["warnings"][:5],
        "new_plan_valid": new["plan_valid"], "new_plan_status": new["status"],
    }


def _horizon_dates(s: PlanningSession):
    start, weeks, _ = resolve_horizon(s.data, s.cfg)
    return start, start + timedelta(days=7 * len(weeks) - 1)


def _norm_order_id(s: PlanningSession, oid) -> str | None:
    """Accept 'O-077', 'o77', '77', 77."""
    ids = set(s.data.orders["order_id"])
    oid = str(oid).strip()
    if oid in ids:
        return oid
    digits = "".join(ch for ch in oid if ch.isdigit())
    for cand in ids:
        if digits and "".join(ch for ch in cand if ch.isdigit()).lstrip("0") == digits.lstrip("0"):
            return cand
    return None


def _norm_plant(s: PlanningSession, plant) -> str | None:
    p = str(plant).strip().upper().replace("PLANT", "").strip()
    return p if p in set(s.data.plants["plant"]) else None


# =============================================================================
# 1. LOOK-UP TOOLS (read-only)
# =============================================================================
def get_data_overview(s: PlanningSession) -> dict:
    d = s.data
    start, end = _horizon_dates(s)
    return {
        "orders": len(d.orders), "plants": sorted(d.plants["plant"].unique()),
        "products": sorted(d.orders["product"].unique()),
        "destinations": sorted(d.orders["destination"].unique()),
        "products_per_plant": d.plants.groupby("plant")["product"].apply(sorted).to_dict(),
        "weekly_capacity": {f"{r.plant}/{r.product}": int(r.weekly_capacity) for r in d.plants.itertuples()},
        "planning_window": f"{start} to {end}", "due_date_range": f"{d.orders['due'].min()} to {d.orders['due'].max()}",
        "capacity_events_in_file": s.all_events["event_id"].tolist(),
        "events_applied_to_base_plan": s.base_event_ids,
        "data_issues": [i["message"] for i in d.issues][:10], "notes": d.notes,
    }


def get_plan_summary(s: PlanningSession) -> dict:
    r = s.base_plan()
    out = _kpi_view(r)
    out["decisions_waiting"] = sum(1 for n in r["notifications"] if n.get("requires_user_choice"))
    naive = cheapest_plant_baseline(s.data, s.cfg)
    out["naive_cheapest_plant_rule"] = {k: naive[k] for k in ("total_cost", "total_carbon_t", "late_orders")}
    out["assumptions"] = r["assumptions"][:6]
    return out


def explain_order(s: PlanningSession, order_id) -> dict:
    """Why an order went to its plant: cost, carbon, delivery and capacity for every plant."""
    oid = _norm_order_id(s, order_id)
    if not oid:
        return {"error": f"order {order_id} not found"}
    r = s.base_plan()
    a = next(x for x in r["allocation"] if x["order_id"] == oid)
    if a["status"] != "ASSIGNED":
        return {"order_id": oid, "status": "NOT ALLOCATED", "reason": a.get("reason"), "reason_code": a.get("reason_code")}
    start, weeks, _ = resolve_horizon(s.data, s.cfg)
    cap, _ = effective_capacity(s.data, start, weeks, s.cfg)
    opts = build_options(s.data, s.cfg)
    dec = delivery_decisions(s.data, opts, cap, start, weeks, s.cfg)[oid]
    all_plants = sorted(s.data.plants["plant"].unique())
    options = []
    for o in dec["options"]:
        u = opts[oid][o["plant"]]["unit"]
        options.append({**{k: o[k] for k in ("plant", "landed_cost", "carbon_kg", "earliest_arrival", "late_days")},
                        "transit_days": u["transit_days"],
                        "cost_per_unit": {k: round(u[k], 3) for k in ("production", "energy", "transport", "duty")},
                        "chosen": o["plant"] == a["plant"]})
    cannot = [p for p in all_plants if p not in opts[oid]]
    cheapest = min(options, key=lambda o: o["landed_cost"])
    greenest = min(options, key=lambda o: o["carbon_kg"])
    on_time = [o for o in options if o["late_days"] == 0]
    reasons = []
    if a["plant"] == cheapest["plant"]:
        reasons.append("chosen plant is the cheapest capable plant")
    elif cheapest["late_days"]:
        reasons.append(f"cheapest plant {cheapest['plant']} would arrive {cheapest['late_days']} days late")
    if a["plant"] == greenest["plant"]:
        reasons.append("chosen plant has the lowest carbon")
    better_on_time = [o for o in on_time if o["landed_cost"] < a["landed_cost"] and o["carbon_kg"] < a["carbon_kg"]]
    if better_on_time:
        reasons.append(f"plant(s) {[o['plant'] for o in better_on_time]} are cheaper and greener and on time, "
                       f"but their capacity is used by other orders in the combined plan")
    if cannot:
        reasons.append(f"plant(s) {cannot} cannot make {a['product']} or have no route to {a['destination']}")
    return {"order_id": oid, "product": a["product"], "qty": a["qty"], "destination": a["destination"],
            "due": a["due"], "assigned_plant": a["plant"], "arrival": a["arrival"], "late_days": a["late_days"],
            "landed_cost": a["landed_cost"], "cost_breakdown": a["cost_breakdown"], "carbon_kg": a["carbon_kg"],
            "production_by_week": a["production_by_week"], "options_by_plant": options,
            "plants_that_cannot_make_it": cannot, "decision_type": dec["kind"], "reasons": reasons,
            "weights_used": {"cost": s.cfg.cost_weight, "carbon": s.cfg.carbon_weight}}


def list_late_orders(s: PlanningSession) -> dict:
    r = s.base_plan()
    by = {a["order_id"]: a for a in r["allocation"]}
    rows = []
    for n in r["notifications"]:
        if n["type"] in ("CHEAPEST_IS_LATE", "ALL_LATE", "LATE_DUE_TO_CAPACITY"):
            a = by[n["order_id"]]
            rows.append({"order_id": n["order_id"], "type": n["type"], "plan_plant": a.get("plant"),
                         "late_days_in_plan": a.get("late_days"), "needs_user_choice": n.get("requires_user_choice"),
                         "message": n["message"]})
    late_in_plan = [x for x in rows if x["late_days_in_plan"]]
    cheapest_late = [x for x in rows if x["type"] == "CHEAPEST_IS_LATE"]
    return {"late_orders_in_plan": len(late_in_plan), "orders_where_cheapest_plant_is_late": len(cheapest_late),
            "orders_late_at_every_plant": sum(1 for x in rows if x["type"] == "ALL_LATE"),
            "late_in_plan": late_in_plan[:MAX_LIST], "cheapest_is_late_examples": cheapest_late[:8]}


def list_unallocated_orders(s: PlanningSession) -> dict:
    r = s.base_plan()
    rows = [{"order_id": a["order_id"], "product": a["product"], "reason_code": a.get("reason_code"),
             "reason": a.get("reason")} for a in r["allocation"] if a["status"] != "ASSIGNED"]
    return {"count": len(rows), "orders": rows[:MAX_LIST], "notes": s.data.notes}


def list_orders(s: PlanningSession, plant=None, product=None, destination=None, only_late=False, limit=15) -> dict:
    df = summary_table(s.base_plan())
    if plant:
        df = df[df["Assigned Plant"].str.upper() == f"PLANT {str(plant).upper().replace('PLANT', '').strip()}"]
    if product:
        df = df[df["Product"].str.upper() == str(product).upper()]
    if destination:
        df = df[df["Destination"].str.lower() == str(destination).lower()]
    if only_late:
        df = df[df["Delivery"].str.startswith("Late")]
    return {"matching_orders": len(df), "total_cost": float(df["Total Cost"].fillna(0).sum()),
            "total_carbon_kg": float(df["Carbon (kg)"].fillna(0).sum()),
            "rows": df.head(int(limit or 15)).to_dict("records")}


def plant_utilisation(s: PlanningSession, plant=None) -> dict:
    r = s.base_plan()
    start, weeks, _ = resolve_horizon(s.data, s.cfg)
    cap = r["_cap"]
    used = {}
    for a in r["allocation"]:
        for w, n in (a.get("production_by_week") or {}).items():
            used[(a["plant"], a["product"], int(w))] = used.get((a["plant"], a["product"], int(w)), 0) + n
    rows = []
    for (p, prod, w), c in sorted(cap.items()):
        if plant and p != _norm_plant(s, plant):
            continue
        u = used.get((p, prod, w), 0)
        rows.append({"plant": p, "product": prod, "week": w, "week_start": str(start + timedelta(days=7 * (w - 1))),
                     "capacity": round(c, 1), "used": round(u, 1), "utilisation": round(u / c, 3) if c else None})
    full = [x for x in rows if x["utilisation"] and x["utilisation"] >= 0.99]
    return {"overall_utilisation": r["kpis"]["plant_utilisation"], "full_lines_and_weeks": full[:MAX_LIST],
            "detail": rows[:40] if plant else None}


def list_capacity_events(s: PlanningSession) -> dict:
    ev = s.all_events
    return {"events": [{"event_id": e.event_id, "plant": e.plant, "from": str(e.from_date), "to": str(e.to_date),
                        "capacity_loss_pct": round(e.loss_rate * 100, 1)} for e in ev.itertuples()],
            "applied_to_base_plan": s.base_event_ids}


# =============================================================================
# 2. OPTIMISATION VARIANTS (trade-offs, cheapest on time, weights)
# =============================================================================
def cheapest_on_time_plan(s: PlanningSession) -> dict:
    """Minimise cost only (carbon weight 0). Lateness still has priority over cost."""
    r = s.solve(cfg=s.cfg_with(cost_weight=1.0, carbon_weight=0.0))
    out = _comparison_view(s.base_plan(), r)
    out["explanation"] = ("Objective = minimise total landed cost only; the solver still minimises late orders "
                          "first. Orders still late cannot reach their due date from any plant with free capacity.")
    out["plan"] = _kpi_view(r)
    return out


def _cost_only_with_cap(s, cap_kg, data=None):
    return s.solve(data=data, cfg=s.cfg_with(cost_weight=1.0, carbon_weight=0.0, carbon_cap_kg=cap_kg))


def carbon_cut_cost(s: PlanningSession, cut_pct: float) -> dict:
    """What does an X% carbon cut cost? Reference = the cheapest plan; then force carbon <= (1-X%)."""
    ref = s.solve(cfg=s.cfg_with(cost_weight=1.0, carbon_weight=0.0))
    cap_kg = ref["kpis"]["total_carbon_kg"] * (1 - float(cut_pct) / 100)
    cut = _cost_only_with_cap(s, cap_kg)
    if not cut.get("kpis"):
        return {"cut_pct": cut_pct, "feasible": False,
                "message": f"No plan can cut carbon by {cut_pct}% with the available capacity."}
    return {"cut_pct": cut_pct, "feasible": True,
            "reference_cheapest_plan": {"total_cost": ref["kpis"]["total_cost"], "total_carbon_t": ref["kpis"]["total_carbon_t"],
                                        "late_orders": ref["kpis"]["late_orders"]},
            "greener_plan": {"total_cost": cut["kpis"]["total_cost"], "total_carbon_t": cut["kpis"]["total_carbon_t"],
                             "late_orders": cut["kpis"]["late_orders"]},
            "extra_cost": round(cut["kpis"]["total_cost"] - ref["kpis"]["total_cost"], 2),
            "extra_cost_pct": round(100 * (cut["kpis"]["total_cost"] / ref["kpis"]["total_cost"] - 1), 3),
            "carbon_saved_t": round(ref["kpis"]["total_carbon_t"] - cut["kpis"]["total_carbon_t"], 3),
            "cost_per_tonne_saved": round((cut["kpis"]["total_cost"] - ref["kpis"]["total_cost"]) /
                                          max(1e-9, ref["kpis"]["total_carbon_t"] - cut["kpis"]["total_carbon_t"]), 2),
            "extra_late_orders": cut["kpis"]["late_orders"] - ref["kpis"]["late_orders"],
            "orders_moved": len(compare_allocations(ref, cut)["orders_moved"])}


def tradeoff_frontier(s: PlanningSession, cut_levels=None) -> dict:
    """Cost / carbon / late orders for several carbon-cut levels (the trade-off table / chart data)."""
    cut_levels = cut_levels or [0, 5, 10, 20, 30, 40]
    ref = s.solve(cfg=s.cfg_with(cost_weight=1.0, carbon_weight=0.0))
    base_c = ref["kpis"]["total_carbon_kg"]
    rows = []
    for pct in cut_levels:
        r = ref if pct == 0 else _cost_only_with_cap(s, base_c * (1 - pct / 100))
        rows.append({"carbon_cut_pct": pct, "feasible": bool(r.get("kpis")),
                     "total_cost": r["kpis"]["total_cost"] if r.get("kpis") else None,
                     "total_carbon_t": r["kpis"]["total_carbon_t"] if r.get("kpis") else None,
                     "late_orders": r["kpis"]["late_orders"] if r.get("kpis") else None})
    return {"frontier": rows, "note": "Each row minimises cost subject to carbon <= (1 - cut%) x cheapest-plan carbon."}


def compare_weight_settings(s: PlanningSession, settings=None) -> dict:
    settings = settings or [[1.0, 0.0], [0.7, 0.3], [0.5, 0.5], [0.3, 0.7], [0.0, 1.0]]
    rows, plans = [], {}
    for wc, wk in settings:
        r = s.solve(cfg=s.cfg_with(cost_weight=float(wc), carbon_weight=float(wk)))
        plans[(wc, wk)] = {a["order_id"]: a["plant"] for a in r["allocation"]}
        rows.append({"cost_weight": wc, "carbon_weight": wk, "total_cost": r["kpis"].get("total_cost"),
                     "total_carbon_t": r["kpis"].get("total_carbon_t"), "late_orders": r["kpis"].get("late_orders"),
                     "orders_per_plant": r["kpis"].get("plant_order_share")})
    first = plans[tuple(settings[0])]
    for row, (wc, wk) in zip(rows, settings):
        row["orders_different_from_first_setting"] = sum(1 for k, v in plans[(wc, wk)].items() if first.get(k) != v)
    return {"settings": rows, "default": {"cost": s.cfg.cost_weight, "carbon": s.cfg.carbon_weight},
            "default_rationale": "0.7 cost / 0.3 carbon: cost stays the main driver, but carbon breaks near-ties "
                                 "toward greener plants. Both are normalised to their best possible totals."}


# =============================================================================
# 3. WHAT-IF SIMULATIONS (all on deep copies; original data untouched)
# =============================================================================
def simulate_capacity_loss(s: PlanningSession, plant, loss_pct, from_date=None, to_date=None) -> dict:
    p = _norm_plant(s, plant)
    if not p:
        return {"error": f"unknown plant {plant}", "known_plants": sorted(s.data.plants["plant"].unique())}
    loss = float(loss_pct)
    if not 0 < loss <= 100:
        return {"error": "loss_pct must be between 0 and 100"}
    h0, h1 = _horizon_dates(s)
    f = date.fromisoformat(from_date) if from_date else h0
    t = date.fromisoformat(to_date) if to_date else h1
    data = s.working_copy()
    new_ev = pd.DataFrame([{"event_id": "WHATIF", "plant": p, "from_date": f, "to_date": t,
                            "capacity_loss_pct": loss, "loss_rate": loss / 100}])
    data.events = pd.concat([data.events, new_ev], ignore_index=True)
    r = s.solve(data=data, baseline=s.base_plants())
    out = _comparison_view(s.base_plan(), r)
    out["scenario"] = {"plant": p, "loss_pct": loss, "from": str(f), "to": str(t),
                       "assumption": None if from_date else "no dates given: loss applied to the whole planning window"}
    return out


def simulate_existing_events(s: PlanningSession, event_ids) -> dict:
    ids = [event_ids] if isinstance(event_ids, str) else list(event_ids)
    unknown = set(ids) - set(s.all_events["event_id"])
    if unknown:
        return {"error": f"unknown events {sorted(unknown)}", "known": s.all_events["event_id"].tolist()}
    data = s.working_copy()
    data.events = s.all_events[s.all_events["event_id"].isin(set(ids) | set(s.base_event_ids))].reset_index(drop=True)
    r = s.solve(data=data, baseline=s.base_plants())
    out = _comparison_view(s.base_plan(), r)
    out["events_simulated"] = ids
    return out


def simulate_new_order(s: PlanningSession, product, qty, destination, due_date, order_id=None) -> dict:
    new = {"order_id": order_id or f"NEW-{len(s.data.orders) + 1:03d}", "product": str(product).upper(),
           "qty": int(qty), "destination": destination, "due": due_date}
    try:
        res = add_order_and_reoptimize(s.working_copy(), s.cfg, new)
    except MissingOrderInfo as e:
        return {"status": "clarification_needed", "message": str(e)}
    except ValueError as e:
        return {"status": "cannot_allocate", "message": str(e)}
    n = res["new_order_result"]
    return {"new_order": new, "assigned_plant": n.get("plant"), "status": n["status"],
            "arrival": n.get("arrival"), "late_days": n.get("late_days"), "landed_cost": n.get("landed_cost"),
            "carbon_kg": n.get("carbon_kg"), "reason": n.get("reason"),
            "impact_on_plan": _comparison_view(res["before"], res["after"])}


def simulate_order_change(s: PlanningSession, order_id, new_qty=None, new_due_date=None, new_destination=None) -> dict:
    oid = _norm_order_id(s, order_id)
    if not oid:
        return {"error": f"order {order_id} not found"}
    data = s.working_copy()
    i = data.orders.index[data.orders["order_id"] == oid][0]
    before = data.orders.loc[i, ["qty", "due", "destination"]].to_dict()
    if new_qty is not None:
        data.orders.at[i, "qty"] = int(new_qty)
    if new_due_date:
        data.orders.at[i, "due"] = date.fromisoformat(new_due_date)
    if new_destination:
        if (data.routes["destination"] == new_destination).sum() == 0:
            return {"error": f"no transport routes to {new_destination}"}
        data.orders.at[i, "destination"] = new_destination
    r = s.solve(data=data, baseline=s.base_plants())
    out = _comparison_view(s.base_plan(), r)
    a = next((x for x in r["allocation"] if x["order_id"] == oid), {})
    out["order_before"] = {k: str(v) for k, v in before.items()}
    out["order_after"] = {k: a.get(k) for k in ("qty", "due", "destination", "plant", "arrival", "late_days",
                                                 "landed_cost", "carbon_kg", "status")}
    return out


def simulate_force_assignment(s: PlanningSession, order_id, plant) -> dict:
    oid, p = _norm_order_id(s, order_id), _norm_plant(s, plant)
    if not oid or not p:
        return {"error": "unknown order or plant"}
    opts = build_options(s.data, s.cfg).get(oid, {})
    if p not in opts:
        return {"status": "refused", "message": f"Plant {p} cannot build {oid}: it does not make the product or has "
                                                f"no route to the destination. Eligible plants: {sorted(opts)}."}
    r = s.solve(fixed={oid: p}, baseline=s.base_plants())
    out = _comparison_view(s.base_plan(), r)
    a = next((x for x in r.get("allocation", []) if x["order_id"] == oid), {})
    out["forced_order"] = {k: a.get(k) for k in ("order_id", "plant", "arrival", "late_days", "landed_cost", "carbon_kg")}
    return out


PLANT_FIELDS = {"unit_cost", "energy_price_per_kwh", "energy_kwh_per_unit", "carbon_kg_per_kwh", "weekly_capacity"}
ROUTE_FIELDS = {"duty_pct", "transit_days", "transport_cost_per_unit", "carbon_kg_per_unit"}


def simulate_plant_parameter_change(s: PlanningSession, plant, field, pct_change, product=None) -> dict:
    """E.g. energy price +20% at Plant A, cleaner grid at Plant B (carbon -30%), capacity +10%."""
    p = _norm_plant(s, plant)
    if not p or field not in PLANT_FIELDS:
        return {"error": f"plant must be one of {sorted(s.data.plants['plant'].unique())}, field one of {sorted(PLANT_FIELDS)}"}
    data = s.working_copy()
    mask = data.plants["plant"] == p
    if product:
        mask &= data.plants["product"].str.upper() == str(product).upper()
    if not mask.any():
        return {"error": f"plant {p} has no row for product {product}"}
    old = data.plants.loc[mask, field].tolist()
    data.plants.loc[mask, field] = data.plants.loc[mask, field] * (1 + float(pct_change) / 100)
    r = s.solve(data=data, baseline=s.base_plants())
    out = _comparison_view(s.base_plan(), r)
    out["scenario"] = {"plant": p, "product": product or "all", "field": field, "pct_change": pct_change,
                       "old_values": old, "new_values": [round(x, 4) for x in data.plants.loc[mask, field]]}
    return out


def simulate_route_change(s: PlanningSession, plant, destination, field, new_value) -> dict:
    """E.g. new 20% duty on Plant A -> USA, or a port delay adding transit days."""
    p = _norm_plant(s, plant)
    if not p or field not in ROUTE_FIELDS:
        return {"error": f"field must be one of {sorted(ROUTE_FIELDS)}"}
    data = s.working_copy()
    mask = (data.routes["plant"] == p) & (data.routes["destination"].str.lower() == str(destination).lower())
    if not mask.any():
        return {"error": f"no route {p} -> {destination}"}
    old = float(data.routes.loc[mask, field].iloc[0])
    if field == "duty_pct":
        scale = (data.routes["duty_rate"] / data.routes["duty_pct"]).replace([float("inf")], 0).dropna()
        factor = scale.iloc[0] if len(scale) and scale.iloc[0] else 0.01
        data.routes.loc[mask, "duty_rate"] = float(new_value) * factor
    data.routes.loc[mask, field] = int(new_value) if field == "transit_days" else float(new_value)
    r = s.solve(data=data, baseline=s.base_plants())
    out = _comparison_view(s.base_plan(), r)
    out["scenario"] = {"route": f"{p} -> {destination}", "field": field, "old_value": old, "new_value": new_value}
    return out


def get_brief_facts(s: PlanningSession) -> dict:
    """Everything a management brief needs, all computed. The brief must use only these numbers."""
    base = get_plan_summary(s)
    late = list_late_orders(s)
    cut10 = carbon_cut_cost(s, 10)
    facts = {"plan": base, "late_and_decisions": {k: late[k] for k in
             ("late_orders_in_plan", "orders_where_cheapest_plant_is_late", "orders_late_at_every_plant")},
             "carbon_cut_10pct": cut10, "capacity_events_available": s.all_events["event_id"].tolist()}
    if s.all_events is not None and len(s.all_events):
        first = s.all_events["event_id"].iloc[0]
        if first not in s.base_event_ids:
            facts["disruption_example"] = {"event": first, **{k: v for k, v in simulate_existing_events(s, [first]).items()
                                                               if k in ("change", "orders_moved_count", "newly_late_orders")}}
    return facts


# =============================================================================
# Registry: name -> (function, Gemini declaration)
# =============================================================================
def _obj(props: dict, required=()):
    return {"type": "OBJECT", "properties": props, "required": list(required)}


S, N, I, B = {"type": "STRING"}, {"type": "NUMBER"}, {"type": "INTEGER"}, {"type": "BOOLEAN"}

TOOLS = {
    "get_data_overview": (get_data_overview, "Overview of the loaded data: plants, products each plant can make, "
                          "weekly capacities, destinations, planning window, capacity events and data issues.", _obj({})),
    "get_plan_summary": (get_plan_summary, "Allocate all orders with the current weights and return total cost, "
                         "carbon, late orders, unallocated orders, plant utilisation, and the naive cheapest-plant "
                         "rule for comparison. Use for 'allocate the orders', 'what is the plan', KPIs.", _obj({})),
    "explain_order": (explain_order, "Explain why one order was assigned to its plant: cost breakdown, carbon, "
                      "arrival vs due date, and every alternative plant. Use for 'why was order 77 built in Plant C'.",
                      _obj({"order_id": {**S, "description": "e.g. O-077 or 77"}}, ["order_id"])),
    "list_late_orders": (list_late_orders, "Orders that are late in the plan, orders where the cheapest plant would "
                         "be late, and orders that are late at every plant (need a user decision).", _obj({})),
    "list_unallocated_orders": (list_unallocated_orders, "Orders that could not be allocated and why "
                                "(no capable plant, no route, no capacity).", _obj({})),
    "list_orders": (list_orders, "Filter the current plan by plant, product, destination, or late only; returns "
                    "matching rows plus their total cost and carbon.",
                    _obj({"plant": S, "product": S, "destination": S, "only_late": B, "limit": I})),
    "plant_utilisation": (plant_utilisation, "Capacity used vs available per plant (optionally per product and "
                          "week); shows which lines are full.", _obj({"plant": S})),
    "list_capacity_events": (list_capacity_events, "List capacity-loss events in capacity_events.csv.", _obj({})),
    "cheapest_on_time_plan": (cheapest_on_time_plan, "The cheapest plan that keeps orders on time where possible "
                              "(cost-only objective) and how it differs from the current plan.", _obj({})),
    "carbon_cut_cost": (carbon_cut_cost, "What does an X% carbon cut cost? Extra cost, carbon saved, cost per tonne, "
                        "and delivery impact.", _obj({"cut_pct": {**N, "description": "e.g. 10 for 10%"}}, ["cut_pct"])),
    "tradeoff_frontier": (tradeoff_frontier, "Trade-off table of cost, carbon and late orders for several carbon-cut "
                          "levels.", _obj({"cut_levels": {"type": "ARRAY", "items": N}})),
    "compare_weight_settings": (compare_weight_settings, "Compare plans under different cost/carbon weights "
                                "(each item [cost_weight, carbon_weight]).",
                                _obj({"settings": {"type": "ARRAY", "items": {"type": "ARRAY", "items": N}}})),
    "simulate_capacity_loss": (simulate_capacity_loss, "WHAT-IF: a plant loses X% capacity (optionally between "
                               "dates). Re-plans on a copy and reports cost/carbon/late changes and which orders moved.",
                               _obj({"plant": S, "loss_pct": N, "from_date": {**S, "description": "YYYY-MM-DD"},
                                     "to_date": {**S, "description": "YYYY-MM-DD"}}, ["plant", "loss_pct"])),
    "simulate_existing_events": (simulate_existing_events, "WHAT-IF: apply events from capacity_events.csv "
                                 "(e.g. EV-01) and compare with the current plan.",
                                 _obj({"event_ids": {"type": "ARRAY", "items": S}}, ["event_ids"])),
    "simulate_new_order": (simulate_new_order, "WHAT-IF: a new order arrives. Re-optimises all orders together "
                           "and reports where it goes and what else moves.",
                           _obj({"product": S, "qty": I, "destination": S, "due_date": {**S, "description": "YYYY-MM-DD"}},
                                ["product", "qty", "destination", "due_date"])),
    "simulate_order_change": (simulate_order_change, "WHAT-IF: change an existing order's quantity, due date or "
                              "destination.", _obj({"order_id": S, "new_qty": I, "new_due_date": S,
                                                    "new_destination": S}, ["order_id"])),
    "simulate_force_assignment": (simulate_force_assignment, "WHAT-IF: force an order to a specific plant. Refuses "
                                  "if the plant cannot make the product.", _obj({"order_id": S, "plant": S},
                                                                                 ["order_id", "plant"])),
    "simulate_plant_parameter_change": (simulate_plant_parameter_change, "WHAT-IF: change a plant parameter by a "
                                        "percentage: unit_cost, energy_price_per_kwh, energy_kwh_per_unit, "
                                        "carbon_kg_per_kwh or weekly_capacity (optionally one product).",
                                        _obj({"plant": S, "field": S, "pct_change": N, "product": S},
                                             ["plant", "field", "pct_change"])),
    "simulate_route_change": (simulate_route_change, "WHAT-IF: set a new value for a plant->destination route: "
                              "duty_pct, transit_days, transport_cost_per_unit or carbon_kg_per_unit.",
                              _obj({"plant": S, "destination": S, "field": S, "new_value": N},
                                   ["plant", "destination", "field", "new_value"])),
    "get_brief_facts": (get_brief_facts, "All computed numbers for a management brief (plan KPIs, late orders, "
                        "10% carbon-cut cost, a disruption example). Call this before writing any brief.", _obj({})),
}


def run_tool(session: PlanningSession, name: str, args: dict | None) -> dict:
    if name not in TOOLS:
        return {"error": f"unknown tool {name}"}
    fn = TOOLS[name][0]
    try:
        return _jsonable(fn(session, **(args or {})))
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:  # report, never crash the chat
        return {"error": f"{type(e).__name__}: {e}"}


def tool_declarations() -> list[dict]:
    return [{"name": n, "description": d, "parameters": p} for n, (_, d, p) in TOOLS.items()]


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (date,)):
        return str(x)
    if hasattr(x, "item"):          # numpy scalars
        return x.item()
    if isinstance(x, float) and x != x:
        return None
    return x
