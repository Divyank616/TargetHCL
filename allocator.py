"""
Combined multi-plant order allocator (cost + carbon + capacity + delivery).

One PuLP mixed-integer model allocates ALL orders together, so orders compete
for the same plant capacity instead of being assigned one at a time.

Run:
    python allocator.py --data ./data --start 2026-10-12
    python allocator.py --data ./data --start 2026-10-12 --interactive
    python allocator.py --data ./data --start 2026-10-12 --events EV-01 --weights 0.5 0.5

Schema (Annex B of the participant guide, extra columns allowed):
    orders.csv            order_id, product, qty, destination, due
    plants.csv            plant, product, weekly_capacity, unit_cost,
                          energy_kwh_per_unit, carbon_kg_per_kwh
                          (+ optional energy_price_per_kwh)
    transport_duties.csv  plant, destination, transport_cost_per_unit,
                          transit_days, duty_pct, carbon_kg_per_unit
    capacity_events.csv   event_id, plant, from_date, to_date, capacity_loss_pct

Sections:
    1. Config             6. Decision options (cheapest-is-late / all-late)
    2. Loading            7. Combined MILP
    3. Validation         8. Independent validation + KPIs
    4. Effective capacity 9. Scenarios (capacity loss, new order, weights)
    5. Cost & carbon     10. CLI
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pulp


# =============================================================================
# 1. CONFIG — every assumption lives here and is reported with each result
# =============================================================================
@dataclass
class Config:
    planning_start: date | None = None   # Monday of week 1. None -> derived (see resolve_horizon)
    horizon_weeks: int | None = None     # None -> enough weeks to reach the latest due date
    ship_day_offset: int = 4             # order ships on day N of its completion week (4 = Friday)
    duty_base: str = "exworks"           # "production" | "exworks" (production+energy) | "exworks_plus_transport"
    default_energy_price: float | None = None  # used only if plants.csv has no energy_price_per_kwh
    event_overlap_rule: str = "sum_capped"     # overlapping events on one plant/day: "sum_capped" | "max"
    cost_weight: float = 0.7
    carbon_weight: float = 0.3
    carbon_cap_kg: float | None = None   # optional hard cap on total carbon
    deadline_mode: str = "soft"          # "soft": late allowed but penalised & flagged; "hard": late options removed
    late_order_penalty: float = 10.0     # per late order (normalised units; dominates cost/carbon)
    late_day_penalty: float = 0.1        # per day late
    unserved_penalty: float = 1000.0     # per order that cannot be produced at all
    move_penalty: float = 0.002          # re-plan stability: per order that changes plant vs baseline
    solver: str = "auto"                 # "auto" (HiGHS if installed, else CBC) | "highs" | "cbc"
    solver_time_limit: int = 120
    solver_gap: float = 0.0              # relative MIP gap (0 = prove optimality; penalties make the objective large, so a relative gap would hide real cost/carbon differences)

    def check(self):
        if self.cost_weight < 0 or self.carbon_weight < 0 or self.cost_weight + self.carbon_weight == 0:
            raise ValueError("weights must be >= 0 and not both zero")
        if self.duty_base not in ("production", "exworks", "exworks_plus_transport"):
            raise ValueError(f"unknown duty_base {self.duty_base}")
        if self.event_overlap_rule not in ("sum_capped", "max"):
            raise ValueError(f"unknown event_overlap_rule {self.event_overlap_rule}")
        if self.deadline_mode not in ("soft", "hard"):
            raise ValueError(f"unknown deadline_mode {self.deadline_mode}")


REQUIRED_COLUMNS = {
    "orders": ["order_id", "product", "qty", "destination", "due"],
    "plants": ["plant", "product", "weekly_capacity", "unit_cost",
               "energy_kwh_per_unit", "carbon_kg_per_kwh"],
    "transport_duties": ["plant", "destination", "transport_cost_per_unit",
                         "transit_days", "duty_pct", "carbon_kg_per_unit"],
    "capacity_events": ["event_id", "plant", "from_date", "to_date", "capacity_loss_pct"],
}


class DataValidationError(Exception):
    """Raised when the input cannot produce a trustworthy allocation."""
    def __init__(self, issues):
        self.issues = issues
        msgs = "\n".join(f"  [{i['table']}] {i['message']}" for i in issues if i["severity"] == "ERROR")
        super().__init__("Input data failed validation:\n" + msgs)


def _issue(severity, table, message, row=None, column=None):
    return dict(severity=severity, table=table, message=message, row=row, column=column)


# =============================================================================
# 2. LOADING
# =============================================================================
@dataclass
class InputData:
    orders: pd.DataFrame
    plants: pd.DataFrame
    routes: pd.DataFrame
    events: pd.DataFrame
    issues: list = field(default_factory=list)
    order_status: dict = field(default_factory=dict)   # order_id -> (code, reason) for orders excluded from the model
    notes: list = field(default_factory=list)          # human-readable notes (e.g. product nobody makes)


def _read_csv(path: Path) -> pd.DataFrame:
    # Read everything as text first: we convert and validate ourselves so that
    # bad values are reported instead of silently coerced (e.g. "077" -> 77).
    df = pd.read_csv(path, dtype=str, keep_default_na=False, skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


def load_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    raw = {}
    for name in REQUIRED_COLUMNS:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            if name == "capacity_events":   # events are optional: no file = no disruption
                raw[name] = pd.DataFrame(columns=REQUIRED_COLUMNS[name])
                continue
            raise FileNotFoundError(f"missing required file {path}")
        raw[name] = _read_csv(path)
    return raw


def inspect_data(raw: dict[str, pd.DataFrame], n=3) -> str:
    """Schema summary: columns, inferred types, row counts, sample rows."""
    out = []
    for name, df in raw.items():
        out.append(f"\n=== {name}.csv  ({len(df)} rows)")
        for c in df.columns:
            num = pd.to_numeric(df[c], errors="coerce")
            kind = "numeric" if len(df) and num.notna().all() else "text"
            if kind == "text" and len(df) and pd.to_datetime(df[c], errors="coerce", format="mixed").notna().all():
                kind = "date"
            req = "required" if c in REQUIRED_COLUMNS[name] else "extra"
            out.append(f"  {c:<26} {kind:<8} {req}")
        out.append(df.head(n).to_string(index=False))
    return "\n".join(out)


# =============================================================================
# 3. VALIDATION
# =============================================================================
def _to_num(df, col, table, issues, *, integer=False, min_value=None, strict_positive=False):
    vals = pd.to_numeric(df[col].replace("", None), errors="coerce")
    bad = vals.isna()
    if integer:
        bad |= vals.notna() & (vals != vals.round())
    if min_value is not None:
        bad |= vals < min_value
    if strict_positive:
        bad |= vals <= 0
    for idx in df.index[bad]:
        issues.append(_issue("ERROR", table, f"invalid {col}={df.at[idx, col]!r}", row=int(idx), column=col))
    return vals, bad


def _to_date(df, col, table, issues):
    vals = pd.to_datetime(df[col].replace("", None), errors="coerce", format="mixed")
    bad = vals.isna()
    for idx in df.index[bad]:
        issues.append(_issue("ERROR", table, f"invalid date {col}={df.at[idx, col]!r}", row=int(idx), column=col))
    return vals.dt.date, bad


def _pct_to_rate(series, table, col, issues):
    """Accept 30 (percent) or 0.30 (fraction); decide per column, never per row."""
    s = series.dropna()
    if len(s) and s.max() <= 1.0 and (s > 0).any():
        issues.append(_issue("WARNING", table, f"{col}: all values <= 1, interpreted as fractions (0.3 = 30%)"))
        return series
    if (s > 100).any():
        issues.append(_issue("ERROR", table, f"{col}: values above 100%"))
    return series / 100.0


def validate_data(raw: dict[str, pd.DataFrame], cfg: Config) -> InputData:
    issues = []
    # ---- required columns (fatal if missing)
    for name, cols in REQUIRED_COLUMNS.items():
        missing = [c for c in cols if c not in raw[name].columns]
        if missing:
            issues.append(_issue("ERROR", name, f"missing required columns {missing}"))
    if any(i["severity"] == "ERROR" for i in issues):
        raise DataValidationError(issues)

    # ---- plants: one row per (plant, product)
    plants = raw["plants"].copy()
    plants["weekly_capacity"], b1 = _to_num(plants, "weekly_capacity", "plants", issues, strict_positive=True)
    plants["unit_cost"], b2 = _to_num(plants, "unit_cost", "plants", issues, min_value=0)
    plants["energy_kwh_per_unit"], b3 = _to_num(plants, "energy_kwh_per_unit", "plants", issues, min_value=0)
    plants["carbon_kg_per_kwh"], b4 = _to_num(plants, "carbon_kg_per_kwh", "plants", issues, min_value=0)
    bad_plant = b1 | b2 | b3 | b4 | (plants["plant"] == "") | (plants["product"] == "")
    if "energy_price_per_kwh" in plants.columns:
        plants["energy_price_per_kwh"], b5 = _to_num(plants, "energy_price_per_kwh", "plants", issues, min_value=0)
        bad_plant |= b5
    elif cfg.default_energy_price is not None:
        plants["energy_price_per_kwh"] = float(cfg.default_energy_price)
        issues.append(_issue("WARNING", "plants", f"no energy_price_per_kwh column; using configured "
                                                  f"{cfg.default_energy_price}/kWh for every plant"))
    else:
        plants["energy_price_per_kwh"] = 0.0
        issues.append(_issue("WARNING", "plants", "no energy_price_per_kwh column and no default given: "
                                                  "energy cost = 0, unit_cost assumed to already include energy"))
    for idx in plants.index[bad_plant]:
        issues.append(_issue("ERROR", "plants", f"row {idx} excluded: plant/product "
                             f"{plants.at[idx, 'plant']}/{plants.at[idx, 'product']} has invalid values", row=int(idx)))
    plants = plants[~bad_plant]
    dup = plants.duplicated(["plant", "product"], keep=False)
    if dup.any():
        pairs = sorted(set(map(tuple, plants.loc[dup, ["plant", "product"]].values)))
        issues.append(_issue("ERROR", "plants", f"duplicate plant/product rows (ambiguous capacity/cost): {pairs}"))

    # ---- routes: one row per (plant, destination)
    routes = raw["transport_duties"].copy()
    routes["transport_cost_per_unit"], r1 = _to_num(routes, "transport_cost_per_unit", "transport_duties", issues, min_value=0)
    routes["transit_days"], r2 = _to_num(routes, "transit_days", "transport_duties", issues, integer=True, min_value=0)
    routes["duty_pct"], r3 = _to_num(routes, "duty_pct", "transport_duties", issues, min_value=0)
    routes["carbon_kg_per_unit"], r4 = _to_num(routes, "carbon_kg_per_unit", "transport_duties", issues, min_value=0)
    bad_route = r1 | r2 | r3 | r4
    for idx in routes.index[bad_route]:
        issues.append(_issue("ERROR", "transport_duties", f"route row {idx} excluded "
                             f"({routes.at[idx, 'plant']}->{routes.at[idx, 'destination']})", row=int(idx)))
    routes = routes[~bad_route].copy()
    routes["transit_days"] = routes["transit_days"].astype(int)
    routes["duty_rate"] = _pct_to_rate(routes["duty_pct"], "transport_duties", "duty_pct", issues)
    dup = routes.duplicated(["plant", "destination"], keep=False)
    if dup.any():
        pairs = sorted(set(map(tuple, routes.loc[dup, ["plant", "destination"]].values)))
        issues.append(_issue("ERROR", "transport_duties", f"duplicate plant/destination routes: {pairs}"))
    unknown = set(routes["plant"]) - set(plants["plant"])
    if unknown:
        issues.append(_issue("WARNING", "transport_duties", f"routes for plants not in plants.csv ignored: {sorted(unknown)}"))

    # ---- orders: never dropped silently; bad rows get status INVALID_INPUT
    orders = raw["orders"].copy()
    order_status, notes = {}, []
    if (orders["order_id"] == "").any():
        issues.append(_issue("ERROR", "orders", "blank order_id present"))
    dup_ids = orders.loc[orders["order_id"].duplicated(keep=False), "order_id"].unique().tolist()
    if dup_ids:
        issues.append(_issue("ERROR", "orders", f"duplicate order_id values: {dup_ids}"))
    orders["qty"], q_bad = _to_num(orders, "qty", "orders", issues, integer=True, strict_positive=True)
    orders["due"], d_bad = _to_date(orders, "due", "orders", issues)
    for idx in orders.index[q_bad | d_bad | (orders["product"] == "") | (orders["destination"] == "")]:
        oid = orders.at[idx, "order_id"]
        order_status[oid] = ("INVALID_INPUT", "order row has missing/invalid qty, due, product or destination")

    # capability + route checks per order
    capable = plants.groupby("product")["plant"].apply(set).to_dict()
    route_set = set(zip(routes["plant"], routes["destination"]))
    for r in orders.itertuples():
        if r.order_id in order_status:
            continue
        makers = capable.get(r.product, set())
        if not makers:
            order_status[r.order_id] = ("NO_CAPABLE_PLANT", f"product {r.product} cannot be made in any plant")
            continue
        reachable = {p for p in makers if (p, r.destination) in route_set}
        if not reachable:
            order_status[r.order_id] = ("NO_ROUTE", f"plants {sorted(makers)} make {r.product} but none has a "
                                                    f"transport row to {r.destination}")
        elif reachable != makers:
            issues.append(_issue("WARNING", "transport_duties",
                                 f"{r.order_id}: no route from {sorted(makers - reachable)} to {r.destination}; "
                                 f"those plants excluded for this order"))
    for prod in sorted(set(orders["product"]) - set(capable)):
        ids = orders.loc[orders["product"] == prod, "order_id"].tolist()
        notes.append(f"Product {prod} cannot be made in any plant (orders {ids}). These orders are not allocated.")
    if (orders["qty"].dropna() <= 0).any():
        pass  # already reported by _to_num

    # ---- events
    events = raw["capacity_events"].copy()
    if len(events):
        events["from_date"], e1 = _to_date(events, "from_date", "capacity_events", issues)
        events["to_date"], e2 = _to_date(events, "to_date", "capacity_events", issues)
        events["capacity_loss_pct"], e3 = _to_num(events, "capacity_loss_pct", "capacity_events", issues, min_value=0)
        bad_ev = e1 | e2 | e3
        bad_ev |= (~bad_ev) & (events["from_date"] > events["to_date"])
        for idx in events.index[bad_ev]:
            issues.append(_issue("ERROR", "capacity_events", f"event row {idx} ({events.at[idx, 'event_id']}) "
                                 f"excluded: invalid dates or loss", row=int(idx)))
        events = events[~bad_ev].copy()
        events["loss_rate"] = _pct_to_rate(events["capacity_loss_pct"], "capacity_events", "capacity_loss_pct", issues)
        # identical duplicate rows -> keep once; conflicting rows with same id -> error
        key = ["event_id", "plant", "from_date", "to_date", "loss_rate"]
        exact_dups = events.duplicated(key)
        if exact_dups.any():
            issues.append(_issue("WARNING", "capacity_events", f"identical duplicate events removed: "
                                 f"{events.loc[exact_dups, 'event_id'].tolist()}"))
            events = events[~exact_dups]
        conflict = events["event_id"].duplicated(keep=False)
        if conflict.any():
            issues.append(_issue("ERROR", "capacity_events", f"same event_id with different contents: "
                                 f"{sorted(events.loc[conflict, 'event_id'].unique())}"))
        unknown = set(events["plant"]) - set(plants["plant"])
        if unknown:
            issues.append(_issue("WARNING", "capacity_events", f"events for unknown plants ignored: {sorted(unknown)}"))
            events = events[events["plant"].isin(plants["plant"])]
    else:
        events = pd.DataFrame(columns=REQUIRED_COLUMNS["capacity_events"] + ["loss_rate"])

    fatal = [i for i in issues if i["severity"] == "ERROR" and i["table"] != "orders" and i["row"] is None]
    fatal += [i for i in issues if i["table"] == "orders" and ("duplicate" in i["message"] or "blank" in i["message"])]
    if fatal:
        raise DataValidationError(issues)
    return InputData(orders=orders.reset_index(drop=True), plants=plants.reset_index(drop=True),
                     routes=routes.reset_index(drop=True), events=events.reset_index(drop=True),
                     issues=issues, order_status=order_status, notes=notes)


# =============================================================================
# 4. PLANNING HORIZON + EFFECTIVE CAPACITY (original capacity never modified)
# =============================================================================
def resolve_horizon(data: InputData, cfg: Config):
    valid = data.orders[~data.orders["order_id"].isin(data.order_status)]
    if cfg.planning_start is not None:
        start = cfg.planning_start
        start_note = f"planning start {start} (given)"
    else:
        # No start date in the data pack: use the Monday on/before (earliest due - 7 days).
        base = (valid["due"].min() if len(valid) else date.today()) - timedelta(days=7)
        start = base - timedelta(days=base.weekday())
        start_note = f"planning start {start} DERIVED (Monday on/before earliest due - 7d); pass --start to override"
    if cfg.horizon_weeks:
        weeks = cfg.horizon_weeks
    else:
        last_due = valid["due"].max() if len(valid) else start
        weeks = max(1, math.ceil(((last_due - start).days + 1) / 7))
    return start, list(range(1, weeks + 1)), start_note


def week_start(start: date, w: int) -> date:
    return start + timedelta(days=7 * (w - 1))


def ship_date(start: date, w: int, cfg: Config) -> date:
    return week_start(start, w) + timedelta(days=cfg.ship_day_offset)


def daily_loss(plant: str, day: date, events: pd.DataFrame, rule: str) -> tuple[float, list]:
    """Combined capacity loss for one plant on one day (each event counted once)."""
    active = events[(events["plant"] == plant) & (events["from_date"] <= day) & (events["to_date"] >= day)]
    if active.empty:
        return 0.0, []
    rates = active.drop_duplicates("event_id")["loss_rate"].tolist()
    loss = max(rates) if rule == "max" else min(1.0, sum(rates))
    return loss, active["event_id"].tolist()


def effective_capacity(data: InputData, start: date, weeks: list[int], cfg: Config,
                       events: pd.DataFrame | None = None):
    """capacity[(plant, product, week)] = weekly_capacity x (1 - average daily loss in that week).
    A partial-week event reduces only the affected days' share."""
    events = data.events if events is None else events
    cap, log = {}, []
    for p, grp in data.plants.groupby("plant"):
        for w in weeks:
            days = [week_start(start, w) + timedelta(days=d) for d in range(7)]
            losses, ids = zip(*(daily_loss(p, d, events, cfg.event_overlap_rule) for d in days))
            factor = 1.0 - sum(losses) / 7.0
            ev_ids = sorted({e for lst in ids for e in lst})
            if ev_ids:
                log.append(dict(plant=p, week=w, week_start=str(days[0]), events=ev_ids,
                                capacity_factor=round(factor, 4)))
            for r in grp.itertuples():
                cap[(p, r.product, w)] = float(r.weekly_capacity) * factor
    return cap, log


# =============================================================================
# 5. COST & CARBON (per unit; multiplied by qty)
# =============================================================================
def unit_cost_and_carbon(plant_row, route_row, cfg: Config) -> dict:
    production = plant_row.unit_cost
    energy = plant_row.energy_kwh_per_unit * plant_row.energy_price_per_kwh
    transport = route_row.transport_cost_per_unit
    base = {"production": production,
            "exworks": production + energy,
            "exworks_plus_transport": production + energy + transport}[cfg.duty_base]
    duty = route_row.duty_rate * base
    mfg_carbon = plant_row.energy_kwh_per_unit * plant_row.carbon_kg_per_kwh
    return dict(production=production, energy=energy, transport=transport, duty=duty,
                cost=production + energy + transport + duty,
                carbon_mfg=mfg_carbon, carbon_transport=route_row.carbon_kg_per_unit,
                carbon=mfg_carbon + route_row.carbon_kg_per_unit, transit_days=int(route_row.transit_days))


def build_options(data: InputData, cfg: Config) -> dict:
    """options[order_id][plant] = per-order cost/carbon/transit for every eligible plant."""
    P = {(r.plant, r.product): r for r in data.plants.itertuples()}
    R = {(r.plant, r.destination): r for r in data.routes.itertuples()}
    opts = {}
    for o in data.orders.itertuples():
        if o.order_id in data.order_status:
            continue
        per_plant = {}
        for (p, prod), prow in P.items():
            if prod != o.product or (p, o.destination) not in R:
                continue
            u = unit_cost_and_carbon(prow, R[(p, o.destination)], cfg)
            per_plant[p] = dict(unit=u, cost=u["cost"] * o.qty, carbon=u["carbon"] * o.qty,
                                transit_days=u["transit_days"])
        opts[o.order_id] = per_plant
    return opts


def earliest_arrival(qty, plant, product, transit, cap, start, weeks, cfg):
    """Best case if this order had the plant to itself: first week by which qty can be built."""
    cum = 0.0
    for w in weeks:
        c = cap.get((plant, product, w), 0.0)
        cum += c
        if cum >= qty - 1e-9:
            return w, ship_date(start, w, cfg) + timedelta(days=transit)
    return None, None


# =============================================================================
# 6. DECISION OPTIONS: cheapest plant late? all plants late?
# =============================================================================
def delivery_decisions(data, opts, cap, start, weeks, cfg) -> dict:
    """For every order, compare the cheapest plant with on-time alternatives (best-case timing)."""
    orders = data.orders.set_index("order_id")
    out = {}
    for oid, per_plant in opts.items():
        o = orders.loc[oid]
        rows = []
        for p, v in per_plant.items():
            wk, arr = earliest_arrival(o.qty, p, o["product"], v["transit_days"], cap, start, weeks, cfg)
            late = None if arr is None else max(0, (arr - o.due).days)
            rows.append(dict(plant=p, landed_cost=round(v["cost"], 2), carbon_kg=round(v["carbon"], 2),
                             earliest_arrival=str(arr) if arr else None,
                             late_days=late, on_time=(late == 0), producible_in_horizon=arr is not None))
        rows.sort(key=lambda r: (r["landed_cost"], r["plant"]))
        producible = [r for r in rows if r["producible_in_horizon"]]
        cheapest = producible[0] if producible else None
        on_time = [r for r in producible if r["on_time"]]
        if not producible:
            kind = "NOT_PRODUCIBLE_IN_HORIZON"
        elif cheapest["on_time"]:
            kind = "OK"
        elif on_time:
            kind = "CHEAPEST_IS_LATE"
        else:
            kind = "ALL_LATE"
        out[oid] = dict(kind=kind, options=rows, cheapest=cheapest,
                        best_on_time=on_time[0] if on_time else None)
    return out


# =============================================================================
# 7. COMBINED MILP
# =============================================================================
def deadline_week(due: date, transit: int, start: date, weeks: list[int], cfg: Config) -> int:
    """Last week whose ship date + transit arrives on/before due (0 = no week is early enough)."""
    ok = [w for w in weeks if ship_date(start, w, cfg) + timedelta(days=transit) <= due]
    return max(ok) if ok else 0


def edf_schedule(assigned: dict, cap: dict, weeks: list[int]) -> dict:
    """
    Turn 'finish by week w' decisions into a weekly production plan for ONE plant-product line.
    assigned = {order_id: (qty, finish_by_week)}. Earliest-deadline-first: orders with the earliest
    finish-by week are built first, filling each week's capacity before moving to the next.
    Returns {order_id: {week: units}}.
    """
    remaining = {w: cap[w] for w in weeks}
    plan = {}
    for oid, (qty, _) in sorted(assigned.items(), key=lambda kv: (kv[1][1], kv[0])):
        need, build = float(qty), {}
        for w in weeks:
            if need <= 1e-9:
                break
            take = min(remaining[w], need)
            if take > 1e-9:
                build[w] = take
                remaining[w] -= take
                need -= take
        if need > 1e-6:      # cannot happen if the cumulative constraints hold; validator will flag it
            build[weeks[-1]] = build.get(weeks[-1], 0) + need
        plan[oid] = build
    return plan


def solve_allocation(data: InputData, cfg: Config, *, events: pd.DataFrame | None = None,
                     fixed: dict | None = None, baseline: dict | None = None) -> dict:
    """
    ONE model for ALL orders.

    Decision variables
      x[i,p,w] binary  order i is made at plant p and finished BY week w.
                       For each (i,p) the candidate weeks are: D(i,p) = last week that still arrives
                       on time (one 'on-time' variable), plus every later week (late variables).
      u[i]     binary  order i cannot be produced at all (shortage slack -> reported, never hidden)

    Constraints
      assign       sum_{p,w} x[i,p,w] + u[i] = 1        each order: exactly one plant, or flagged
      capability   x only exists for plants that make the product AND have a route to the destination
      capacity     for every plant p, product k, week t:
                   sum_{i of product k, w <= t} qty_i * x[i,p,w]  <=  sum_{s<=t} effective_cap[p,k,s]
                   (cumulative earliest-deadline-first condition: with production allowed to span
                   weeks, this is exactly the condition for all orders to finish by their week,
                   and weekly capacity is never exceeded - checked again by the EDF schedule
                   and the independent validator)
      fixed        confirmed orders: x only for the confirmed plant, u[i] = 0
      carbon cap   optional: total carbon <= cap

    Objective (normalised so cost and carbon are comparable)
      1000*u + 10*[late] + 0.1*late_days
      + w_cost * cost/C_ref + w_carbon * carbon/K_ref  (+ move_penalty vs baseline when re-planning)
      C_ref / K_ref = sum over orders of the per-order cheapest cost / lowest carbon.
    """
    cfg.check()
    fixed = fixed or {}
    events = data.events if events is None else events
    start, weeks, start_note = resolve_horizon(data, cfg)
    cap, cap_log = effective_capacity(data, start, weeks, cfg, events)
    opts = build_options(data, cfg)
    decisions = delivery_decisions(data, opts, cap, start, weeks, cfg)
    orders = data.orders.set_index("order_id")

    for oid, plant in fixed.items():
        if oid not in opts:
            raise ValueError(f"fixed order {oid} is not allocatable ({data.order_status.get(oid)})")
        if plant not in opts[oid]:
            raise ValueError(f"fixed assignment {oid}->{plant} is not eligible (eligible: {sorted(opts[oid])})")

    model_ids = [i for i in opts if opts[i]]
    c_ref = sum(min(v["cost"] for v in opts[i].values()) for i in model_ids) or 1.0
    k_ref = sum(min(v["carbon"] for v in opts[i].values()) for i in model_ids) or 1.0

    prob = pulp.LpProblem("combined_allocation", pulp.LpMinimize)
    x, u, obj, bound_late = {}, {}, [], {}
    for n, oid in enumerate(model_ids):
        o = orders.loc[oid]
        u[oid] = pulp.LpVariable(f"u_{n}", cat="Binary")
        obj.append(cfg.unserved_penalty * u[oid])
        for p, v in opts[oid].items():
            if oid in fixed and fixed[oid] != p:
                continue
            d = deadline_week(o.due, v["transit_days"], start, weeks, cfg)
            cand = ([d] if d >= 1 else []) + ([w for w in weeks if w > d] if cfg.deadline_mode == "soft" else [])
            for w in cand:
                late = max(0, (ship_date(start, w, cfg) + timedelta(days=v["transit_days"]) - o.due).days)
                bound_late[oid, p, w] = late
                x[oid, p, w] = pulp.LpVariable(f"x_{n}_{p}_{w}", cat="Binary")
                coef = (cfg.cost_weight * v["cost"] / c_ref + cfg.carbon_weight * v["carbon"] / k_ref
                        + (cfg.late_order_penalty if late else 0) + cfg.late_day_penalty * late)
                if baseline and baseline.get(oid) and baseline[oid] != p:
                    coef += cfg.move_penalty
                obj.append(coef * x[oid, p, w])
    prob += pulp.lpSum(obj)

    by_order = {}
    for (i, p, w), var in x.items():
        by_order.setdefault(i, []).append(var)
    for oid in model_ids:
        prob += pulp.lpSum(by_order.get(oid, [])) + u[oid] == 1, f"assign_{oid}"
        if oid in fixed:
            prob += u[oid] == 0, f"fixed_{oid}"

    # cumulative capacity per plant-product line
    line_vars = {}
    for (i, p, w), var in x.items():
        line_vars.setdefault((p, orders.at[i, "product"]), []).append((w, float(orders.at[i, "qty"]), var))
    for (p, prod), items in line_vars.items():
        cum = 0.0
        for t in weeks:
            cum += cap.get((p, prod, t), 0.0)
            prob += pulp.lpSum(q * var for (w, q, var) in items if w <= t) <= cum, f"cap_{p}_{prod}_{t}"
    if cfg.carbon_cap_kg is not None:
        prob += pulp.lpSum(opts[i][p]["carbon"] * var for (i, p, w), var in x.items()) <= cfg.carbon_cap_kg, "carbon_cap"

    prob.solve(_solver(cfg))
    status = pulp.LpStatus[prob.status]
    sol_status = getattr(prob, "sol_status", None)   # 1 optimal, 2 feasible but stopped early
    if status == "Optimal" and sol_status == 2:
        status = "TimeLimit"

    result = dict(run_config=_config_dict(cfg), planning_start=str(start), weeks=weeks,
                  events_applied=sorted(events["event_id"].unique().tolist()) if len(events) else [],
                  capacity_event_log=cap_log, data_issues=data.issues, notes=list(data.notes),
                  assumptions=_assumptions(cfg, start_note, data), limitations=_limitations())
    if status not in ("Optimal", "TimeLimit"):
        result.update(status="INFEASIBLE" if status == "Infeasible" else status.upper(), plan_valid=False,
                      allocation=[], kpis={}, notifications=[],
                      message=("No allocation satisfies all hard constraints. "
                               + (f"Confirmed/fixed assignments that cannot all be honoured: {sorted(fixed)}. " if fixed else "")
                               + ("The carbon cap may be too tight. " if cfg.carbon_cap_kg is not None else "")))
        return result

    # ---- decisions -> weekly production plan (EDF per plant-product line)
    chosen = {i: (p, w) for (i, p, w), var in x.items() if var.value() is not None and var.value() > 0.5}
    lines = {}
    for i, (p, w) in chosen.items():
        lines.setdefault((p, orders.at[i, "product"]), {})[i] = (orders.at[i, "qty"], w)
    builds = {}
    for (p, prod), assigned in lines.items():
        builds.update(edf_schedule(assigned, {w: cap[(p, prod, w)] for w in weeks}, weeks))

    alloc = []
    for o in data.orders.itertuples():
        rec = dict(order_id=o.order_id, product=o.product, qty=None if pd.isna(o.qty) else int(o.qty),
                   destination=o.destination, due=str(o.due) if pd.notna(o.due) else None,
                   status="UNASSIGNED", plant=None, flags=[])
        if o.order_id in data.order_status:
            code, why = data.order_status[o.order_id]
            rec.update(reason_code=code, reason=why)
            alloc.append(rec); continue
        if o.order_id not in chosen:
            d = decisions[o.order_id]
            if d["kind"] == "NOT_PRODUCIBLE_IN_HORIZON":
                rec.update(reason_code="EXCEEDS_CAPACITY_IN_HORIZON",
                           reason="qty exceeds what any eligible plant can build in the planning horizon")
            elif cfg.deadline_mode == "hard" and d["best_on_time"] is None:
                rec.update(reason_code="NO_ON_TIME_OPTION",
                           reason="hard deadline mode and no plant can arrive by the due date")
            else:
                rec.update(reason_code="CAPACITY_SHORTAGE",
                           reason="eligible plants have no capacity left once all orders compete")
            alloc.append(rec); continue
        p, bound_w = chosen[o.order_id]
        v = opts[o.order_id][p]
        build = {int(w): round(n, 3) for w, n in builds[o.order_id].items()}
        done_w = max(build)
        arr = ship_date(start, done_w, cfg) + timedelta(days=v["transit_days"])
        late = max(0, (arr - o.due).days)
        uc = v["unit"]
        rec.update(status="ASSIGNED", plant=p, completion_week=done_w, finish_by_week=bound_w,
                   ship_date=str(ship_date(start, done_w, cfg)), arrival=str(arr), on_time=late == 0,
                   late_days=late, production_by_week=build,
                   landed_cost=round(v["cost"], 2), carbon_kg=round(v["carbon"], 3),
                   cost_breakdown={k: round(uc[k] * o.qty, 2) for k in ("production", "energy", "transport", "duty")},
                   carbon_breakdown={"manufacturing": round(uc["carbon_mfg"] * o.qty, 3),
                                     "transport": round(uc["carbon_transport"] * o.qty, 3)},
                   fixed=o.order_id in fixed)
        if late:
            rec["flags"].append("LATE")
        alloc.append(rec)

    result.update(status="OPTIMAL" if status == "Optimal" else "FEASIBLE_NOT_PROVEN_OPTIMAL",
                  objective_value=round(pulp.value(prob.objective), 6),
                  normalisation={"cost_ref": round(c_ref, 2), "carbon_ref_kg": round(k_ref, 2)},
                  allocation=alloc, _cap=cap, _start=start, _opts=opts)
    result["notifications"] = build_notifications(result, decisions, fixed)
    result["kpis"] = calculate_kpis(result, cap)
    result["validation"] = validate_allocation(result, data, cfg, events, fixed)
    result["plan_valid"] = result["validation"]["ok"]
    if not result["plan_valid"]:
        result["status"] = "INVALID_PLAN"
    return result


def _solver(cfg: Config):
    """HiGHS (fast) if installed, else the CBC solver bundled with PuLP. Single thread = repeatable."""
    kw = dict(msg=False, timeLimit=cfg.solver_time_limit, gapRel=cfg.solver_gap, threads=1)
    if cfg.solver == "highs" or (cfg.solver == "auto" and "HiGHS" in pulp.listSolvers(onlyAvailable=True)):
        return pulp.HiGHS(**kw)
    return pulp.PULP_CBC_CMD(**kw)


def build_notifications(result, decisions, fixed) -> list:
    """Late notices + choices the user must/can make."""
    notes = []
    by_id = {a["order_id"]: a for a in result["allocation"]}
    for oid, d in decisions.items():
        a = by_id[oid]
        planned = a.get("plant")
        if d["kind"] == "CHEAPEST_IS_LATE":
            ch, ot = d["cheapest"], d["best_on_time"]
            opts = [dict(ch, label="cheapest (LATE)"), dict(ot, label="next best on time")]
            if planned and planned not in (ch["plant"], ot["plant"]):
                opts.append(dict(next(r for r in d["options"] if r["plant"] == planned), label="optimiser's choice"))
            notes.append(dict(order_id=oid, type="CHEAPEST_IS_LATE", requires_user_choice=oid not in fixed,
                              planned_plant=planned, options=opts,
                              message=(f"{oid}: cheapest plant {ch['plant']} arrives {ch['earliest_arrival']}, "
                                       f"{ch['late_days']} day(s) after due {a['due']}. "
                                       f"Next best on-time plant {ot['plant']} costs "
                                       f"{ot['landed_cost'] - ch['landed_cost']:+,.2f} more "
                                       f"({ot['carbon_kg'] - ch['carbon_kg']:+,.1f} kg CO2e). "
                                       f"Plan uses {planned}. Choose {ch['plant']} (cheaper, late) or {ot['plant']} (on time).")))
        elif d["kind"] == "ALL_LATE":
            notes.append(dict(order_id=oid, type="ALL_LATE", requires_user_choice=oid not in fixed,
                              planned_plant=planned, options=d["options"],
                              message=(f"{oid}: every capable plant arrives after due {a['due']} "
                                       f"({', '.join(f'{r['plant']}: {r['late_days']}d late, {r['landed_cost']:,.0f}' for r in d['options'] if r['producible_in_horizon'])}). "
                                       f"Provisionally assigned to {planned}. Please choose a plant.")))
        if a["status"] == "ASSIGNED" and a["late_days"] and d["kind"] == "OK":
            notes.append(dict(order_id=oid, type="LATE_DUE_TO_CAPACITY", requires_user_choice=False,
                              planned_plant=planned,
                              message=(f"{oid}: on-time delivery was possible in isolation but capacity is taken "
                                       f"by other orders; arrives {a['arrival']}, {a['late_days']} day(s) late.")))
    for a in result["allocation"]:
        if a["status"] != "ASSIGNED":
            notes.append(dict(order_id=a["order_id"], type=a["reason_code"], requires_user_choice=False,
                              message=f"{a['order_id']}: NOT ALLOCATED - {a['reason']}"))
    return notes


# =============================================================================
# 8. INDEPENDENT VALIDATION + KPIs (recomputed from the raw tables, not from the solver)
# =============================================================================
def validate_allocation(result, data: InputData, cfg: Config, events, fixed) -> dict:
    errors, warnings = [], []
    start = result["_start"]
    P = {(r.plant, r.product): r for r in data.plants.itertuples()}
    R = {(r.plant, r.destination): r for r in data.routes.itertuples()}
    # capacity recomputed with a separate simple loop
    used, cap = {}, {}
    for (p, prod), row in P.items():
        for w in result["weeks"]:
            factor = 0.0
            for d in range(7):
                day = week_start(start, w) + timedelta(days=d)
                act = events[(events["plant"] == p) & (events["from_date"] <= day) & (events["to_date"] >= day)]
                rates = act.drop_duplicates("event_id")["loss_rate"].tolist()
                loss = 0.0 if not rates else (max(rates) if cfg.event_overlap_rule == "max" else min(1.0, sum(rates)))
                factor += (1 - loss) / 7
            cap[(p, prod, w)] = row.weekly_capacity * factor
    ids = [a["order_id"] for a in result["allocation"]]
    if sorted(ids) != sorted(data.orders["order_id"]):
        errors.append("allocation does not account for every order exactly once")
    tot_cost = tot_carbon = 0.0
    for a in result["allocation"]:
        if a["status"] != "ASSIGNED":
            continue
        key = (a["plant"], a["product"])
        if key not in P:
            errors.append(f"{a['order_id']}: plant {a['plant']} cannot make {a['product']}"); continue
        if (a["plant"], a["destination"]) not in R:
            errors.append(f"{a['order_id']}: no route {a['plant']}->{a['destination']}"); continue
        if a["order_id"] in fixed and fixed[a["order_id"]] != a["plant"]:
            errors.append(f"{a['order_id']}: confirmed assignment {fixed[a['order_id']]} was changed")
        built = sum(a["production_by_week"].values())
        if abs(built - a["qty"]) > 1e-4:
            errors.append(f"{a['order_id']}: built {built} != qty {a['qty']}")
        if any(int(w) > a["completion_week"] for w in a["production_by_week"]):
            errors.append(f"{a['order_id']}: production after ship week")
        for w, n in a["production_by_week"].items():
            used[(a["plant"], a["product"], int(w))] = used.get((a["plant"], a["product"], int(w)), 0) + n
        pr, rt = P[key], R[(a["plant"], a["destination"])]
        u = unit_cost_and_carbon(pr, rt, cfg)
        cost, carbon = u["cost"] * a["qty"], u["carbon"] * a["qty"]
        if abs(cost - a["landed_cost"]) > 0.01 + 1e-9 * cost or abs(carbon - a["carbon_kg"]) > 0.001 + 1e-9 * carbon:
            errors.append(f"{a['order_id']}: cost/carbon mismatch on recomputation")
        tot_cost += cost; tot_carbon += carbon
        arr = ship_date(start, a["completion_week"], cfg) + timedelta(days=int(rt.transit_days))
        if str(arr) != a["arrival"]:
            errors.append(f"{a['order_id']}: arrival mismatch")
        late = max(0, (arr - date.fromisoformat(a["due"])).days)
        if late != a["late_days"]:
            errors.append(f"{a['order_id']}: late-days mismatch")
        if late and cfg.deadline_mode == "hard":
            errors.append(f"{a['order_id']}: late in hard deadline mode")
    for k, n in used.items():
        if n > cap.get(k, 0) + 1e-6:
            errors.append(f"capacity exceeded at {k}: {n:.1f} > {cap.get(k, 0):.1f}")
    for k in cap:
        if abs(cap[k] - result["_cap"][k]) > 1e-6:
            errors.append(f"effective capacity mismatch at {k}")
    k = result.get("kpis", {})
    if k and (abs(k["total_cost"] - round(tot_cost, 2)) > 0.05 or abs(k["total_carbon_kg"] - round(tot_carbon, 3)) > 0.05):
        errors.append("KPI totals do not match independent recomputation")
    late_unflagged = [a["order_id"] for a in result["allocation"] if a.get("late_days") and "LATE" not in a["flags"]]
    if late_unflagged:
        errors.append(f"late orders not flagged: {late_unflagged}")
    return dict(ok=not errors, errors=errors, warnings=warnings,
                capacity_ok=not any("capacity" in e for e in errors),
                eligibility_ok=not any("cannot make" in e or "no route" in e for e in errors),
                recomputed_total_cost=round(tot_cost, 2), recomputed_total_carbon_kg=round(tot_carbon, 3))


def calculate_kpis(result, cap) -> dict:
    assigned = [a for a in result["allocation"] if a["status"] == "ASSIGNED"]
    used, avail = {}, {}
    for (p, prod, w), cval in cap.items():
        avail[p] = avail.get(p, 0) + cval
    for a in assigned:
        used[a["plant"]] = used.get(a["plant"], 0) + a["qty"]
    plant_week = {}
    for a in assigned:
        for w, n in a["production_by_week"].items():
            key = f"{a['plant']}|{a['product']}|W{w}"
            plant_week[key] = plant_week.get(key, 0) + n
    return dict(
        total_cost=round(sum(a["landed_cost"] for a in assigned), 2),
        total_carbon_kg=round(sum(a["carbon_kg"] for a in assigned), 3),
        total_carbon_t=round(sum(a["carbon_kg"] for a in assigned) / 1000, 3),
        cost_breakdown={k: round(sum(a["cost_breakdown"][k] for a in assigned), 2)
                        for k in ("production", "energy", "transport", "duty")},
        orders_total=len(result["allocation"]), orders_assigned=len(assigned),
        orders_unassigned=len(result["allocation"]) - len(assigned),
        late_orders=sum(1 for a in assigned if a["late_days"]),
        total_late_days=sum(a["late_days"] for a in assigned),
        plant_utilisation={p: round(used.get(p, 0) / avail[p], 4) for p in sorted(avail) if avail[p] > 0},
        plant_order_share={p: sum(1 for a in assigned if a["plant"] == p) for p in sorted(avail)},
        units_by_plant_product_week={k: round(v, 2) for k, v in sorted(plant_week.items())},
        weights_used={"cost": result["run_config"]["cost_weight"], "carbon": result["run_config"]["carbon_weight"]},
        objective_value=result.get("objective_value"))


# =============================================================================
# 9. SCENARIOS
# =============================================================================
def compare_allocations(base: dict, new: dict) -> dict:
    b = {a["order_id"]: a for a in base["allocation"]}
    n = {a["order_id"]: a for a in new["allocation"]}
    moved, same, shifted = [], [], []
    for oid in n:
        pb, pn = b.get(oid, {}).get("plant"), n[oid].get("plant")
        if oid not in b:
            continue
        if pb != pn:
            moved.append(dict(order_id=oid, from_plant=pb, to_plant=pn,
                              cost_change=round((n[oid].get("landed_cost") or 0) - (b[oid].get("landed_cost") or 0), 2),
                              carbon_change_kg=round((n[oid].get("carbon_kg") or 0) - (b[oid].get("carbon_kg") or 0), 3)))
        else:
            same.append(oid)
            if b[oid].get("completion_week") != n[oid].get("completion_week"):
                shifted.append(oid)
    kb, kn = base["kpis"], new["kpis"]
    return dict(
        baseline=dict(total_cost=kb["total_cost"], total_carbon_t=kb["total_carbon_t"],
                      late_orders=kb["late_orders"], unassigned=kb["orders_unassigned"]),
        revised=dict(total_cost=kn["total_cost"], total_carbon_t=kn["total_carbon_t"],
                     late_orders=kn["late_orders"], unassigned=kn["orders_unassigned"]),
        delta=dict(cost=round(kn["total_cost"] - kb["total_cost"], 2),
                   cost_pct=round(100 * (kn["total_cost"] - kb["total_cost"]) / kb["total_cost"], 3) if kb["total_cost"] else None,
                   carbon_t=round(kn["total_carbon_t"] - kb["total_carbon_t"], 3),
                   late_orders=kn["late_orders"] - kb["late_orders"],
                   unassigned=kn["orders_unassigned"] - kb["orders_unassigned"]),
        orders_moved=moved, orders_same_plant=len(same), orders_week_shifted=shifted,
        newly_late=[o for o in n if n[o].get("late_days") and not b.get(o, {}).get("late_days")],
        newly_unassigned=[o for o in n if n[o]["status"] != "ASSIGNED" and b.get(o, {}).get("status") == "ASSIGNED"],
        warnings=[x["message"] for x in new["notifications"] if x["type"] in
                  ("CAPACITY_SHORTAGE", "EXCEEDS_CAPACITY_IN_HORIZON", "NO_ON_TIME_OPTION")])


def run_capacity_scenario(data: InputData, cfg: Config, event_ids=None, extra_events: list[dict] | None = None,
                          baseline: dict | None = None) -> dict:
    """Baseline = no events (unless given). Revised = selected events applied, with a stability term."""
    ev = data.events if event_ids is None else data.events[data.events["event_id"].isin(event_ids)]
    if extra_events:
        add = pd.DataFrame(extra_events)
        add["from_date"] = pd.to_datetime(add["from_date"]).dt.date
        add["to_date"] = pd.to_datetime(add["to_date"]).dt.date
        add["loss_rate"] = add["capacity_loss_pct"].astype(float) / (1 if add["capacity_loss_pct"].max() <= 1 else 100)
        ev = pd.concat([ev, add], ignore_index=True)
    if baseline is None:
        baseline = solve_allocation(data, cfg, events=ev.iloc[0:0])
    base_plants = {a["order_id"]: a["plant"] for a in baseline["allocation"] if a["plant"]}
    revised = solve_allocation(data, cfg, events=ev, baseline=base_plants)
    return dict(baseline=baseline, revised=revised,
                comparison=compare_allocations(baseline, revised) if revised.get("kpis") else None)


def run_weight_scenarios(data: InputData, cfg: Config, weight_sets) -> list:
    rows = []
    for wc, wk in weight_sets:
        c2 = Config(**{**asdict(cfg), "cost_weight": wc, "carbon_weight": wk})
        r = solve_allocation(data, c2)
        rows.append(dict(cost_weight=wc, carbon_weight=wk, status=r["status"],
                         total_cost=r["kpis"].get("total_cost"), total_carbon_t=r["kpis"].get("total_carbon_t"),
                         late_orders=r["kpis"].get("late_orders"),
                         plant_order_share=r["kpis"].get("plant_order_share"),
                         plan={a["order_id"]: a["plant"] for a in r["allocation"]}))
    return rows


class MissingOrderInfo(ValueError):
    pass


def add_order_and_reoptimize(data: InputData, cfg: Config, new_order: dict, *, mode="reoptimize_all",
                             confirmed: dict | None = None, current_plan: dict | None = None) -> dict:
    """
    mode = "reoptimize_all": every non-confirmed order may move (main workflow)
    mode = "insert_only"   : every existing order keeps its current plant; only the new one is placed
    confirmed              : {order_id: plant} that must never change, in either mode
    """
    missing = [k for k in ("order_id", "product", "qty", "destination", "due") if not new_order.get(k)]
    if missing:
        raise MissingOrderInfo(f"new order needs {missing}: destination and due date drive transport cost, "
                               f"duty, transit time and delivery feasibility")
    if new_order["order_id"] in set(data.orders["order_id"]):
        raise ValueError(f"order_id {new_order['order_id']} already exists")
    raw_orders = data.orders.copy()
    raw_orders["due"] = raw_orders["due"].astype(str)
    raw_orders["qty"] = raw_orders["qty"].astype("Int64").astype(str)
    raw = dict(orders=pd.concat([raw_orders[REQUIRED_COLUMNS["orders"]],
                                 pd.DataFrame([{k: str(new_order[k]) for k in REQUIRED_COLUMNS["orders"]}])],
                                ignore_index=True).astype(str),
               plants=_back_to_raw(data.plants), transport_duties=_back_to_raw(data.routes),
               capacity_events=_back_to_raw(data.events))
    new_data = validate_data(raw, cfg)
    if new_order["order_id"] in new_data.order_status:
        code, why = new_data.order_status[new_order["order_id"]]
        raise ValueError(f"new order cannot be allocated: {code} - {why}")
    fixed = dict(confirmed or {})
    if mode == "insert_only":
        if current_plan is None:
            raise ValueError("insert_only mode needs current_plan {order_id: plant}")
        fixed.update({k: v for k, v in current_plan.items() if v})
    before = solve_allocation(data, cfg, fixed=confirmed or {})
    after = solve_allocation(new_data, cfg, fixed=fixed,
                             baseline={a["order_id"]: a["plant"] for a in before["allocation"] if a["plant"]})
    return dict(mode=mode, before=before, after=after,
                new_order_result=next(a for a in after["allocation"] if a["order_id"] == new_order["order_id"]),
                comparison=compare_allocations(before, after) if after.get("kpis") else None)


def _back_to_raw(df):
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].astype(str)
    if "duty_rate" in out:   # keep the original percent column, drop derived rate
        out = out.drop(columns=["duty_rate"])
    if "loss_rate" in out:
        out = out.drop(columns=["loss_rate"])
    return out


def cheapest_plant_baseline(data: InputData, cfg: Config) -> dict:
    """Naive rule (Level 2): every order to its cheapest plant if capacity is free, ignoring due dates.
    Used only to show why combined optimisation is needed."""
    start, weeks, _ = resolve_horizon(data, cfg)
    cap, _ = effective_capacity(data, start, weeks, cfg)
    cap = dict(cap)
    opts = build_options(data, cfg)
    orders = data.orders.set_index("order_id")
    plan = {}
    for oid in sorted(opts, key=lambda i: (orders.at[i, "due"], i)):
        o = orders.loc[oid]
        for p in sorted(opts[oid], key=lambda p: opts[oid][p]["cost"]):
            remaining, wk = o.qty, None
            trial = dict(cap)
            for w in weeks:
                take = min(trial[(p, o["product"], w)], remaining)
                trial[(p, o["product"], w)] -= take; remaining -= take
                if remaining <= 1e-9:
                    wk = w; break
            if wk:
                cap = trial
                arr = ship_date(start, wk, cfg) + timedelta(days=opts[oid][p]["transit_days"])
                plan[oid] = dict(plant=p, cost=opts[oid][p]["cost"], carbon=opts[oid][p]["carbon"],
                                 late=(arr > o.due))
                break
    return dict(total_cost=round(sum(v["cost"] for v in plan.values()), 2),
                total_carbon_t=round(sum(v["carbon"] for v in plan.values()) / 1000, 3),
                late_orders=sum(v["late"] for v in plan.values()),
                unassigned=len(opts) - len(plan), plan={k: v["plant"] for k, v in plan.items()})


# ---------------------------------------------------------------- helpers
def _config_dict(cfg):
    d = asdict(cfg)
    d["planning_start"] = str(cfg.planning_start) if cfg.planning_start else None
    return d


def _assumptions(cfg, start_note, data):
    energy_col = "energy_price_per_kwh column" if "energy_price_per_kwh" in data.plants.columns else "n/a"
    return [
        start_note,
        "Capacity is per plant AND product per week (plants.csv has one row per plant-product); "
        "each order consumes the capacity of its own product line only.",
        f"An order ships on day {cfg.ship_day_offset} (0=Mon) of the week its last unit is built; "
        "arrival = ship date + transit_days. Production lead time inside a week is not modelled.",
        "Orders are never split across plants; a large order may be built over several weeks at the same "
        "plant (earliest-deadline-first) and ships when its last unit is built.",
        f"Landed cost = production + energy (kWh x price from {energy_col}) + transport + duty; "
        f"duty base = {cfg.duty_base}.",
        "Carbon = energy_kwh_per_unit x carbon_kg_per_kwh + transport carbon_kg_per_unit, times qty.",
        f"Weights cost {cfg.cost_weight} / carbon {cfg.carbon_weight}; each normalised by the sum of per-order "
        "minima so 1.0 = 'best possible' on that axis.",
        "Priority: serve every order > minimise late orders > minimise late days > weighted cost/carbon.",
        f"Overlapping capacity events on the same plant and day combine by '{cfg.event_overlap_rule}'; "
        "a partial-week event reduces capacity by the share of affected days.",
        f"Deadline mode '{cfg.deadline_mode}'.",
    ]


def _limitations():
    return ["Duty rates and transit days assumed constant over the horizon.",
            "No weekly variation in carbon intensity (no such column in the schema).",
            "No inventory/holding cost for building early.",
            "Capacity unit assumed to be 'units of that product per week'."]


def public(result: dict) -> dict:
    """Strip internal keys so the result is JSON-serialisable (for FastAPI / LangGraph)."""
    out = {k: v for k, v in result.items() if not k.startswith("_")}
    for key in ("baseline", "revised", "before", "after"):
        if isinstance(out.get(key), dict):
            out[key] = public(out[key])
    return out


# =============================================================================
# 10. CLI
# =============================================================================
def print_summary(r: dict, title="PLAN"):
    print(f"\n{'=' * 78}\n{title}: status={r['status']}  valid={r.get('plan_valid')}  "
          f"events={r.get('events_applied')}")
    if not r.get("kpis"):
        print(r.get("message", "")); return
    k = r["kpis"]
    print(f"cost {k['total_cost']:,.2f} | carbon {k['total_carbon_t']:,.3f} t | assigned {k['orders_assigned']}/"
          f"{k['orders_total']} | late {k['late_orders']} | utilisation {k['plant_utilisation']}")
    print(f"weights {k['weights_used']}  objective {k['objective_value']}")
    df = pd.DataFrame(r["allocation"])
    cols = [c for c in ["order_id", "product", "qty", "destination", "due", "plant", "arrival",
                        "late_days", "landed_cost", "carbon_kg", "status"] if c in df]
    print(df[cols].head(15).to_string(index=False), "\n  ...")
    for n in r.get("notes", []):
        print("NOTE:", n)


def summary_table(result: dict) -> pd.DataFrame:
    """One row per order: Order ID, Product, Qty, Destination, Assigned Plant, Total Cost, Carbon (kg), Delivery."""
    rows = []
    for a in result["allocation"]:
        if a["status"] == "ASSIGNED":
            delivery = "On time" if a["on_time"] else f"Late ({a['late_days']} days)"
            plant, cost, carbon = f"Plant {a['plant']}", round(a["landed_cost"]), round(a["carbon_kg"])
        else:
            delivery = f"Not allocated - {a.get('reason', '')}"
            plant, cost, carbon = "-", None, None
        rows.append({"Order ID": a["order_id"], "Product": a["product"], "Qty": a["qty"],
                     "Destination": a["destination"], "Assigned Plant": plant,
                     "Total Cost": cost, "Carbon (kg)": carbon, "Delivery": delivery})
    return pd.DataFrame(rows)


def interactive_choices(result: dict) -> dict:
    """Ask the user to choose for CHEAPEST_IS_LATE and ALL_LATE orders. Blank = keep the plan's plant."""
    choices = {}
    pending = [n for n in result["notifications"] if n.get("requires_user_choice")]
    if not pending:
        return choices
    print(f"\n{len(pending)} order(s) need a delivery decision.")
    for n in pending:
        print("\n" + n["message"])
        opts = [o for o in n["options"] if o.get("producible_in_horizon", True)]
        for j, o in enumerate(opts, 1):
            tag = o.get("label", "")
            late = f"{o['late_days']}d LATE" if o["late_days"] else "on time"
            print(f"   [{j}] plant {o['plant']}: cost {o['landed_cost']:,.2f}, carbon {o['carbon_kg']:,.1f} kg, "
                  f"earliest arrival {o['earliest_arrival']} ({late}) {tag}")
        ans = input(f"   choose 1-{len(opts)} (Enter = keep {n['planned_plant']}): ").strip()
        if ans.isdigit() and 1 <= int(ans) <= len(opts):
            choices[n["order_id"]] = opts[int(ans) - 1]["plant"]
        elif n["planned_plant"]:
            choices[n["order_id"]] = n["planned_plant"]
    return choices


def main(argv=None):
    ap = argparse.ArgumentParser(description="Combined multi-plant order allocation (PuLP)")
    ap.add_argument("--data", default="data", help="folder with the 4 CSVs")
    ap.add_argument("--start", help="planning start date YYYY-MM-DD (Monday of week 1)")
    ap.add_argument("--weeks", type=int, help="planning horizon in weeks")
    ap.add_argument("--weights", nargs=2, type=float, metavar=("COST", "CARBON"), default=(0.7, 0.3))
    ap.add_argument("--events", default="all", help="'all', 'none', or comma list e.g. EV-01,EV-02")
    ap.add_argument("--deadline-mode", choices=["soft", "hard"], default="soft")
    ap.add_argument("--duty-base", choices=["production", "exworks", "exworks_plus_transport"], default="exworks")
    ap.add_argument("--energy-price", type=float, help="default energy price if plants.csv lacks the column")
    ap.add_argument("--carbon-cap-kg", type=float)
    ap.add_argument("--choices", help="JSON file {order_id: plant} of user decisions to hold fixed")
    ap.add_argument("--interactive", action="store_true", help="ask for late-delivery decisions, then re-solve")
    ap.add_argument("--compare-weights", action="store_true")
    ap.add_argument("--inspect", action="store_true", help="only print schema summary + validation")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    cfg = Config(planning_start=date.fromisoformat(args.start) if args.start else None, horizon_weeks=args.weeks,
                 cost_weight=args.weights[0], carbon_weight=args.weights[1], deadline_mode=args.deadline_mode,
                 duty_base=args.duty_base, default_energy_price=args.energy_price, carbon_cap_kg=args.carbon_cap_kg)
    raw = load_data(args.data)
    print(inspect_data(raw))
    try:
        data = validate_data(raw, cfg)
    except DataValidationError as e:
        print("\n" + str(e)); return 2
    print(f"\nValidation: {sum(i['severity'] == 'ERROR' for i in data.issues)} errors, "
          f"{sum(i['severity'] == 'WARNING' for i in data.issues)} warnings")
    for i in data.issues:
        print(f"  {i['severity']:<7} [{i['table']}] {i['message']}")
    for n in data.notes:
        print("  NOTE:", n)
    if args.inspect:
        return 0

    if args.events == "none":
        ev_ids = []
    elif args.events == "all":
        ev_ids = data.events["event_id"].tolist()
    else:
        ev_ids = [e.strip() for e in args.events.split(",")]
        unknown = set(ev_ids) - set(data.events["event_id"])
        if unknown:
            print(f"unknown event ids {sorted(unknown)}"); return 2
    events = data.events[data.events["event_id"].isin(ev_ids)]
    fixed = json.loads(Path(args.choices).read_text()) if args.choices else {}

    result = solve_allocation(data, cfg, events=events, fixed=fixed)
    print_summary(result, "COMBINED PLAN")
    for n in result.get("notifications", []):
        print(" !", n["message"])

    if args.interactive and result.get("kpis"):
        choices = interactive_choices(result)
        if choices:
            fixed = {**fixed, **choices}
            result = solve_allocation(data, cfg, events=events, fixed=fixed,
                                      baseline={a["order_id"]: a["plant"] for a in result["allocation"] if a["plant"]})
            print_summary(result, "RE-SOLVED WITH YOUR CHOICES")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    bundle = dict(plan=public(result))
    if ev_ids and result.get("kpis"):
        scen = run_capacity_scenario(data, cfg, ev_ids)
        bundle["capacity_scenario"] = dict(comparison=scen["comparison"])
        cmp_ = scen["comparison"]
        if cmp_:
            print(f"\nCAPACITY EVENTS {ev_ids} vs no events: cost {cmp_['delta']['cost']:+,.2f} "
                  f"({cmp_['delta']['cost_pct']:+}%), carbon {cmp_['delta']['carbon_t']:+.3f} t, "
                  f"late {cmp_['delta']['late_orders']:+d}, unassigned {cmp_['delta']['unassigned']:+d}, "
                  f"moved {len(cmp_['orders_moved'])} orders")
    naive = cheapest_plant_baseline(data, cfg)
    bundle["cheapest_plant_baseline"] = {k: v for k, v in naive.items() if k != "plan"}
    print(f"\nNaive cheapest-plant rule (no events): cost {naive['total_cost']:,.2f}, carbon "
          f"{naive['total_carbon_t']} t, late {naive['late_orders']}, unassigned {naive['unassigned']}")
    if args.compare_weights:
        rows = run_weight_scenarios(data, cfg, [(1.0, 0.0), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.0, 1.0)])
        bundle["weight_scenarios"] = [{k: v for k, v in r.items() if k != "plan"} for r in rows]
        print("\nWEIGHT SCENARIOS")
        print(pd.DataFrame(bundle["weight_scenarios"]).to_string(index=False))

    (out / "result.json").write_text(json.dumps(bundle, indent=2, default=str))
    pd.DataFrame(result["allocation"]).to_csv(out / "allocation.csv", index=False)
    if result.get("allocation"):
        table = summary_table(result)
        table.to_csv(out / "allocation_summary.csv", index=False)
        print("\n" + table.to_string(index=False))
    print(f"\nWrote {out / 'allocation_summary.csv'}, {out / 'allocation.csv'} and {out / 'result.json'}")
    return 0 if result.get("plan_valid") else 1


if __name__ == "__main__":
    sys.exit(main())
