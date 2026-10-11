# Multi-Plant Order Allocation with Cost and Carbon

HCLTech Future Ready AI Engineer Hackathon project.

This system decides which plant (A, B or C) should build each customer order. It balances **cost**, **carbon** and **delivery dates**, and it never puts more work on a plant than it can handle. You can ask questions in plain English through a simple chat app.

## How it works

```
User question  ->  Streamlit app  ->  Gemini picks a function  ->  Python runs it (PuLP solver)
                                                                        |
User gets answer  <-  Gemini writes it in plain English  <-  results ---+
```

- **The solver makes the plan, not the AI.** All numbers come from Python code.
- **Gemini only chooses which function to run** and explains the result.
- **What-if questions never change the real data.** They run on a copy.
- **Every answer is checked.** If the AI states a number that no function returned, the app shows a warning.

## Files

| File | What it does |
|---|---|
| `allocator.py` | The optimiser. Loads and checks the CSVs, calculates cost and carbon, and assigns all orders together with PuLP |
| `tools.py` | 20 ready-made functions (plan summary, explain an order, what-ifs, trade-offs, brief facts) |
| `agent.py` | Sends the question to Gemini, runs the functions it picks, and saves a trace of each run |
| `app.py` | Streamlit web app (chat, plan table, trade-off chart, traces) |
| `data/` | `orders.csv`, `plants.csv`, `transport_duties.csv`, `capacity_events.csv` |
| `results/` | Output CSVs from the command-line optimiser |
| `traces/` | One JSON file per question (steps, tools, data used) |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo 'GEMINI_API_KEY=your_key_here' > .env
```

## Run

**Chat app:**
```bash
streamlit run app.py
```
Then open http://localhost:8501.

**Optimiser only (no AI):**
```bash
python allocator.py --data data --start 2026-10-12
```
This writes `results/allocation_summary.csv`.

No API key? Turn on **Mock mode** in the sidebar. The app still runs every function and shows the raw results.

## Example questions

- Allocate this week's orders and summarise the plan.
- What does a 10% carbon cut cost?
- What is the cheapest plan that is on time?
- Plant B loses 30% capacity. Re-plan.
- Why was order 77 built in Plant C?
- What if a new order of 400 units of P1 to Germany is due on 2026-11-10?
- Write the brief for the leadership team.

## Rules the plan follows

- An order only goes to a plant that can make its product and has a route to the destination.
- Weekly capacity per plant and product is never exceeded.
- Late orders are flagged. If the cheapest plant is late, the user can choose between it and the best on-time plant. If every plant is late, the user decides.
- A product that no plant can make is reported, never dropped silently.
- Capacity events only reduce capacity during their dates. The original capacity is kept.

## Key formulas

- **Landed cost** = production + energy + transport + import duty
- **Carbon** = energy used × plant carbon intensity + transport carbon
- **Objective**, in priority order:
  1. Allocate every order.
  2. Have as few late orders as possible.
  3. Have as few total late days as possible.
  4. Minimise **0.7 × cost + 0.3 × carbon**. Both are scaled so they can be compared.

**Why 0.7 / 0.3:** cost stays the main driver, but carbon pushes close decisions toward greener plants. You can change the weights with the slider in the app.

## Assumptions

- Planning starts on Monday 2026-10-12, in weekly periods.
- An order ships on the Friday of the week its last unit is built. Arrival = ship date + transit days.
- An order is never split across plants, but a large order can be built over several weeks at one plant.
- Import duty is charged on production + energy cost.

## Limitations

- Duty and transit times are fixed for the whole period.
- Carbon intensity does not change by week.
- There is no cost for building early or storing stock.
