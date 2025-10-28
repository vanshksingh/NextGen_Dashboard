
"""
Realistic-Looking Logistics Dashboard (Single-File Demo)
Run: streamlit run dummy_ui_pro.py

What this does
- Looks/feels like a real logistics analytics app (no external CSVs required)
- Generates synthetic data for Orders, Costs, Routes, Delivery, Feedback
- Interactive dashboard with KPIs + 5 charts
- Downloads section for filtered synthetic datasets
- "Ops Copilot" chatbox that:
  • Gives overall summary of current synthetic data
  • Finds per-order insights via fuzzy matching (difflib) from free-text
  • Suggests actions to improve customer happiness
  • Lists priority issues sorted by lowest ratings
- Optional Gemini client wiring (not executed automatically)
- Modular, callable functions; implementation in `if __name__ == "__main__":`
"""

from __future__ import annotations

import os
import io
import math
import difflib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
from dotenv import load_dotenv

# Optional Google GenAI client (safe if missing). Not executed unless you wire it.
try:
    import google.genai as genai
except Exception:
    genai = None


# =========================
# Infrastructure / Clients
# =========================
def get_genai_client() -> Optional["genai.Client"]:
    """
    Initializes a Google GenAI client if GEMINI_API_KEY exists and library is installed.
    Per your preference:
        load_dotenv()
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        return None
    client = genai.Client(api_key=api_key)
    return client


# =========================
# Synthetic Data
# =========================
def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_dummy_orders(n: int, rng: np.random.Generator, start: datetime, days: int) -> pd.DataFrame:
    order_ids = [f"ORD{str(i).zfill(6)}" for i in range(1, n + 1)]
    dates = [start + timedelta(days=int(rng.integers(0, days))) for _ in range(n)]
    segments = rng.choice(["Individual", "SMB", "Enterprise"], size=n, p=[0.5, 0.3, 0.2]).tolist()
    priorities = rng.choice(["Standard", "Express", "Critical"], size=n, p=[0.6, 0.3, 0.1]).tolist()
    categories = rng.choice(["Electronics", "Industrial", "Apparel", "FMCG"], size=n).tolist()
    origins = rng.choice(["Delhi", "Mumbai", "Kolkata", "Chennai", "Hyderabad", "Bengaluru"], size=n).tolist()
    dests = rng.choice(["Delhi", "Mumbai", "Kolkata", "Chennai", "Hyderabad", "Bengaluru"], size=n).tolist()
    base_values = rng.normal(12000, 3500, size=n).clip(1500, 40000)
    special = rng.choice(["None", "Fragile", "Cold-Chain", "Hazmat"], size=n, p=[0.7, 0.15, 0.1, 0.05]).tolist()
    return pd.DataFrame({
        "Order_ID": order_ids,
        "Order_Date": pd.to_datetime(dates),
        "Customer_Segment": segments,
        "Priority": priorities,
        "Product_Category": categories,
        "Order_Value_INR": np.round(base_values, 2),
        "Origin": origins,
        "Destination": dests,
        "Special_Handling": special,
    })


def make_dummy_costs(orders: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(orders)
    fuel = rng.normal(1800, 500, n).clip(200, 4000)
    labor = rng.normal(1200, 400, n).clip(150, 3000)
    maint = rng.normal(600, 200, n).clip(50, 1500)
    insur = rng.normal(300, 100, n).clip(50, 800)
    pack = rng.normal(250, 120, n).clip(50, 700)
    tech = rng.normal(180, 60, n).clip(50, 600)
    other = rng.normal(200, 90, n).clip(0, 800)
    df = pd.DataFrame({
        "Order_ID": orders["Order_ID"],
        "Fuel_Cost": np.round(fuel, 2),
        "Labor_Cost": np.round(labor, 2),
        "Vehicle_Maintenance": np.round(maint, 2),
        "Insurance": np.round(insur, 2),
        "Packaging_Cost": np.round(pack, 2),
        "Technology_Platform_Fee": np.round(tech, 2),
        "Other_Overhead": np.round(other, 2),
    })
    df["Total_Logistics_Cost_INR"] = df[
        ["Fuel_Cost", "Labor_Cost", "Vehicle_Maintenance", "Insurance",
         "Packaging_Cost", "Technology_Platform_Fee", "Other_Overhead"]
    ].sum(axis=1).round(2)
    return df


def make_dummy_routes(orders: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(orders)
    route = (orders["Origin"] + "-" + orders["Destination"]).tolist()
    distance = rng.normal(980, 420, n).clip(10, 2500)
    fuel_l = (distance / rng.normal(6.5, 1.4, n)).clip(5, 600)
    toll = (distance * rng.uniform(0.5, 2.0, n)).clip(0, 4500)
    traffic = rng.integers(0, 180, n)
    weather = rng.choice(["None", "Rain", "Fog", "Storm"], size=n, p=[0.65, 0.2, 0.1, 0.05])
    return pd.DataFrame({
        "Order_ID": orders["Order_ID"],
        "Route": route,
        "Distance_KM": np.round(distance, 2),
        "Fuel_Consumption_L": np.round(fuel_l, 2),
        "Toll_Charges_INR": np.round(toll, 2),
        "Traffic_Delay_Minutes": traffic,
        "Weather_Impact": weather,
    })


def make_dummy_delivery(orders: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(orders)
    carriers = rng.choice(["SpeedyLogistics", "BlueDartX", "DelivPro", "ShipFast"], size=n)
    promised = rng.integers(1, 6, n)
    actual = (promised + rng.normal(0.3, 1.1, n)).round().astype(int).clip(1, 14)
    statuses = np.where(actual <= promised, "On-Time",
                        np.where(actual - promised <= 2, "Slightly-Delayed", "Delayed"))
    quality_issue = rng.choice(["Perfect", "Minor Damage", "Packaging Issue", "Missing Item"], size=n,
                               p=[0.75, 0.13, 0.08, 0.04])
    customer_rating = np.clip(rng.normal(4.1, 0.7, n), 1, 5)
    delivery_cost = np.round(rng.normal(950, 260, n).clip(150, 3200), 2)
    df = pd.DataFrame({
        "Order_ID": orders["Order_ID"],
        "Carrier": carriers,
        "Promised_Delivery_Days": promised,
        "Actual_Delivery_Days": actual,
        "Delivery_Status": statuses,
        "Quality_Issue": quality_issue,
        "Customer_Rating": np.round(customer_rating, 2),
        "Delivery_Cost_INR": delivery_cost,
    })
    df["SLA_Delta_Days"] = df["Actual_Delivery_Days"] - df["Promised_Delivery_Days"]
    df["On_Time"] = df["SLA_Delta_Days"] <= 0
    return df


def make_dummy_feedback(orders: pd.DataFrame, rng: np.random.Generator, start: datetime, days: int) -> pd.DataFrame:
    n = len(orders)
    dates = [start + timedelta(days=int(rng.integers(0, days))) for _ in range(n)]
    rating = np.clip(rng.normal(4.0, 0.9, n), 1, 5)
    texts = rng.choice([
        "Great service, very fast delivery!",
        "Packaging could be better.",
        "Arrived on time, as expected.",
        "Minor delay due to traffic.",
        "Driver was helpful.",
        "Product slightly damaged.",
        "Excellent experience overall."
    ], size=n)
    would = rng.choice(["Yes", "No"], size=n, p=[0.82, 0.18])
    issue = rng.choice(["None", "Timing", "Damage", "Wrong Item", "Support"], size=n, p=[0.55, 0.25, 0.1, 0.05, 0.05])
    return pd.DataFrame({
        "Order_ID": orders["Order_ID"],
        "Feedback_Date": pd.to_datetime(dates),
        "Rating": np.round(rating, 2),
        "Feedback_Text": texts,
        "Would_Recommend": would,
        "Issue_Category": issue,
    })


# =========================
# Metrics / KPIs
# =========================
def compute_kpis(
    orders: pd.DataFrame,
    costs: pd.DataFrame,
    routes: pd.DataFrame,
    delivery: pd.DataFrame,
    feedback: pd.DataFrame,
) -> Dict[str, float]:
    total_revenue = float(orders["Order_Value_INR"].sum())
    total_log_cost = float(costs["Total_Logistics_Cost_INR"].sum())
    total_del_cost = float(delivery["Delivery_Cost_INR"].sum())
    total_cost = total_log_cost + total_del_cost
    margin = total_revenue - total_cost
    margin_pct = (margin / total_revenue * 100.0) if total_revenue else float("nan")
    on_time_pct = float(delivery["On_Time"].mean() * 100.0) if len(delivery) else float("nan")
    avg_sla_delta = float(delivery["SLA_Delta_Days"].mean()) if len(delivery) else float("nan")
    avg_rating = float(feedback["Rating"].mean()) if len(feedback) else float("nan")

    d = routes.merge(costs[["Order_ID", "Total_Logistics_Cost_INR"]], on="Order_ID", how="left")
    d["cpk"] = d["Total_Logistics_Cost_INR"] / d["Distance_KM"].replace(0, np.nan)
    avg_cpk = float(d["cpk"].mean())

    return {
        "orders_count": int(orders.shape[0]),
        "total_revenue": round(total_revenue, 2),
        "total_cost": round(total_cost, 2),
        "gross_margin": round(margin, 2),
        "gross_margin_pct": round(margin_pct, 2) if not math.isnan(margin_pct) else float("nan"),
        "on_time_pct": round(on_time_pct, 2) if not math.isnan(on_time_pct) else float("nan"),
        "avg_sla_delta_days": round(avg_sla_delta, 2) if not math.isnan(avg_sla_delta) else float("nan"),
        "avg_cost_per_km": round(avg_cpk, 2) if not math.isnan(avg_cpk) else float("nan"),
        "avg_rating": round(avg_rating, 2) if not math.isnan(avg_rating) else float("nan"),
    }


# =========================
# Charts
# =========================
def chart_cost_stacked(costs: pd.DataFrame) -> alt.Chart:
    parts = ["Fuel_Cost", "Labor_Cost", "Vehicle_Maintenance", "Insurance",
             "Packaging_Cost", "Technology_Platform_Fee", "Other_Overhead"]
    df = costs[["Order_ID"] + parts].copy()
    long = df.melt(id_vars="Order_ID", var_name="Cost_Type", value_name="INR").fillna(0)
    return alt.Chart(long).mark_bar().encode(
        x=alt.X("Order_ID:N", sort="-y", title="Order"),
        y=alt.Y("sum(INR):Q", title="Total Cost (INR)"),
        color=alt.Color("Cost_Type:N"),
        tooltip=["Order_ID", "Cost_Type", alt.Tooltip("INR:Q", format=".2f")],
    ).properties(height=320).interactive()


def chart_sla_bar(delivery: pd.DataFrame) -> alt.Chart:
    df = delivery.copy()
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Delivery_Status:N", title="Status"),
        y=alt.Y("count():Q", title="Orders"),
        color=alt.Color("On_Time:N", title="On Time"),
        tooltip=["Delivery_Status", "On_Time", "count()"],
    ).properties(height=320).interactive()


def chart_leadtime_hist(delivery: pd.DataFrame) -> alt.Chart:
    df = delivery.copy()
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Actual_Delivery_Days:Q", bin=alt.Bin(maxbins=25), title="Actual Delivery Days"),
        y=alt.Y("count():Q", title="Orders"),
        tooltip=[alt.Tooltip("count():Q", title="Orders")],
    ).properties(height=320).interactive()


def chart_rating_line(feedback: pd.DataFrame) -> alt.Chart:
    df = feedback.copy().sort_values("Feedback_Date")
    return alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("Feedback_Date:T", title="Date"),
        y=alt.Y("mean(Rating):Q", title="Avg Rating"),
        tooltip=[alt.Tooltip("mean(Rating):Q", format=".2f"), "count()"],
    ).properties(height=320).interactive()


def chart_route_cost(routes: pd.DataFrame, costs: pd.DataFrame) -> alt.Chart:
    df = routes.merge(costs[["Order_ID", "Total_Logistics_Cost_INR"]], on="Order_ID", how="left")
    return alt.Chart(df).mark_circle(size=100).encode(
        x=alt.X("Distance_KM:Q", title="Distance (KM)"),
        y=alt.Y("Total_Logistics_Cost_INR:Q", title="Logistics Cost (INR)"),
        color=alt.Color("Route:N", title="Route"),
        tooltip=["Order_ID", "Route",
                 alt.Tooltip("Distance_KM:Q", format=".0f"),
                 alt.Tooltip("Total_Logistics_Cost_INR:Q", format=".0f")],
    ).properties(height=340).interactive()


# =========================
# Insight Engine (Chatbox)
# =========================
def fuzzy_match_order_ids(user_text: str, order_ids: List[str], n: int = 5) -> List[str]:
    """
    Find likely order IDs mentioned by user_text using difflib fuzzy match.
    Accepts arbitrary strings; returns top-N similar order IDs.
    """
    user_text_upper = user_text.upper()
    # Extract candidates that look like ORD... if present, else use global match
    tokens = [t for t in user_text_upper.replace(",", " ").split() if any(ch.isalnum() for ch in t)]
    explicit = [t for t in tokens if "ORD" in t]
    candidates = []
    if explicit:
        for tok in explicit:
            candidates.extend(difflib.get_close_matches(tok, order_ids, n=n, cutoff=0.4))
    else:
        # If no explicit token, try entire text against all IDs
        candidates.extend(difflib.get_close_matches(user_text_upper, order_ids, n=n, cutoff=0.4))
    # Deduplicate, preserve order
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:n]


def summarize_overall(kpi: Dict[str, float], delivery: pd.DataFrame, feedback: pd.DataFrame) -> str:
    on_time = kpi.get("on_time_pct", float("nan"))
    avg_sla = kpi.get("avg_sla_delta_days", float("nan"))
    avg_rating = kpi.get("avg_rating", float("nan"))
    delayed_ratio = float((delivery["Delivery_Status"] == "Delayed").mean() * 100) if len(delivery) else float("nan")
    top_issue = (feedback["Issue_Category"].value_counts().idxmax()
                 if ("Issue_Category" in feedback.columns and len(feedback)) else "None")

    lines = [
        f"Orders analysed: {kpi['orders_count']}",
        f"Revenue (₹): {kpi['total_revenue']:.0f} | Total Cost (₹): {kpi['total_cost']:.0f} | Margin (₹): {kpi['gross_margin']:.0f} ({kpi['gross_margin_pct']:.2f}%)" if not math.isnan(kpi['gross_margin_pct']) else
        f"Revenue (₹): {kpi['total_revenue']:.0f} | Total Cost (₹): {kpi['total_cost']:.0f} | Margin (₹): {kpi['gross_margin']:.0f}",
        f"On-time delivery: {on_time:.2f}% | Avg SLA delta (days): {avg_sla:.2f}" if not math.isnan(on_time) and not math.isnan(avg_sla) else
        f"On-time delivery: {on_time:.2f}%" if not math.isnan(on_time) else "On-time delivery: —",
        f"Average customer rating: {avg_rating:.2f}" if not math.isnan(avg_rating) else "Average customer rating: —",
        f"Delayed deliveries: {delayed_ratio:.2f}%" if not math.isnan(delayed_ratio) else "Delayed deliveries: —",
        f"Most frequent issue category: {top_issue}",
    ]
    return "\n".join(lines)


def summarize_orders(
    orders: pd.DataFrame, costs: pd.DataFrame, routes: pd.DataFrame,
    delivery: pd.DataFrame, feedback: pd.DataFrame, order_ids: List[str]
) -> List[str]:
    """
    Per-order bullet summaries. Returns text lines for each matched order.
    """
    out = []
    for oid in order_ids:
        row_o = orders.loc[orders["Order_ID"] == oid]
        row_c = costs.loc[costs["Order_ID"] == oid]
        row_r = routes.loc[routes["Order_ID"] == oid]
        row_d = delivery.loc[delivery["Order_ID"] == oid]
        row_f = feedback.loc[feedback["Order_ID"] == oid]

        if row_o.empty:
            continue

        value = float(row_o["Order_Value_INR"].iloc[0]) if "Order_Value_INR" in row_o.columns else float("nan")
        prio = row_o["Priority"].iloc[0] if "Priority" in row_o.columns else "—"
        route = row_r["Route"].iloc[0] if not row_r.empty else "—"
        distance = float(row_r["Distance_KM"].iloc[0]) if not row_r.empty else float("nan")
        tcost = float(row_c["Total_Logistics_Cost_INR"].iloc[0]) if not row_c.empty else float("nan")
        dcost = float(row_d["Delivery_Cost_INR"].iloc[0]) if not row_d.empty else float("nan")
        status = row_d["Delivery_Status"].iloc[0] if not row_d.empty else "—"
        sla_d = float(row_d["SLA_Delta_Days"].iloc[0]) if not row_d.empty else float("nan")
        rating = float(row_f["Rating"].mean()) if not row_f.empty else float("nan")
        issues = ", ".join(row_f["Issue_Category"].dropna().unique().tolist()) if not row_f.empty else "None"

        bullets = [
            f"Order {oid} | Priority: {prio} | Value ₹{value:.0f}",
            f"Route: {route} | Distance: {distance:.0f} km" if not math.isnan(distance) else f"Route: {route}",
            f"Costs: Logistics ₹{tcost:.0f}, Delivery ₹{dcost:.0f}" if not math.isnan(tcost) and not math.isnan(dcost) else "Costs: —",
            f"Status: {status} | SLA Δ: {sla_d:+.0f} days" if not math.isnan(sla_d) else f"Status: {status}",
            f"Avg Rating: {rating:.2f}" if not math.isnan(rating) else "Avg Rating: —",
            f"Issues: {issues}",
        ]
        out.append("\n".join(bullets))
    return out


def recommend_actions(feedback: pd.DataFrame, delivery: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Produce priority issues sorted by lowest ratings, blended with delivery status.
    Returns a list of dicts: {"issue":..., "avg_rating":..., "actions":...}
    """
    if feedback.empty:
        return []

    # Compute per-issue average rating and frequency
    grp = (feedback.groupby("Issue_Category")
           .agg(avg_rating=("Rating", "mean"), count=("Order_ID", "count"))
           .reset_index())

    # Join delayed share per issue
    deliv_issue = (feedback.merge(delivery[["Order_ID", "Delivery_Status"]], on="Order_ID", how="left")
                   .assign(delayed=lambda d: (d["Delivery_Status"] == "Delayed").astype(int)))
    delay_grp = (deliv_issue.groupby("Issue_Category")
                 .agg(delay_rate=("delayed", "mean"))
                 .reset_index())

    merged = grp.merge(delay_grp, on="Issue_Category", how="left").fillna({"delay_rate": 0.0})

    # Sort: lowest avg_rating first, then by count desc
    merged = merged.sort_values(by=["avg_rating", "count"], ascending=[True, False])

    # Map issues to actions
    action_map = {
        "Timing": "Increase buffer in promised ETA, add proactive delay alerts, route around peak congestion windows.",
        "Damage": "Reinforce packaging SOP, add fragile routing flags, audit last-mile handling for recurring lanes.",
        "Wrong Item": "Tighten pick-pack QA with barcode scans at staging and dispatch; require photo confirmation.",
        "Support": "Introduce 24×7 chat handoff to human within 3 minutes, add order-status quick links in emails.",
        "None": "Maintain current process; periodically A/B test comms for improved CSAT.",
    }

    results = []
    for _, r in merged.iterrows():
        issue = r["Issue_Category"]
        avg_rating = float(r["avg_rating"]) if pd.notna(r["avg_rating"]) else float("nan")
        dr = float(r["delay_rate"]) * 100.0
        actions = action_map.get(issue, "Investigate root causes; implement targeted SOP and monitoring.")
        results.append({
            "issue": str(issue),
            "avg_rating": f"{avg_rating:.2f}" if not math.isnan(avg_rating) else "—",
            "delay_rate": f"{dr:.1f}%",
            "actions": actions,
        })
    return results


def build_text_report(
    overall: str,
    per_order_blocks: List[str],
    priority_items: List[Dict[str, str]],
) -> str:
    """
    Render a plain-text/Markdown report from the chat analysis.
    """
    buf = []
    buf.append("# Logistics Insight Report")
    buf.append("\n## Overall Summary\n")
    buf.append(overall)

    if per_order_blocks:
        buf.append("\n## Per-Order Summaries\n")
        for block in per_order_blocks:
            buf.append(f"{block}\n")

    if priority_items:
        buf.append("\n## Priority Issues & Actions (Sorted by Lowest Avg Rating)\n")
        for item in priority_items:
            buf.append(f"- Issue: {item['issue']} | Avg Rating: {item['avg_rating']} | Delay Rate: {item['delay_rate']}\n  Action: {item['actions']}")
    return "\n".join(buf)


# =========================
# UI Components
# =========================
def set_page() -> None:
    st.set_page_config(
        page_title="AI Logistics — Live Ops",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    alt.data_transformers.disable_max_rows()


def sidebar_filters() -> Dict:
    st.sidebar.header("Global Controls")
    n_orders = st.sidebar.slider("Sample size (orders)", 100, 2000, 400, 100)
    months = st.sidebar.slider("Time window (months)", 1, 12, 6, 1)
    seed = st.sidebar.number_input("Random seed", 0, 9999, 42, 1)
    st.sidebar.caption("Synthetic data only. Designed to look like production.")
    return {"n_orders": n_orders, "months": months, "seed": seed}


def kpi_row(k: Dict[str, float]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orders", f"{k['orders_count']}")
    c2.metric("Revenue (₹)", f"{k['total_revenue']:.0f}")
    c3.metric("Total Cost (₹)", f"{k['total_cost']:.0f}")
    c4.metric("Gross Margin (₹)", f"{k['gross_margin']:.0f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Margin %", f"{k['gross_margin_pct']:.2f}" if not math.isnan(k['gross_margin_pct']) else "—")
    c6.metric("On-Time %", f"{k['on_time_pct']:.2f}" if not math.isnan(k['on_time_pct']) else "—")
    c7.metric("Avg SLA Δ (days)", f"{k['avg_sla_delta_days']:.2f}" if not math.isnan(k['avg_sla_delta_days']) else "—")
    c8.metric("Avg Rating", f"{k['avg_rating']:.2f}" if not math.isnan(k['avg_rating']) else "—")


def downloads_section(dfs: Dict[str, pd.DataFrame]) -> None:
    st.subheader("Downloads")
    cols = st.columns(len(dfs))
    for (name, df), col in zip(dfs.items(), cols):
        data = df.to_csv(index=False).encode("utf-8")
        col.download_button(
            label=f"Download {name}.csv",
            data=data,
            file_name=f"{name}.csv",
            mime="text/csv"
        )


# =========================
# Pages
# =========================
def page_dashboard(orders, costs, routes, delivery, feedback) -> None:
    st.title("Operational Overview")
    st.caption("Live KPIs and drill-down visuals")

    k = compute_kpis(orders, costs, routes, delivery, feedback)
    kpi_row(k)
    st.divider()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Cost Breakdown by Order")
        st.altair_chart(chart_cost_stacked(costs), use_container_width=True)
    with c2:
        st.subheader("Delivery Reliability")
        st.altair_chart(chart_sla_bar(delivery), use_container_width=True)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.subheader("Lead Time Distribution")
        st.altair_chart(chart_leadtime_hist(delivery), use_container_width=True)
    with c4:
        st.subheader("Customer Rating Trend")
        st.altair_chart(chart_rating_line(feedback), use_container_width=True)

    st.subheader("Route Efficiency")
    st.altair_chart(chart_route_cost(routes, costs), use_container_width=True)

    with st.expander("Peek data"):
        st.dataframe(orders.head(50), use_container_width=True)
        st.dataframe(costs.head(50), use_container_width=True)
        st.dataframe(routes.head(50), use_container_width=True)
        st.dataframe(delivery.head(50), use_container_width=True)
        st.dataframe(feedback.head(50), use_container_width=True)


def page_copilot(orders, costs, routes, delivery, feedback) -> None:
    st.title("Ops Copilot")
    st.caption("Ask about orders, health of the operation, or how to improve CSAT. Free-text supported.")

    # Chat history
    if "chat" not in st.session_state:
        st.session_state.chat = []  # list of dicts: {"role": "user"/"assistant", "content": str}

    # Input area
    user_text = st.text_area(
        "Message", placeholder="e.g., Overall health? Any priority issues? What's up with ORD000123 or ORD15?",
        height=120
    )

    colA, colB = st.columns([1, 1])
    with colA:
        run_btn = st.button("Analyze")
    with colB:
        clear_btn = st.button("Clear Chat")

    if clear_btn:
        st.session_state.chat = []

    if run_btn and user_text.strip():
        # Append user message
        st.session_state.chat.append({"role": "user", "content": user_text})

        # Build insights
        k = compute_kpis(orders, costs, routes, delivery, feedback)
        overall = summarize_overall(k, delivery, feedback)

        order_matches = fuzzy_match_order_ids(user_text, orders["Order_ID"].tolist(), n=5)
        per_order = summarize_orders(orders, costs, routes, delivery, feedback, order_matches)

        priorities = recommend_actions(feedback, delivery)

        # Compose assistant message
        report = build_text_report(overall, per_order, priorities)
        st.session_state.chat.append({"role": "assistant", "content": report})

    # Render chat
    for msg in st.session_state.chat:
        if msg["role"] == "user":
            st.markdown(f"**You:**\n\n{msg['content']}")
        else:
            st.markdown(f"**Ops Copilot:**\n\n{msg['content']}")

    # Download last assistant report
    if st.session_state.chat and st.session_state.chat[-1]["role"] == "assistant":
        last = st.session_state.chat[-1]["content"].encode("utf-8")
        st.download_button(
            label="Download Last Insight Report (Markdown)",
            data=last,
            file_name="insight_report.md",
            mime="text/markdown",
            use_container_width=True
        )

    # Quick reference: order list for fuzzy hints
    with st.expander("Order ID directory (for fuzzy matching)"):
        st.write(", ".join(orders["Order_ID"].head(200).tolist()) + (" ..." if len(orders) > 200 else ""))


def page_downloads(orders, costs, routes, delivery, feedback) -> None:
    st.title("Data Exports")
    st.caption("Hand off filtered synthetic datasets downstream")
    downloads_section({
        "orders": orders,
        "costs": costs,
        "routes": routes,
        "delivery": delivery,
        "feedback": feedback,
    })


# =========================
# Orchestration
# =========================
def synthesize_data(n_orders: int, months: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = _rng(seed)
    end = datetime.now()
    start = end - timedelta(days=30 * months)
    orders = make_dummy_orders(n_orders, rng, start, 30 * months)
    costs = make_dummy_costs(orders, rng)
    routes = make_dummy_routes(orders, rng)
    delivery = make_dummy_delivery(orders, rng)
    feedback = make_dummy_feedback(orders, rng, start, 30 * months)
    return orders, costs, routes, delivery, feedback


def route_app(orders, costs, routes, delivery, feedback) -> None:
    tabs = st.tabs(["Dashboard", "Ops Copilot", "Downloads"])
    with tabs[0]:
        page_dashboard(orders, costs, routes, delivery, feedback)
    with tabs[1]:
        page_copilot(orders, costs, routes, delivery, feedback)
    with tabs[2]:
        page_downloads(orders, costs, routes, delivery, feedback)


# =========================
# Main
# =========================
def main() -> None:
    set_page()
    controls = sidebar_filters()
    with st.spinner("Synthesizing operational dataset..."):
        orders, costs, routes, delivery, feedback = synthesize_data(
            n_orders=controls["n_orders"], months=controls["months"], seed=controls["seed"]
        )
    route_app(orders, costs, routes, delivery, feedback)
    st.divider()
    st.caption("Note: All data is synthetic and for demonstration purposes only.")

    # Optional: prime Gemini client (not used automatically)
    _ = get_genai_client()


if __name__ == "__main__":
    # Call main so Streamlit renders the UI.
    main()

