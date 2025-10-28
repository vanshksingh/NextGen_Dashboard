"""
Frontend Orchestrator — enhanced Priority Insights + PDF + collapsible sidebar
Run: streamlit run app.py
"""
from __future__ import annotations

import os
import streamlit as st
import altair as alt
from typing import Dict
from dotenv import load_dotenv

from data_loader import load_data_entrypoint, synthesize_data
from charts import make_all_charts, chart_issue_volume, chart_issue_rating_vs_volume
from metrics import compute_kpis, build_overall_report_text
from ai_chat import (
    init_ai_cache,
    genai_status,
    ai_analyze_query,
    fuzzy_match_order_ids,
    priority_insights_top_n,
    priority_issue_summary,
)
from report import build_download_artifacts
from report_pdf import build_issue_plan_pdf
from utils import bytes_download_button


def _sidebar_controls() -> Dict:
    # Sidebar starts collapsed; controls wrapped in expander
    with st.sidebar:
        with st.expander("Data Source & Options", expanded=False):
            src = st.radio("Use data from", ["CSV", "Synthetic"], index=0)
            cfg = {"src": src}

            if src == "CSV":
                st.caption("Upload CSVs or set a directory path where files exist.")
                data_dir = st.text_input("Directory (optional)", value=os.getcwd())
                up_cost = st.file_uploader("cost_breakdown.csv", type=["csv"], key="up_cost")
                up_cust = st.file_uploader("customer_feedback.csv", type=["csv"], key="up_cust")
                up_delv = st.file_uploader("delivery_performance.csv", type=["csv"], key="up_delv")
                up_ord  = st.file_uploader("orders.csv", type=["csv"], key="up_ord")
                up_rout = st.file_uploader("routes_distance.csv", type=["csv"], key="up_rout")
                up_fleet= st.file_uploader("vehicle_fleet.csv", type=["csv"], key="up_fleet")
                up_wh   = st.file_uploader("warehouse_inventory.csv", type=["csv"], key="up_wh")
                cfg.update(dict(
                    data_dir=data_dir,
                    up_cost=up_cost, up_cust=up_cust, up_delv=up_delv, up_ord=up_ord,
                    up_rout=up_rout, up_fleet=up_fleet, up_wh=up_wh
                ))
            else:
                n = st.slider("Orders", 200, 2000, 600, 100)
                months = st.slider("Window (months)", 1, 12, 6, 1)
                seed = st.number_input("Seed", 0, 9999, 42)
                cfg.update(dict(n=n, months=months, seed=seed))

        with st.expander("Priority Insights Settings", expanded=False):
            num_issues = st.slider("Number of issues (Top-N)", 3, 20, 10, 1)
            cfg["num_issues"] = int(num_issues)

        with st.expander("Ops Copilot Settings", expanded=False):
            st.caption("Answers are cached until you press Done.")
            # room for future knobs (e.g., model selection)

    return cfg


def main() -> None:
    st.set_page_config(
        page_title="AI Logistics — Pro",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="collapsed",   # collapsible by default
    )
    alt.data_transformers.disable_max_rows()
    load_dotenv()
    init_ai_cache()

    cfg = _sidebar_controls()

    # Load data
    if cfg["src"] == "CSV":
        with st.spinner("Loading & cleaning CSV data..."):
            data = load_data_entrypoint(
                data_dir=cfg.get("data_dir", os.getcwd()),
                up_cost=cfg.get("up_cost"), up_cust=cfg.get("up_cust"), up_delv=cfg.get("up_delv"),
                up_ord=cfg.get("up_ord"), up_rout=cfg.get("up_rout"), up_fleet=cfg.get("up_fleet"),
                up_wh=cfg.get("up_wh")
            )
    else:
        with st.spinner("Synthesizing dataset..."):
            data = synthesize_data(
                n_orders=cfg.get("n", 600),
                months=cfg.get("months", 6),
                seed=cfg.get("seed", 42),
            )

    orders = data["orders"]
    costs = data["cost_breakdown"]
    routes = data["routes_distance"]
    delivery = data["delivery_performance"]
    feedback = data["customer_feedback"]

    # KPIs
    kpi = compute_kpis(orders, costs, routes, delivery, feedback)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Ops Copilot", "Priority Insights", "Downloads"])

    # ===== Dashboard =====
    with tab1:
        st.title("Operational Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Orders", f"{kpi['orders_count']}")
        c2.metric("Revenue (₹)", f"{kpi['total_revenue']:.0f}")
        c3.metric("Total Cost (₹)", f"{kpi['total_cost']:.0f}")
        c4.metric("Gross Margin (₹)", f"{kpi['gross_margin']:.0f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Margin %", f"{kpi['gross_margin_pct']:.2f}" if kpi['gross_margin_pct']==kpi['gross_margin_pct'] else "—")
        c6.metric("On-Time %", f"{kpi['on_time_pct']:.2f}" if kpi['on_time_pct']==kpi['on_time_pct'] else "—")
        c7.metric("Avg SLA Δ", f"{kpi['avg_sla_delta_days']:.2f}" if kpi['avg_sla_delta_days']==kpi['avg_sla_delta_days'] else "—")
        c8.metric("Avg Rating", f"{kpi['avg_rating']:.2f}" if kpi['avg_rating']==kpi['avg_rating'] else "—")

        charts = make_all_charts(orders, costs, routes, delivery, feedback)
        colA, colB = st.columns(2, gap="large")
        with colA:
            st.subheader("Cost Breakdown")
            st.altair_chart(charts["cost_breakdown"], use_container_width=True)
        with colB:
            st.subheader("Delivery Reliability")
            st.altair_chart(charts["sla_bar"], use_container_width=True)
        colC, colD = st.columns(2, gap="large")
        with colC:
            st.subheader("Lead Time Distribution")
            st.altair_chart(charts["leadtime_hist"], use_container_width=True)
        with colD:
            st.subheader("Rating Trend")
            st.altair_chart(charts["rating_trend"], use_container_width=True)

        st.subheader("Route Efficiency")
        st.altair_chart(charts["route_cost_scatter"], use_container_width=True)

        st.subheader("Overall Report")
        report_text = build_overall_report_text(kpi, orders, costs, routes, delivery, feedback)
        st.text_area("Report (auto-generated)", report_text, height=220)

    # ===== Ops Copilot =====
    with tab2:
        st.title("Ops Copilot")
        status = genai_status()
        if status["ready"] != "yes":
            st.caption(f"AI fallback active — {status['reason']}")
        else:
            st.caption("Gemini is active for action insights.")

        query = st.text_area("Query", placeholder="e.g., 'Status and actions for ORD000345' or 'order 345 damage?'", height=120)
        done = st.button("Done (Generate / Refresh Answer)")

        cache_key = (query or "").strip()
        if done and cache_key and "ai_cache" in st.session_state:
            st.session_state.ai_cache.pop(cache_key, None)

        response_text = None
        if cache_key:
            if "ai_cache" in st.session_state and cache_key in st.session_state.ai_cache:
                response_text = st.session_state.ai_cache[cache_key]
            else:
                with st.spinner("Analyzing..."):
                    best_ids = fuzzy_match_order_ids(cache_key, orders["Order_ID"].tolist(), top_n=1)
                    response_text = ai_analyze_query(
                        query=cache_key,
                        orders=orders, costs=costs, routes=routes, delivery=delivery, feedback=feedback,
                        matched_ids=best_ids
                    )
                st.session_state.ai_cache[cache_key] = response_text

        if response_text:
            st.markdown("**Ops Copilot Response**")
            st.text_area("Answer", response_text, height=280)
            bytes_download_button("Download Response (txt)", response_text.encode("utf-8"), "copilot_response.txt", "text/plain")

    # ===== Priority Insights =====
    with tab3:
        st.title("Priority Insights")
        n = cfg.get("num_issues", 10)

        # Compute Top-N
        top_issues = priority_insights_top_n(feedback, delivery, n=n)

        if not top_issues:
            st.info("Not enough feedback to compute priority insights.")
        else:
            st.caption(f"Showing Top {len(top_issues)} lowest-rated issues")

            # Charts
            vol_chart = chart_issue_volume(feedback)
            rv_chart = chart_issue_rating_vs_volume(feedback)
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.subheader("Issue Volume")
                st.altair_chart(vol_chart, use_container_width=True)
            with c2:
                st.subheader("Rating vs Volume")
                st.altair_chart(rv_chart, use_container_width=True)

            # Interactive selection of one issue
            issue_list = [ti["issue"] for ti in top_issues]
            sel_issue = st.selectbox("Select an issue to inspect", options=issue_list, index=0)

            # Show cards for all issues
            for idx, item in enumerate(top_issues, start=1):
                with st.container(border=True):
                    st.markdown(f"**{idx}. Issue:** {item['issue']}")
                    st.write(f"Average Rating: {item['avg_rating']}  |  Delay Rate: {item['delay_rate']}")
                    st.write(f"Action: {item['actions']}")

            st.markdown("---")
            st.subheader("Selected Issue — Related Orders & AI Summary")
            if sel_issue:
                # Filter related orders by selected issue
                rel = feedback.loc[feedback["Issue_Category"] == sel_issue, ["Order_ID", "Feedback_Date", "Rating", "Feedback_Text", "Would_Recommend"]].copy()
                # Deduplicate to unique order IDs for table join
                unique_orders = rel["Order_ID"].dropna().unique().tolist()
                show_orders = orders[orders["Order_ID"].isin(unique_orders)][[
                    "Order_ID", "Order_Date", "Priority", "Product_Category", "Order_Value_INR", "Origin", "Destination"
                ]].copy()

                st.markdown(f"**Orders linked to '{sel_issue}'**")
                st.dataframe(show_orders, use_container_width=True, height=300)

                # Gemini-powered issue summary (fallback-safe)
                with st.spinner("Summarizing issue root causes & remediation..."):
                    issue_summary = priority_issue_summary(
                        issue=sel_issue,
                        orders=orders, costs=costs, routes=routes, delivery=delivery, feedback=feedback
                    )
                st.text_area("AI Root-Cause Summary & Actions", issue_summary, height=220)

                # PDF export button
                pdf_bytes = build_issue_plan_pdf(top_issues=top_issues, feedback=feedback, delivery=delivery)
                bytes_download_button("Download Issue Remediation Plan (PDF)", pdf_bytes, "issue_remediation_plan.pdf", "application/pdf")

    # ===== Downloads =====
    with tab4:
        st.title("Downloads")
        charts = make_all_charts(orders, costs, routes, delivery, feedback)
        artifacts = build_download_artifacts(
            orders=orders, costs=costs, routes=routes, delivery=delivery, feedback=feedback, charts=charts
        )

        col1, col2, col3, col4, col5 = st.columns(5)
        bytes_download_button("orders.csv", artifacts["orders.csv"], "orders.csv", "text/csv", container=col1)
        bytes_download_button("costs.csv", artifacts["costs.csv"], "costs.csv", "text/csv", container=col2)
        bytes_download_button("routes.csv", artifacts["routes.csv"], "routes.csv", "text/csv", container=col3)
        bytes_download_button("delivery.csv", artifacts["delivery.csv"], "delivery.csv", "text/csv", container=col4)
        bytes_download_button("feedback.csv", artifacts["feedback.csv"], "feedback.csv", "text/csv", container=col5)

        st.markdown("#### Chart Downloads")
        ch1, ch2, ch3 = st.columns(3)
        bytes_download_button("cost_breakdown.html", artifacts["cost_breakdown.html"], "cost_breakdown.html", "text/html", container=ch1)
        bytes_download_button("sla_bar.html", artifacts["sla_bar.html"], "sla_bar.html", "text/html", container=ch2)
        bytes_download_button("leadtime_hist.html", artifacts["leadtime_hist.html"], "leadtime_hist.html", "text/html", container=ch3)
        ch4, ch5 = st.columns(2)
        bytes_download_button("rating_trend.html", artifacts["rating_trend.html"], "rating_trend.html", "text/html", container=ch4)
        bytes_download_button("route_cost_scatter.html", artifacts["route_cost_scatter.html"], "route_cost_scatter.html", "text/html", container=ch5)


if __name__ == "__main__":
    main()
