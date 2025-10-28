from __future__ import annotations
import pandas as pd
import altair as alt
from typing import Dict

def chart_cost_breakdown(costs: pd.DataFrame) -> alt.Chart:
    parts = ["Fuel_Cost","Labor_Cost","Vehicle_Maintenance","Insurance","Packaging_Cost","Technology_Platform_Fee","Other_Overhead"]
    df = costs[["Order_ID"] + [c for c in parts if c in costs.columns]].copy()
    long = df.melt(id_vars="Order_ID", var_name="Cost_Type", value_name="INR").fillna(0)
    return alt.Chart(long).mark_bar().encode(
        x=alt.X("Order_ID:N", sort="-y", title="Order"),
        y=alt.Y("sum(INR):Q", title="Total Cost (INR)"),
        color=alt.Color("Cost_Type:N"),
        tooltip=["Order_ID","Cost_Type",alt.Tooltip("INR:Q", format=".2f")],
    ).properties(height=320).interactive()

def chart_sla_bar(delivery: pd.DataFrame) -> alt.Chart:
    df = delivery.copy()
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Delivery_Status:N"),
        y=alt.Y("count():Q", title="Orders"),
        color=alt.Color("On_Time:N", title="On Time"),
        tooltip=["Delivery_Status","On_Time","count()"],
    ).properties(height=320).interactive()

def chart_leadtime_hist(delivery: pd.DataFrame) -> alt.Chart:
    df = delivery.copy()
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Actual_Delivery_Days:Q", bin=alt.Bin(maxbins=24), title="Actual Delivery Days"),
        y=alt.Y("count():Q", title="Orders"),
        tooltip=[alt.Tooltip("count():Q", title="Orders")],
    ).properties(height=320).interactive()

def chart_rating_trend(feedback: pd.DataFrame) -> alt.Chart:
    df = feedback.copy().sort_values("Feedback_Date")
    return alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("Feedback_Date:T"),
        y=alt.Y("mean(Rating):Q", title="Avg Rating"),
        tooltip=[alt.Tooltip("mean(Rating):Q", format=".2f"), "count()"],
    ).properties(height=320).interactive()

def chart_route_cost(routes: pd.DataFrame, costs: pd.DataFrame) -> alt.Chart:
    k = "Order_ID"
    df = routes.merge(costs[[k, "Total_Logistics_Cost_INR"]], on=k, how="left")
    return alt.Chart(df).mark_circle(size=100).encode(
        x=alt.X("Distance_KM:Q"),
        y=alt.Y("Total_Logistics_Cost_INR:Q", title="Logistics Cost (INR)"),
        color=alt.Color("Route:N"),
        tooltip=[k,"Route",alt.Tooltip("Distance_KM:Q", format=".0f"),alt.Tooltip("Total_Logistics_Cost_INR:Q", format=".0f")],
    ).properties(height=340).interactive()

# NEW: Issue volume bar
def chart_issue_volume(feedback: pd.DataFrame) -> alt.Chart:
    if feedback.empty or "Issue_Category" not in feedback.columns:
        return alt.Chart(pd.DataFrame({"Issue_Category": [], "count": []})).mark_bar()
    df = (feedback.groupby("Issue_Category", as_index=False)
          .agg(count=("Order_ID", "count")))
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Issue_Category:N", sort="-y", title="Issue"),
        y=alt.Y("count:Q", title="Records"),
        tooltip=["Issue_Category", "count:Q"]
    ).properties(height=320).interactive()

# NEW: Issue rating vs volume
def chart_issue_rating_vs_volume(feedback: pd.DataFrame) -> alt.Chart:
    if feedback.empty or "Issue_Category" not in feedback.columns:
        return alt.Chart(pd.DataFrame({"Issue_Category": [], "count": [], "avg_rating": []})).mark_circle()
    df = (feedback.groupby("Issue_Category", as_index=False)
          .agg(count=("Order_ID", "count"),
               avg_rating=("Rating", "mean")))
    return alt.Chart(df).mark_circle(size=140).encode(
        x=alt.X("count:Q", title="Volume"),
        y=alt.Y("avg_rating:Q", title="Avg Rating"),
        color=alt.Color("Issue_Category:N", title="Issue"),
        tooltip=["Issue_Category", alt.Tooltip("avg_rating:Q", format=".2f"), "count:Q"],
    ).properties(height=320).interactive()

def make_all_charts(orders, costs, routes, delivery, feedback) -> Dict[str, alt.Chart]:
    return {
        "cost_breakdown": chart_cost_breakdown(costs),
        "sla_bar": chart_sla_bar(delivery),
        "leadtime_hist": chart_leadtime_hist(delivery),
        "rating_trend": chart_rating_trend(feedback),
        "route_cost_scatter": chart_route_cost(routes, costs),
    }

if __name__ == "__main__":
    print("charts: import and call from app.")
