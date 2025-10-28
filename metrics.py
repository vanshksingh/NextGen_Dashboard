from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict

def compute_kpis(orders: pd.DataFrame, costs: pd.DataFrame, routes: pd.DataFrame, delivery: pd.DataFrame, feedback: pd.DataFrame) -> Dict[str, float]:
    total_revenue = float(orders["Order_Value_INR"].sum()) if "Order_Value_INR" in orders.columns else 0.0
    total_log_cost = float(costs["Total_Logistics_Cost_INR"].sum()) if "Total_Logistics_Cost_INR" in costs.columns else 0.0
    total_del_cost = float(delivery["Delivery_Cost_INR"].sum()) if "Delivery_Cost_INR" in delivery.columns else 0.0
    total_cost = total_log_cost + total_del_cost
    gross_margin = total_revenue - total_cost
    gm_pct = (gross_margin / total_revenue * 100.0) if total_revenue else np.nan
    on_time = (delivery["On_Time"].mean() * 100.0) if "On_Time" in delivery.columns and len(delivery) else np.nan
    sla_delta = float(delivery["SLA_Delta_Days"].mean()) if "SLA_Delta_Days" in delivery.columns and len(delivery) else np.nan
    avg_rating = float(feedback["Rating"].mean()) if "Rating" in feedback.columns and len(feedback) else np.nan

    # avg cost per km
    cpk = np.nan
    if "Total_Logistics_Cost_INR" in costs.columns and "Distance_KM" in routes.columns:
        j = costs[["Order_ID","Total_Logistics_Cost_INR"]].merge(routes[["Order_ID","Distance_KM"]], on="Order_ID", how="left")
        j["cpk"] = j["Total_Logistics_Cost_INR"] / j["Distance_KM"].replace(0,np.nan)
        cpk = float(j["cpk"].mean())

    return {
        "orders_count": int(orders.shape[0]),
        "total_revenue": float(total_revenue),
        "total_cost": float(total_cost),
        "gross_margin": float(gross_margin),
        "gross_margin_pct": float(gm_pct) if not pd.isna(gm_pct) else np.nan,
        "on_time_pct": float(on_time) if not pd.isna(on_time) else np.nan,
        "avg_sla_delta_days": float(sla_delta) if not pd.isna(sla_delta) else np.nan,
        "avg_cost_per_km": float(cpk) if not pd.isna(cpk) else np.nan,
        "avg_rating": float(avg_rating) if not pd.isna(avg_rating) else np.nan,
    }

def build_overall_report_text(kpi: Dict[str, float], orders, costs, routes, delivery, feedback) -> str:
    lines = []
    lines.append(f"Orders: {kpi['orders_count']}")
    lines.append(f"Revenue: ₹{kpi['total_revenue']:.2f}")
    lines.append(f"Total Cost: ₹{kpi['total_cost']:.2f}")
    lines.append(f"Gross Margin: ₹{kpi['gross_margin']:.2f} ({kpi['gross_margin_pct']:.2f}%)" if kpi['gross_margin_pct']==kpi['gross_margin_pct'] else
                 f"Gross Margin: ₹{kpi['gross_margin']:.2f}")
    lines.append(f"On-Time %: {kpi['on_time_pct']:.2f}" if kpi['on_time_pct']==kpi['on_time_pct'] else "On-Time %: —")
    lines.append(f"Avg SLA Δ (days): {kpi['avg_sla_delta_days']:.2f}" if kpi['avg_sla_delta_days']==kpi['avg_sla_delta_days'] else "Avg SLA Δ (days): —")
    lines.append(f"Avg Cost / KM: ₹{kpi['avg_cost_per_km']:.2f}" if kpi['avg_cost_per_km']==kpi['avg_cost_per_km'] else "Avg Cost / KM: —")
    lines.append(f"Avg Rating: {kpi['avg_rating']:.2f}" if kpi['avg_rating']==kpi['avg_rating'] else "Avg Rating: —")
    return "\n".join(lines)

if __name__ == "__main__":
    print("metrics: import and call from app.")
