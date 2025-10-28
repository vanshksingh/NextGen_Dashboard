"""
CSV & Synthetic Data Loaders with cleaning, error handling
"""
from __future__ import annotations
import io
import os
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta

# ---------- Helpers ----------
def _read_csv_either(path: Optional[str], upload) -> pd.DataFrame:
    """
    Read from a Streamlit UploadedFile or from disk path if present.
    Returns empty DataFrame on failure, with coerced dtypes later.
    """
    try:
        if upload is not None:
            return pd.read_csv(upload)
        if path and os.path.exists(path):
            return pd.read_csv(path)
    except Exception as e:
        # Fall-through to empty df on any error
        pass
    return pd.DataFrame()

def _coerce_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _coerce_date(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def _fillna_str(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")
    return df

# ---------- Cleaning ----------
def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=[
        "Order_ID","Order_Date","Customer_Segment","Priority","Product_Category",
        "Order_Value_INR","Origin","Destination","Special_Handling"
    ])
    df = _coerce_date(df, ["Order_Date"])
    df = _coerce_num(df, ["Order_Value_INR"])
    df = _fillna_str(df, ["Customer_Segment","Priority","Product_Category","Origin","Destination","Special_Handling"])
    return df

def clean_costs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=[
        "Order_ID","Fuel_Cost","Labor_Cost","Vehicle_Maintenance","Insurance",
        "Packaging_Cost","Technology_Platform_Fee","Other_Overhead","Total_Logistics_Cost_INR"
    ])
    num_cols = ["Fuel_Cost","Labor_Cost","Vehicle_Maintenance","Insurance","Packaging_Cost","Technology_Platform_Fee","Other_Overhead"]
    df = _coerce_num(df, num_cols)
    if "Total_Logistics_Cost_INR" not in df.columns:
        df["Total_Logistics_Cost_INR"] = df[num_cols].sum(axis=1, min_count=1)
    df["Total_Logistics_Cost_INR"] = pd.to_numeric(df["Total_Logistics_Cost_INR"], errors="coerce")
    return df

def clean_delivery(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=[
        "Order_ID","Carrier","Promised_Delivery_Days","Actual_Delivery_Days","Delivery_Status",
        "Quality_Issue","Customer_Rating","Delivery_Cost_INR","SLA_Delta_Days","On_Time"
    ])
    df = _coerce_num(df, ["Promised_Delivery_Days","Actual_Delivery_Days","Customer_Rating","Delivery_Cost_INR"])
    if "SLA_Delta_Days" not in df.columns:
        df["SLA_Delta_Days"] = df["Actual_Delivery_Days"] - df["Promised_Delivery_Days"]
    if "On_Time" not in df.columns:
        df["On_Time"] = df["SLA_Delta_Days"].le(0)
    return df

def clean_routes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=[
        "Order_ID","Route","Distance_KM","Fuel_Consumption_L","Toll_Charges_INR","Traffic_Delay_Minutes","Weather_Impact"
    ])
    df = _coerce_num(df, ["Distance_KM","Fuel_Consumption_L","Toll_Charges_INR","Traffic_Delay_Minutes"])
    return df

def clean_feedback(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=[
        "Order_ID","Feedback_Date","Rating","Feedback_Text","Would_Recommend","Issue_Category"
    ])
    df = _coerce_date(df, ["Feedback_Date"])
    df = _coerce_num(df, ["Rating"])
    df = _fillna_str(df, ["Feedback_Text","Would_Recommend","Issue_Category"])
    return df

def clean_fleet(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=[
        "Vehicle_ID","Vehicle_Type","Capacity_KG","Fuel_Efficiency_KM_per_L",
        "Current_Location","Status","Age_Years","CO2_Emissions_Kg_per_KM"
    ])
    df = _coerce_num(df, ["Capacity_KG","Fuel_Efficiency_KM_per_L","Age_Years","CO2_Emissions_Kg_per_KM"])
    return df

def clean_warehouse(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=[
        "Warehouse_ID","Location","Product_Category","Current_Stock_Units","Reorder_Level","Storage_Cost_per_Unit","Last_Restocked_Date"
    ])
    df = _coerce_date(df, ["Last_Restocked_Date"])
    df = _coerce_num(df, ["Current_Stock_Units","Reorder_Level","Storage_Cost_per_Unit"])
    return df

# ---------- Entrypoints ----------
def load_data_entrypoint(
    data_dir: str,
    up_cost, up_cust, up_delv, up_ord, up_rout, up_fleet, up_wh
) -> Dict[str, pd.DataFrame]:
    # Prefer uploaded files; else try disk; else empty
    cost = _read_csv_either(os.path.join(data_dir, "cost_breakdown.csv"), up_cost)
    cust = _read_csv_either(os.path.join(data_dir, "customer_feedback.csv"), up_cust)
    delv = _read_csv_either(os.path.join(data_dir, "delivery_performance.csv"), up_delv)
    orde = _read_csv_either(os.path.join(data_dir, "orders.csv"), up_ord)
    rout = _read_csv_either(os.path.join(data_dir, "routes_distance.csv"), up_rout)
    fleet= _read_csv_either(os.path.join(data_dir, "vehicle_fleet.csv"), up_fleet)
    ware = _read_csv_either(os.path.join(data_dir, "warehouse_inventory.csv"), up_wh)

    return {
        "orders": clean_orders(orde),
        "cost_breakdown": clean_costs(cost),
        "delivery_performance": clean_delivery(delv),
        "routes_distance": clean_routes(rout),
        "customer_feedback": clean_feedback(cust),
        "vehicle_fleet": clean_fleet(fleet),
        "warehouse_inventory": clean_warehouse(ware),
    }

# Synthetic generator (for app toggle)
def synthesize_data(n_orders: int, months: int, seed: int) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    end = datetime.now()
    start = end - timedelta(days=30 * months)

    # orders
    oid = [f"ORD{str(i).zfill(6)}" for i in range(1, n_orders + 1)]
    dates = pd.to_datetime([start + timedelta(days=int(rng.integers(0, 30*months))) for _ in range(n_orders)])
    orders = pd.DataFrame({
        "Order_ID": oid,
        "Order_Date": dates,
        "Customer_Segment": rng.choice(["Individual","SMB","Enterprise"], n_orders, p=[0.5,0.3,0.2]),
        "Priority": rng.choice(["Standard","Express","Critical"], n_orders, p=[0.6,0.3,0.1]),
        "Product_Category": rng.choice(["Electronics","Industrial","Apparel","FMCG"], n_orders),
        "Order_Value_INR": np.round(rng.normal(12000,3500,n_orders).clip(1500,40000),2),
        "Origin": rng.choice(["Delhi","Mumbai","Kolkata","Chennai","Hyderabad","Bengaluru"], n_orders),
        "Destination": rng.choice(["Delhi","Mumbai","Kolkata","Chennai","Hyderabad","Bengaluru"], n_orders),
        "Special_Handling": rng.choice(["None","Fragile","Cold-Chain","Hazmat"], n_orders, p=[0.7,0.15,0.1,0.05]),
    })
    # costs
    cost = pd.DataFrame({
        "Order_ID": oid,
        "Fuel_Cost": np.round(rng.normal(1800,500,n_orders).clip(200,4000),2),
        "Labor_Cost": np.round(rng.normal(1200,400,n_orders).clip(150,3000),2),
        "Vehicle_Maintenance": np.round(rng.normal(600,200,n_orders).clip(50,1500),2),
        "Insurance": np.round(rng.normal(300,100,n_orders).clip(50,800),2),
        "Packaging_Cost": np.round(rng.normal(250,120,n_orders).clip(50,700),2),
        "Technology_Platform_Fee": np.round(rng.normal(180,60,n_orders).clip(50,600),2),
        "Other_Overhead": np.round(rng.normal(200,90,n_orders).clip(0,800),2),
    })
    cost["Total_Logistics_Cost_INR"] = cost[[
        "Fuel_Cost","Labor_Cost","Vehicle_Maintenance","Insurance",
        "Packaging_Cost","Technology_Platform_Fee","Other_Overhead"
    ]].sum(axis=1)

    # routes
    route = (orders["Origin"] + "-" + orders["Destination"]).tolist()
    distance = rng.normal(980,420,n_orders).clip(10,2500)
    routes = pd.DataFrame({
        "Order_ID": oid,
        "Route": route,
        "Distance_KM": np.round(distance,2),
        "Fuel_Consumption_L": np.round(distance / rng.normal(6.5,1.4,n_orders),2),
        "Toll_Charges_INR": np.round(distance * rng.uniform(0.5,2.0,n_orders),2),
        "Traffic_Delay_Minutes": rng.integers(0,180,n_orders),
        "Weather_Impact": rng.choice(["None","Rain","Fog","Storm"], n_orders, p=[0.65,0.2,0.1,0.05]),
    })

    # delivery
    promised = rng.integers(1,6,n_orders)
    actual = (promised + rng.normal(0.3,1.1,n_orders)).round().astype(int).clip(1,14)
    status = np.where(actual <= promised, "On-Time", np.where(actual - promised <= 2,"Slightly-Delayed","Delayed"))
    delivery = pd.DataFrame({
        "Order_ID": oid,
        "Carrier": rng.choice(["SpeedyLogistics","BlueDartX","DelivPro","ShipFast"], n_orders),
        "Promised_Delivery_Days": promised,
        "Actual_Delivery_Days": actual,
        "Delivery_Status": status,
        "Quality_Issue": rng.choice(["Perfect","Minor Damage","Packaging Issue","Missing Item"], n_orders, p=[0.75,0.13,0.08,0.04]),
        "Customer_Rating": np.round(np.clip(rng.normal(4.1,0.7,n_orders),1,5),2),
        "Delivery_Cost_INR": np.round(rng.normal(950,260,n_orders).clip(150,3200),2),
    })
    delivery["SLA_Delta_Days"] = delivery["Actual_Delivery_Days"] - delivery["Promised_Delivery_Days"]
    delivery["On_Time"] = delivery["SLA_Delta_Days"] <= 0

    # feedback
    fb_dates = [start + timedelta(days=int(rng.integers(0, 30*months))) for _ in range(n_orders)]
    feedback = pd.DataFrame({
        "Order_ID": oid,
        "Feedback_Date": pd.to_datetime(fb_dates),
        "Rating": np.round(np.clip(rng.normal(4.0,0.9,n_orders),1,5),2),
        "Feedback_Text": rng.choice([
            "Great service, very fast delivery!", "Packaging could be better.", "Arrived on time, as expected.",
            "Minor delay due to traffic.", "Driver was helpful.", "Product slightly damaged.",
            "Excellent experience overall."
        ], n_orders),
        "Would_Recommend": rng.choice(["Yes","No"], n_orders, p=[0.82,0.18]),
        "Issue_Category": rng.choice(["None","Timing","Damage","Wrong Item","Support"], n_orders, p=[0.55,0.25,0.1,0.05,0.05]),
    })

    return {
        "orders": clean_orders(orders),
        "cost_breakdown": clean_costs(cost),
        "delivery_performance": clean_delivery(delivery),
        "routes_distance": clean_routes(routes),
        "customer_feedback": clean_feedback(feedback),
        "vehicle_fleet": clean_fleet(pd.DataFrame()),
        "warehouse_inventory": clean_warehouse(pd.DataFrame()),
    }

if __name__ == "__main__":
    # simple sanity check
    d = synthesize_data(300, 6, 42)
    for k, v in d.items():
        print(k, v.shape)
