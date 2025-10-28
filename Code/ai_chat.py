"""
Ops Copilot with Gemini integration, robust fuzzy match, and Priority Insights Top-N + per-issue AI summary.

Init pattern respected:
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
"""
from __future__ import annotations

import os
import re
import difflib
import time
from typing import List, Dict, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

try:
    import google.genai as genai
except Exception:
    genai = None

try:
    import streamlit as st  # for session cache + small notices
except Exception:
    st = None


# ===== session cache & status =====
def init_ai_cache() -> None:
    if st is not None and "ai_cache" not in st.session_state:
        st.session_state.ai_cache = {}

def genai_status() -> Dict[str, str]:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if genai is None and key:
        return {"ready": "no", "reason": "Gemini key found but google-genai SDK is not installed."}
    if genai is None and not key:
        return {"ready": "no", "reason": "Gemini SDK not installed and GEMINI_API_KEY not set."}
    if genai is not None and not key:
        return {"ready": "no", "reason": "Gemini SDK installed but GEMINI_API_KEY is not set."}
    return {"ready": "yes", "reason": "Gemini SDK and key detected."}

def get_genai_client() -> Optional["genai.Client"]:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if genai is None or not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


# ===== fuzzy match (improved) =====
_ORD_RE = re.compile(r"\bORD(?:ER)?[-_ ]?(\d{1,9})\b", re.IGNORECASE)
_NUM_RE = re.compile(r"\b(\d{3,9})\b")

def _extract_numeric_tokens(text: str) -> List[int]:
    text = text or ""
    nums = [int(m.group(1)) for m in _ORD_RE.finditer(text)]
    if nums:
        return nums
    return [int(m.group(1)) for m in _NUM_RE.finditer(text)]

def _oid_to_int(oid: str) -> Optional[int]:
    m = re.search(r"(\d+)$", oid)
    return int(m.group(1)) if m else None

def _score_candidate(token_str: str, oid: str, token_ints: List[int]) -> float:
    token_up = token_str.upper()
    oid_up = oid.upper()
    if oid_up in token_up or token_up in oid_up:
        if oid_up == token_up:
            return 1.0
        if oid_up.startswith(token_up) or token_up.startswith(oid_up):
            return 0.96
    best_num_score = 0.0
    oid_int = _oid_to_int(oid)
    if token_ints and oid_int is not None:
        for ti in token_ints:
            diff = abs(oid_int - ti)
            if diff == 0:
                best_num_score = max(best_num_score, 0.93)
            else:
                prox = max(0.0, 1.0 - min(diff, 2000) / 2000.0)
                best_num_score = max(best_num_score, 0.70 + 0.20 * prox)
    dl_score = difflib.SequenceMatcher(None, token_up, oid_up).ratio() * 0.80
    return max(best_num_score, dl_score)

def fuzzy_match_order_ids(text: str, order_ids: List[str], top_n: int = 1) -> List[str]:
    if not text or not order_ids:
        return []
    sample = next((oid for oid in order_ids if re.search(r"(\d+)$", oid)), None)
    width = len(re.search(r"(\d+)$", sample).group(1)) if sample else 6
    token_ints = _extract_numeric_tokens(text)
    explicit_tokens = [f"ORD{str(n).zfill(width)}" for n in token_ints]
    token_strs = explicit_tokens + [text]

    scored: List[Tuple[str, float]] = []
    seen = set()
    for oid in order_ids:
        best = 0.0
        for tok in token_strs:
            best = max(best, _score_candidate(tok, oid, token_ints))
        if explicit_tokens and any(oid.upper().startswith(t[:3].upper()) for t in explicit_tokens):
            best = max(best, 0.95)
        if oid not in seen:
            seen.add(oid)
            scored.append((oid, best))

    def tie_key(item: Tuple[str, float]) -> Tuple[float, int]:
        oid, sc = item
        oi = _oid_to_int(oid) or 10**9
        if token_ints:
            best_diff = min(abs(oi - ti) for ti in token_ints)
        else:
            best_diff = 10**9
        return (sc, -best_diff)

    scored.sort(key=tie_key, reverse=True)
    return [oid for oid, _ in scored[:top_n]]


# ===== record builders & local format =====
def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None

def _row_or_none(df: pd.DataFrame, oid: str) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    m = df.loc[df["Order_ID"] == oid]
    if m.empty:
        return None
    return m.iloc[0]

def build_order_record(oid: str, orders: pd.DataFrame, costs: pd.DataFrame, routes: pd.DataFrame,
                       delivery: pd.DataFrame, feedback: pd.DataFrame) -> Dict:
    ro, rc, rr, rd = (_row_or_none(orders, oid), _row_or_none(costs, oid),
                      _row_or_none(routes, oid), _row_or_none(delivery, oid))
    rf_rows = feedback.loc[feedback["Order_ID"] == oid] if (feedback is not None and not feedback.empty) else pd.DataFrame()
    rec = {
        "order_id": oid,
        "order": {
            "date": str(ro.get("Order_Date")) if ro is not None and "Order_Date" in ro else None,
            "segment": ro.get("Customer_Segment") if ro is not None else None,
            "priority": ro.get("Priority") if ro is not None else None,
            "category": ro.get("Product_Category") if ro is not None else None,
            "value_inr": _safe_float(ro.get("Order_Value_INR")) if ro is not None else None,
            "origin": ro.get("Origin") if ro is not None else None,
            "destination": ro.get("Destination") if ro is not None else None,
            "special_handling": ro.get("Special_Handling") if ro is not None else None,
        },
        "costs": {
            "fuel": _safe_float(rc.get("Fuel_Cost")) if rc is not None else None,
            "labor": _safe_float(rc.get("Labor_Cost")) if rc is not None else None,
            "vehicle_maintenance": _safe_float(rc.get("Vehicle_Maintenance")) if rc is not None else None,
            "insurance": _safe_float(rc.get("Insurance")) if rc is not None else None,
            "packaging": _safe_float(rc.get("Packaging_Cost")) if rc is not None else None,
            "technology_platform_fee": _safe_float(rc.get("Technology_Platform_Fee")) if rc is not None else None,
            "other_overhead": _safe_float(rc.get("Other_Overhead")) if rc is not None else None,
            "total_logistics_cost_inr": _safe_float(rc.get("Total_Logistics_Cost_INR")) if rc is not None else None,
        },
        "route": {
            "route": rr.get("Route") if rr is not None else None,
            "distance_km": _safe_float(rr.get("Distance_KM")) if rr is not None else None,
            "fuel_consumption_l": _safe_float(rr.get("Fuel_Consumption_L")) if rr is not None else None,
            "toll_charges_inr": _safe_float(rr.get("Toll_Charges_INR")) if rr is not None else None,
            "traffic_delay_minutes": _safe_float(rr.get("Traffic_Delay_Minutes")) if rr is not None else None,
            "weather_impact": rr.get("Weather_Impact") if rr is not None else None,
        },
        "delivery": {
            "carrier": rd.get("Carrier") if rd is not None else None,
            "promised_days": _safe_float(rd.get("Promised_Delivery_Days")) if rd is not None else None,
            "actual_days": _safe_float(rd.get("Actual_Delivery_Days")) if rd is not None else None,
            "status": rd.get("Delivery_Status") if rd is not None else None,
            "quality_issue": rd.get("Quality_Issue") if rd is not None else None,
            "customer_rating": _safe_float(rd.get("Customer_Rating")) if rd is not None else None,
            "delivery_cost_inr": _safe_float(rd.get("Delivery_Cost_INR")) if rd is not None else None,
            "sla_delta_days": _safe_float(rd.get("SLA_Delta_Days")) if rd is not None else None,
            "on_time": bool(rd.get("On_Time")) if rd is not None and "On_Time" in rd else None,
        },
        "feedback": [],
    }
    if rf_rows is not None and not rf_rows.empty:
        rf_rows = rf_rows.sort_values("Feedback_Date").tail(8)
        for _, r in rf_rows.iterrows():
            rec["feedback"].append({
                "date": str(r.get("Feedback_Date")),
                "rating": _safe_float(r.get("Rating")),
                "text": r.get("Feedback_Text"),
                "would_recommend": r.get("Would_Recommend"),
                "issue_category": r.get("Issue_Category"),
            })
    return rec

def format_order_context_text(record: Dict) -> str:
    fb = record.get("feedback", []) or []
    o, c, r, d = record["order"], record["costs"], record["route"], record["delivery"]
    lines = [
        f"ORDER {record.get('order_id')}",
        f"Date: {o.get('date')}  Segment: {o.get('segment')}  Priority: {o.get('priority')}  Category: {o.get('category')}",
        f"Origin→Destination: {o.get('origin')} → {o.get('destination')}  Special: {o.get('special_handling')}",
        f"Order Value (₹): {o.get('value_inr')}",
        f"Costs (₹): fuel={c.get('fuel')}, labor={c.get('labor')}, maint={c.get('vehicle_maintenance')}, ins={c.get('insurance')}, pack={c.get('packaging')}, tech={c.get('technology_platform_fee')}, other={c.get('other_overhead')}  | Total: {c.get('total_logistics_cost_inr')}",
        f"Route: {r.get('route')}  Dist(km)={r.get('distance_km')}  Fuel(L)={r.get('fuel_consumption_l')}  Toll(₹)={r.get('toll_charges_inr')}  Traffic(min)={r.get('traffic_delay_minutes')}  Weather={r.get('weather_impact')}",
        f"Delivery: carrier={d.get('carrier')} promised={d.get('promised_days')}d actual={d.get('actual_days')}d status={d.get('status')} SLAΔ={d.get('sla_delta_days')} on_time={d.get('on_time')} quality={d.get('quality_issue')} rating={d.get('customer_rating')} cost(₹)={d.get('delivery_cost_inr')}",
    ]
    if fb:
        lines.append("Recent Feedback:")
        for f in fb:
            lines.append(f"- {f['date']}: rating={f['rating']} recommend={f['would_recommend']} issue={f['issue_category']} text={f['text']}")
    else:
        lines.append("Recent Feedback: none")
    return "\n".join(lines)

def _compose_per_order_text_from_record(record: Dict) -> str:
    o, c, r, d = record["order"], record["costs"], record["route"], record["delivery"]
    oid = record.get("order_id", "—")
    value = o.get("value_inr")
    pr = o.get("priority", "—")
    route = r.get("route", "—")
    dist = r.get("distance_km")
    tcost = c.get("total_logistics_cost_inr")
    dcost = d.get("delivery_cost_inr")
    status = d.get("status", "—")
    sla = d.get("sla_delta_days")
    rating = d.get("customer_rating")
    issues = sorted(set([fb.get("issue_category") for fb in record.get("feedback", []) if fb.get("issue_category")]))

    lines = [
        f"Order {oid} | Priority: {pr} | Value ₹{value:.0f}" if value is not None else f"Order {oid} | Priority: {pr}",
        f"Route: {route} | Distance: {dist:.0f} km" if dist is not None else f"Route: {route}",
        f"Costs: Logistics ₹{tcost:.0f}, Delivery ₹{dcost:.0f}" if (tcost is not None and dcost is not None) else "Costs: —",
        f"Status: {status} | SLA Δ: {sla:+.0f} days" if sla is not None else f"Status: {status}",
        f"Latest Delivery Rating: {rating:.2f}" if rating is not None else "Latest Delivery Rating: —",
        f"Issues seen: {', '.join(issues) if issues else 'None'}",
        "Customer Happiness Actions: If delayed, send proactive ETA SMS/email; consider goodwill coupon for low ratings; add packaging QA for 'Damage' issues.",
    ]
    return "\n".join(lines)


# ===== Gemini helpers + extraction =====
def _extract_text(resp) -> Optional[str]:
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    cand = getattr(resp, "candidates", None)
    if isinstance(cand, list) and cand:
        content = getattr(cand[0], "content", None)
        parts = getattr(content, "parts", None)
        if isinstance(parts, list):
            buf = []
            for p in parts:
                t = getattr(p, "text", None) or (p if isinstance(p, str) else None)
                if isinstance(t, str) and t.strip():
                    buf.append(t.strip())
            if buf:
                return "\n".join(buf).strip()
    try:
        s = str(resp)
        return s.strip() or None
    except Exception:
        return None

def _gemini(client: "genai.Client", prompt: str, model: str = "gemini-2.0-flash") -> str:
    last_err = None
    for _ in range(3):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            text = _extract_text(resp)
            if isinstance(text, str) and text.strip():
                return text.strip()
        except Exception as e:
            last_err = e
            time.sleep(0.6)
    raise RuntimeError(f"Gemini generation failed: {last_err}")


# ===== public: per-order analyze =====
def ai_analyze_query(query: str, orders: pd.DataFrame, costs: pd.DataFrame, routes: pd.DataFrame,
                     delivery: pd.DataFrame, feedback: pd.DataFrame, matched_ids: List[str]) -> str:
    if not matched_ids:
        return "No matching order ID detected. Please include or hint an Order_ID like 'ORD000123'."

    oid = matched_ids[0]
    record = build_order_record(oid, orders, costs, routes, delivery, feedback)
    context_text = format_order_context_text(record)

    client = get_genai_client()
    if client is not None:
        try:
            prompt = (
                "You are an operations analyst. Read the order context below and produce a concise, "
                "action-oriented summary for ONE order. Output exactly these sections:\n"
                "1) Status — one sentence, include on-time/delay info and SLA delta.\n"
                "2) Cost View — one sentence on logistics + delivery cost vs order value, flag anomalies.\n"
                "3) Customer Happiness — one sentence referencing rating and issues.\n"
                "4) Actions — 3 bullet points, specific operational steps.\n"
                "5) Risks — 1–2 short bullets, if any.\n\n"
                f"Order Context:\n{context_text}\n"
            )
            ai_text = _gemini(client, prompt)
            return f"Order Insight — {oid}\n{ai_text}"
        except Exception as e:
            if st is not None:
                st.info(f"Using local fallback (Gemini error: {type(e).__name__}).")

    return _compose_per_order_text_from_record(record)


# ===== public: Priority Insights Top-N =====
def priority_insights_top_n(feedback: pd.DataFrame, delivery: pd.DataFrame, n: int = 10) -> List[Dict[str, str]]:
    if feedback is None or feedback.empty:
        return []
    grp = (feedback.groupby("Issue_Category")
           .agg(avg_rating=("Rating", "mean"), count=("Order_ID", "count"))
           .reset_index())
    deliv_issue = (feedback.merge(delivery[["Order_ID","Delivery_Status"]], on="Order_ID", how="left")
                   .assign(delayed=lambda d: (d["Delivery_Status"]=="Delayed").astype(int)))
    delay = (deliv_issue.groupby("Issue_Category")
             .agg(delay_rate=("delayed", "mean"))
             .reset_index())
    merged = grp.merge(delay, on="Issue_Category", how="left").fillna({"delay_rate":0.0})
    merged = merged.sort_values(by=["avg_rating","count"], ascending=[True,False]).head(n)

    action_map = {
        "Timing": "Increase ETA buffers; proactive delay alerts; reroute around peak congestion.",
        "Damage": "Reinforce packaging SOP; fragile labeling; audit last-mile handling on recurring lanes.",
        "Wrong Item": "Tighten pick-pack QA with barcode scans; photo confirmation at dispatch.",
        "Support": "Reduce first response <3 min; clear escalation; add order-status quick links.",
        "None": "Maintain SOPs; A/B test customer comms for incremental CSAT gains.",
    }

    out = []
    for _, r in merged.iterrows():
        issue = str(r["Issue_Category"])
        avg_rating = float(r["avg_rating"]) if pd.notna(r["avg_rating"]) else float("nan")
        dr = float(r["delay_rate"]) * 100.0
        out.append({
            "issue": issue,
            "avg_rating": f"{avg_rating:.2f}" if pd.notna(avg_rating) else "—",
            "delay_rate": f"{dr:.1f}%",
            "actions": action_map.get(issue, "Investigate root causes; implement targeted SOP and monitoring."),
        })
    return out


# ===== public: Gemini summary per issue (fallback-safe) =====
def priority_issue_summary(issue: str, orders: pd.DataFrame, costs: pd.DataFrame, routes: pd.DataFrame,
                           delivery: pd.DataFrame, feedback: pd.DataFrame) -> str:
    """
    Build an issue-level context (related orders + basic aggregates) and ask Gemini to propose root-causes & actions.
    Falls back to deterministic summary if Gemini is unavailable.
    """
    if not issue:
        return "No issue selected."

    rel = feedback.loc[feedback["Issue_Category"] == issue]
    if rel.empty:
        return f"No records for issue '{issue}'."

    # small aggregate context
    avg_rating = rel["Rating"].mean() if "Rating" in rel.columns else None
    orders_set = rel["Order_ID"].dropna().unique().tolist()
    deliv = delivery[delivery["Order_ID"].isin(orders_set)]
    on_time_rate = deliv["On_Time"].mean() if "On_Time" in deliv.columns and not deliv.empty else None
    delay_share = (deliv["Delivery_Status"] == "Delayed").mean() if "Delivery_Status" in deliv.columns and not deliv.empty else None
    carriers = ", ".join(deliv["Carrier"].value_counts().head(5).index) if "Carrier" in deliv.columns and not deliv.empty else "—"
    lanes = routes[routes["Order_ID"].isin(orders_set)] if not routes.empty else pd.DataFrame()
    top_lanes = ", ".join(lanes["Route"].value_counts().head(5).index) if "Route" in lanes.columns and not lanes.empty else "—"

    context_lines = [
        f"Issue: {issue}",
        f"Orders affected: {len(orders_set)}",
        f"Avg Rating: {avg_rating:.2f}" if avg_rating == avg_rating else "Avg Rating: —",
        f"On-Time Rate: {on_time_rate*100:.2f}%" if on_time_rate == on_time_rate else "On-Time Rate: —",
        f"Delayed Share: {delay_share*100:.2f}%" if delay_share == delay_share else "Delayed Share: —",
        f"Top Carriers: {carriers}",
        f"Top Lanes: {top_lanes}",
    ]

    client = get_genai_client()
    if client is not None:
        prompt = (
            "You are an operations analyst. Read the issue context below and produce root-cause hypotheses "
            "and a concrete remediation plan suitable for a weekly ops review. "
            "Return:\n"
            "- Root Causes (3 bullets)\n"
            "- Immediate Actions (3 bullets)\n"
            "- Structural Fixes (2 bullets)\n"
            "- KPI to Watch (2)\n\n"
            "Issue Context:\n" + "\n".join(context_lines)
        )
        try:
            ai = _gemini(client, prompt)
            return ai
        except Exception as e:
            if st is not None:
                st.info(f"Using local fallback (Gemini error: {type(e).__name__}).")

    # fallback deterministic text
    fallback = [
        "Root Causes:",
        "- Process variability on affected lanes and carriers.",
        "- Inadequate packaging SOP adherence for high-risk categories.",
        "- Customer comms lag during delays, leading to lower perceived service quality.",
        "",
        "Immediate Actions:",
        "- Add proactive SLA breach alerts and ETA updates to customers.",
        "- Reinforce packaging checks for affected product categories/lane hubs.",
        "- QA last-mile handling partners on lanes with high damage/delay reports.",
        "",
        "Structural Fixes:",
        "- Introduce pick-pack barcode scans and photo confirmation before dispatch.",
        "- Implement carrier scorecards tied to on-time and damage metrics.",
        "",
        "KPI to Watch:",
        "- On-Time Percentage by lane/shift.",
        "- Issue frequency per 1,000 orders (7-day rolling).",
    ]
    return "\n".join(fallback)


if __name__ == "__main__":
    load_dotenv()
    print("google-genai available:", bool(genai))
    print("has GEMINI_API_KEY:", bool(os.getenv("GEMINI_API_KEY")))
