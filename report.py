# report.py
"""
Chart & Data Export Utilities
- Exports dataframes as CSV bytes
- Exports Altair charts as standalone HTML (downloadable)
- Creates a ZIP bundle of everything
"""

from __future__ import annotations
import io
import zipfile
from typing import Dict
import pandas as pd
import altair as alt


def _df_bytes_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _chart_html_bytes(ch: alt.Chart, title: str) -> bytes:
    """
    Produce a fully self-contained HTML for an Altair chart using vega-embed via CDN.
    Opens offline in any browser; no Python needed.
    """
    spec = ch.to_json()
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
  <style>html,body,#vis{{height:100%;margin:0;padding:0}}</style>
</head>
<body>
  <div id="vis"></div>
  <script>
    const spec = {spec};
    vegaEmbed('#vis', spec, {{ actions: false }});
  </script>
</body>
</html>"""
    return html.encode("utf-8")


def build_download_artifacts(
    orders: pd.DataFrame,
    costs: pd.DataFrame,
    routes: pd.DataFrame,
    delivery: pd.DataFrame,
    feedback: pd.DataFrame,
    charts: Dict[str, alt.Chart],
) -> Dict[str, bytes]:
    """
    Returns a dict of file_name -> bytes for:
      - orders.csv, costs.csv, routes.csv, delivery.csv, feedback.csv
      - cost_breakdown.html, sla_bar.html, leadtime_hist.html, rating_trend.html, route_cost_scatter.html
      - bundle.zip (zip of all above)
    """
    out: Dict[str, bytes] = {
        "orders.csv": _df_bytes_csv(orders),
        "costs.csv": _df_bytes_csv(costs),
        "routes.csv": _df_bytes_csv(routes),
        "delivery.csv": _df_bytes_csv(delivery),
        "feedback.csv": _df_bytes_csv(feedback),
        "cost_breakdown.html": _chart_html_bytes(charts["cost_breakdown"], "Cost Breakdown"),
        "sla_bar.html": _chart_html_bytes(charts["sla_bar"], "Delivery Reliability"),
        "leadtime_hist.html": _chart_html_bytes(charts["leadtime_hist"], "Lead Time"),
        "rating_trend.html": _chart_html_bytes(charts["rating_trend"], "Ratings"),
        "route_cost_scatter.html": _chart_html_bytes(charts["route_cost_scatter"], "Route vs Cost"),
    }

    # Bundle everything into a single ZIP
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, blob in out.items():
            z.writestr(name, blob)
    out["bundle.zip"] = mem.getvalue()

    return out


if __name__ == "__main__":
    # No runtime side effects; import and call build_download_artifacts() from your app.
    print("report.py loaded. Use build_download_artifacts(...) from your Streamlit app.")
