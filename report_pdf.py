"""
Generate a simple 'Issue Remediation Plan' PDF (standalone, no external deps).
"""
from __future__ import annotations

from typing import List, Dict
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from io import BytesIO
import pandas as pd

def _para(text: str, style):
    return Paragraph(text.replace("\n", "<br/>"), style)

def build_issue_plan_pdf(top_issues: List[Dict[str, str]], feedback: pd.DataFrame, delivery: pd.DataFrame) -> bytes:
    """
    Creates a compact PDF summarizing Top-N issues with basic stats and actions.
    Returns bytes suitable for Streamlit download_button.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="Issue Remediation Plan")
    styles = getSampleStyleSheet()
    story = []

    story.append(_para("<b>Issue Remediation Plan</b>", styles["Title"]))
    story.append(_para(datetime.now().strftime("%Y-%m-%d %H:%M"), styles["Normal"]))
    story.append(Spacer(1, 12))

    # Overview table
    data = [["#", "Issue", "Avg Rating", "Delay Rate", "Suggested Action"]]
    for idx, it in enumerate(top_issues, start=1):
        data.append([idx, it["issue"], it["avg_rating"], it["delay_rate"], it["actions"]])

    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#222222")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.gray),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey]),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 16))

    # Notes
    story.append(_para("<b>Notes</b>", styles["Heading3"]))
    story.append(_para(
        "This plan prioritizes lowest-rated issues. Delay rates are computed as the share of 'Delayed' deliveries "
        "among orders with this issue. Actions are suggested starting points and should be tailored to specific lanes/carriers.",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 12))

    doc.build(story)
    return buf.getvalue()

if __name__ == "__main__":
    # No execution here;
    pass
