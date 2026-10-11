"""Privacy-conscious student PDF report generation."""

from io import BytesIO

import pandas as pd

from src.data.college import make_study_plan


def _build_college_pdf(record: dict, prediction: dict, subjects: pd.DataFrame,
                       weak: list[str], strong: list[str], remarks: str = "") -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    stream = BytesIO()
    pdf = canvas.Canvas(stream, pagesize=letter)
    _, height = letter
    y = height - 50
    pdf.setTitle("Trekverse Student Academic Report")
    pdf.setFont("Helvetica-Bold", 17)
    pdf.drawString(48, y, "TREKVERSE AI ACADEMY")
    y -= 25
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(48, y, "Student Academic Performance Report")
    y -= 28
    pdf.setFont("Helvetica", 10)
    lines = [
        "Student Information", f"Name: {record.get('Name', '')}", f"Student ID: {record.get('Student_ID', '')}",
        f"Department: {record.get('Department', '')}", f"Semester: {record.get('Semester', '')}",
        "", "Academic Performance", f"Current CGPA: {record.get('Current_CGPA', 'Not provided')}",
        f"Previous CGPA: {record.get('Previous_CGPA', 'Not provided')}",
        f"Attendance: {record.get('Attendance', '')}%", f"Average marks: {subjects['Final_Marks'].mean():.1f}" if not subjects.empty else "Average marks: unavailable",
        "", "AI analysis",
        (f"Model-estimated score: {prediction['predicted_score']:.1f}/100"
         if prediction.get("predicted_score") is not None else "Model-estimated score: not generated"),
        f"Risk level: {prediction.get('risk_level', 'Unavailable')}",
        f"Early-warning risk score: {prediction.get('risk_score', 'Unavailable')} / 100",
        f"Assignment completion: {record.get('Assignment_Submission_Rate', 'Not provided')}%",
        f"Previous failures / backlogs: {record.get('Backlogs', 'Not provided')}",
        "Strong subjects: " + (", ".join(strong) if strong else "None recorded"),
        "Subjects needing attention: " + (", ".join(weak) if weak else "None recorded"),
        "", "Study plan:",
    ]
    risk = record.get("Risk_Assessment", {})
    flagged = [item["factor"] for item in risk.get("factors", []) if item.get("flagged")]
    lines.append("Factors for review: " + (", ".join(flagged) if flagged else "No available factor crossed the review threshold"))
    for item in prediction.get("explanation", [])[:5]:
        value = item.get("shap", item.get("delta_if_typical", 0))
        lines.append(f"Model explanation {item.get('feature', '')}: {float(value):+.3f}")
    for item in lines:
        pdf.drawString(48, y, str(item)[:105])
        y -= 16
    for item in make_study_plan(weak, record):
        pdf.drawString(58, y, f"{item['day']}: {item['task']}"[:105])
        y -= 15
    y -= 4
    pdf.drawString(48, y, "Faculty remarks: " + (remarks or "No remarks provided.")[:95])
    y -= 20
    pdf.setFont("Helvetica-Oblique", 8)
    pdf.drawString(48, y, "Model output is an estimate for supportive decision-making, not a guaranteed outcome.")
    pdf.save()
    return stream.getvalue()

def _build_pdf_report(payload: dict, result: dict) -> bytes:
    """Create a compact, privacy-conscious prediction report in memory."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    stream = BytesIO()
    pdf = canvas.Canvas(stream, pagesize=letter)
    _, height = letter
    y = height - 54
    pdf.setTitle("Student Academic Performance Report")
    pdf.setFont("Helvetica-Bold", 17)
    pdf.drawString(48, y, "Student Academic Performance Report")
    y -= 32
    pdf.setFont("Helvetica", 11)
    lines = [
        "Model estimate (0-20): " + str(result["predicted_score"]),
        "Support group: " + str(result["predicted_support"]).replace("_", " "),
        "Engagement segment: " + str(result["engagement_cluster"]),
        "Explanation method: " + str(result.get("explanation_method", "local explanation")),
        "Study-time band: " + str(payload.get("studytime")),
        "Past failures: " + str(payload.get("failures")),
        "Absences: " + str(payload.get("absences")),
        "",
        "Recommendations:",
    ]
    for item in result.get("explanation", {}).get("top", [])[:5]:
        value = item.get("shap", item.get("delta_if_typical", 0))
        lines.append(f"Explanation: {item.get('feature', '')}: {float(value):+.3f}")
    for line in lines:
        pdf.drawString(48, y, line[:100])
        y -= 19
    for rec in result.get("recommendations", []):
        for line in (rec["gap"] + ": " + rec["action"]).splitlines():
            pdf.drawString(58, y, line[:95])
            y -= 16
            if y < 90:
                pdf.showPage()
                y = height - 54
    y -= 8
    pdf.setFont("Helvetica-Oblique", 9)
    disclaimer = result.get("disclaimer", "")
    for start in range(0, len(disclaimer), 105):
        pdf.drawString(48, y, disclaimer[start : start + 105])
        y -= 13
    pdf.save()
    return stream.getvalue()
