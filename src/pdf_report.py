"""
src/pdf_report.py

Generates a single-page PDF report for one MRI prediction result.
Used by app.py — one report per uploaded MRI.
"""

import io
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.enums import TA_CENTER


def build_pdf_report(filename: str, prediction: str, confidence: float,
                      probabilities: dict, gradcam_img_path: str = None,
                      original_img_path: str = None) -> bytes:
    """
    Builds a PDF report in memory and returns it as bytes, ready to be
    handed to st.download_button.

    Args:
        filename: original uploaded MRI filename (for display only)
        prediction: predicted class label, e.g. "Glioma"
        confidence: float 0-100 (percentage)
        probabilities: dict like {"Glioma": 96.42, "Meningioma": 2.13, ...}
        gradcam_img_path: path to the Grad-CAM overlay image (optional)
        original_img_path: path to the original MRI image (optional)

    Returns:
        bytes of the generated PDF
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=25 * mm, rightMargin=25 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"],
        fontSize=22, textColor=colors.HexColor("#1f6feb"), alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle", parent=styles["Normal"],
        fontSize=11, textColor=colors.HexColor("#555555"), alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        "SectionStyle", parent=styles["Heading2"],
        fontSize=13, textColor=colors.HexColor("#1f6feb"), spaceBefore=14, spaceAfter=6,
    )
    normal = styles["Normal"]

    story = []

    # Header
    story.append(Paragraph("NeuroVision AI", title_style))
    story.append(Paragraph("Brain MRI Analysis Report", subtitle_style))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
        subtitle_style,
    ))
    story.append(Spacer(1, 8 * mm))

    # File / prediction summary table
    summary_data = [
        ["Filename", filename],
        ["Prediction", prediction],
        ["Confidence", f"{confidence:.2f}%"],
    ]
    summary_table = Table(summary_data, colWidths=[45 * mm, 100 * mm])
    summary_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10.5),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#333333")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 6 * mm))

    # Probabilities section
    story.append(Paragraph("Class Probabilities", section_style))
    prob_rows = [["Class", "Probability"]]
    for cls, val in sorted(probabilities.items(), key=lambda x: -x[1]):
        prob_rows.append([cls, f"{val:.2f}%"])

    prob_table = Table(prob_rows, colWidths=[70 * mm, 40 * mm])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f6feb")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fb")]),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 8 * mm))

    # Images: original + Grad-CAM side by side
    if original_img_path or gradcam_img_path:
        story.append(Paragraph("Explainable AI — Grad-CAM", section_style))
        img_row = []
        img_labels = []
        if original_img_path:
            img_row.append(RLImage(original_img_path, width=65 * mm, height=65 * mm))
            img_labels.append(Paragraph("Original MRI", normal))
        if gradcam_img_path:
            img_row.append(RLImage(gradcam_img_path, width=65 * mm, height=65 * mm))
            img_labels.append(Paragraph("Grad-CAM Overlay", normal))

        img_table = Table([img_row, img_labels], colWidths=[70 * mm] * len(img_row))
        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 1), (-1, 1), 4),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 8 * mm))

    # Footer
    story.append(Spacer(1, 10 * mm))
    footer_style = ParagraphStyle(
        "Footer", parent=styles["Normal"], fontSize=9,
        textColor=colors.HexColor("#888888"), alignment=TA_CENTER,
    )
    story.append(Paragraph(
        "Generated by NeuroVision AI — for research and educational purposes only. "
        "Not a substitute for professional medical diagnosis.",
        footer_style,
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()