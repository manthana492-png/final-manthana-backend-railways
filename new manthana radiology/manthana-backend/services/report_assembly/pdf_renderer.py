"""
Manthana — PDF Report Renderer
WeasyPrint + Google Noto fonts for full Unicode/Indic script support.
"""

import os
import uuid
import logging
from typing import Optional

logger = logging.getLogger("manthana.pdf_renderer")

# Check WeasyPrint availability
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    logger.warning("WeasyPrint not available — PDF generation disabled")

PDF_OUTPUT_DIR = os.getenv("PDF_OUTPUT_DIR", "/tmp/manthana_reports")
os.makedirs(PDF_OUTPUT_DIR, exist_ok=True)


# Google Noto font CSS — covers all 10 Indian scripts
NOTO_FONT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;600&family=Noto+Sans+Tamil:wght@400;600&family=Noto+Sans+Telugu:wght@400;600&family=Noto+Sans+Kannada:wght@400;600&family=Noto+Sans+Malayalam:wght@400;600&family=Noto+Sans+Bengali:wght@400;600&family=Noto+Sans+Gujarati:wght@400;600&family=Noto+Sans+Gurmukhi:wght@400;600&family=Noto+Sans:wght@400;600&display=swap');
"""

# Language → CSS font stack
FONT_STACKS = {
    "hi": "'Noto Sans Devanagari', 'Noto Sans', Arial, sans-serif",
    "mr": "'Noto Sans Devanagari', 'Noto Sans', Arial, sans-serif",
    "ta": "'Noto Sans Tamil', 'Noto Sans', Arial, sans-serif",
    "te": "'Noto Sans Telugu', 'Noto Sans', Arial, sans-serif",
    "kn": "'Noto Sans Kannada', 'Noto Sans', Arial, sans-serif",
    "ml": "'Noto Sans Malayalam', 'Noto Sans', Arial, sans-serif",
    "bn": "'Noto Sans Bengali', 'Noto Sans', Arial, sans-serif",
    "gu": "'Noto Sans Gujarati', 'Noto Sans', Arial, sans-serif",
    "pa": "'Noto Sans Gurmukhi', 'Noto Sans', Arial, sans-serif",
    "en": "'Noto Sans', Arial, sans-serif",
}

# LTR only for Indian languages
TEXT_DIRECTION = "ltr"


def _build_report_html(
    narrative: str,
    impression: str,
    modality: str,
    patient_id: str,
    language: str,
    findings_header: str,
    impression_header: str,
    disclaimer: str,
    institution: str = "Manthana Radiology Suite",
) -> str:
    """Build the HTML template for PDF rendering."""
    font_stack = FONT_STACKS.get(language, FONT_STACKS["en"])

    # Format narrative paragraphs
    narrative_html = "\n".join(
        f"<p>{line}</p>" for line in narrative.strip().split("\n") if line.strip()
    )
    impression_html = "\n".join(
        f"<p>{line}</p>" for line in impression.strip().split("\n") if line.strip()
    )

    return f"""<!DOCTYPE html>
<html lang="{language}" dir="ltr">
<head>
  <meta charset="UTF-8">
  <style>
    {NOTO_FONT_CSS}

    * {{ box-sizing: border-box; }}

    body {{
      font-family: {font_stack};
      font-size: 11pt;
      color: #1a1a1a;
      background: #fff;
      margin: 0;
      padding: 0;
      line-height: 1.6;
    }}

    .page {{
      width: 210mm;
      min-height: 297mm;
      padding: 18mm 20mm;
      margin: 0 auto;
    }}

    /* Header */
    .report-header {{
      border-bottom: 2px solid #1a6b5a;
      padding-bottom: 12px;
      margin-bottom: 20px;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
    }}

    .report-title {{
      font-size: 20pt;
      font-weight: 600;
      color: #1a6b5a;
      margin: 0;
      font-family: {font_stack};
    }}

    .report-subtitle {{
      font-size: 9pt;
      color: #666;
      margin: 3px 0 0 0;
    }}

    .report-logo {{
      text-align: right;
      font-size: 9pt;
      color: #888;
    }}

    /* Meta table */
    .meta-table {{
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 20px;
      font-size: 9.5pt;
    }}

    .meta-table td {{
      padding: 4px 8px;
      border: 1px solid #e0e0e0;
    }}

    .meta-table td:first-child {{
      background: #f5f5f5;
      font-weight: 600;
      white-space: nowrap;
      width: 30%;
    }}

    /* Section headers */
    .section-header {{
      font-size: 13pt;
      font-weight: 600;
      color: #1a6b5a;
      margin-top: 20px;
      margin-bottom: 8px;
      padding-bottom: 4px;
      border-bottom: 1px solid #c8e6c9;
      font-family: {font_stack};
    }}

    /* Content */
    .section-content {{
      font-size: 10.5pt;
      color: #2c2c2c;
      text-align: justify;
      font-family: {font_stack};
    }}

    .section-content p {{
      margin: 0 0 8px 0;
    }}

    /* Impression box */
    .impression-box {{
      background: #f0faf5;
      border-left: 4px solid #1a6b5a;
      padding: 12px 16px;
      margin-top: 8px;
      border-radius: 0 4px 4px 0;
    }}

    /* Disclaimer */
    .disclaimer {{
      margin-top: 24px;
      padding: 10px 14px;
      background: #fff8e1;
      border: 1px solid #ffe082;
      border-radius: 4px;
      font-size: 8.5pt;
      color: #5d4037;
      font-family: {font_stack};
    }}

    /* Footer */
    .report-footer {{
      position: fixed;
      bottom: 12mm;
      left: 20mm;
      right: 20mm;
      border-top: 1px solid #e0e0e0;
      padding-top: 6px;
      font-size: 8pt;
      color: #aaa;
      display: flex;
      justify-content: space-between;
    }}
  </style>
</head>
<body>
  <div class="page">

    <!-- Header -->
    <div class="report-header">
      <div>
        <h1 class="report-title">{institution}</h1>
        <p class="report-subtitle">AI Radiology Second-Opinion Report</p>
      </div>
      <div class="report-logo">
        🏥 AI-Assisted<br>Radiology
      </div>
    </div>

    <!-- Patient Meta -->
    <table class="meta-table">
      <tr><td>Patient ID</td><td>{patient_id}</td></tr>
      <tr><td>Modality</td><td>{modality.upper()}</td></tr>
      <tr><td>Report Language</td><td>{language.upper()}</td></tr>
      <tr><td>Generated</td><td>AI-Assisted (Second Opinion)</td></tr>
    </table>

    <!-- Findings Section -->
    <div class="section-header">{findings_header}</div>
    <div class="section-content">
      {narrative_html}
    </div>

    <!-- Impression Section -->
    <div class="section-header">{impression_header}</div>
    <div class="impression-box">
      <div class="section-content">
        {impression_html}
      </div>
    </div>

    <!-- Disclaimer -->
    <div class="disclaimer">
      ⚠ {disclaimer}
    </div>

  </div>

  <!-- Footer -->
  <div class="report-footer">
    <span>Manthana Radiology Suite — Powered by AI</span>
    <span>Confidential — For clinical review only</span>
  </div>

</body>
</html>"""


async def render_report_pdf(
    narrative: str,
    impression: str,
    modality: str,
    patient_id: str,
    language: str,
    findings_header: str,
    impression_header: str,
    disclaimer: str,
    institution: str = "Manthana Radiology Suite",
) -> Optional[str]:
    """
    Render a radiology report to PDF with correct Unicode font.

    Returns:
        Absolute path to the generated PDF file, or None if WeasyPrint unavailable.
    """
    if not WEASYPRINT_AVAILABLE:
        logger.warning("WeasyPrint not installed — skipping PDF generation")
        return None

    html_content = _build_report_html(
        narrative=narrative,
        impression=impression,
        modality=modality,
        patient_id=patient_id,
        language=language,
        findings_header=findings_header,
        impression_header=impression_header,
        disclaimer=disclaimer,
        institution=institution,
    )

    output_filename = f"report_{patient_id}_{language}_{uuid.uuid4().hex[:8]}.pdf"
    output_path = os.path.join(PDF_OUTPUT_DIR, output_filename)

    try:
        doc = HTML(string=html_content, base_url="https://fonts.googleapis.com")
        doc.write_pdf(output_path)
        logger.info(f"PDF generated: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return None
