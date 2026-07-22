"""Regenerate test fixtures. Run: python tests/fixtures/make_fixtures.py"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

OUT = Path(__file__).parent


def make_sample_pdf():
    c = canvas.Canvas(str(OUT / "sample.pdf"), pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 760, "Annual Maintenance Report")
    c.setFont("Helvetica", 11)
    c.drawString(72, 730, "The service contract number is SC-4471.")
    c.drawString(72, 710, "Coverage runs from January to December 2024.")
    c.showPage()
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 760, "Escalation Procedure")
    c.setFont("Helvetica", 11)
    c.drawString(72, 730, "Critical faults escalate to the duty engineer.")
    c.showPage()
    c.save()


if __name__ == "__main__":
    make_sample_pdf()
    print("wrote sample.pdf")
