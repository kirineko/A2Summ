import re
from pathlib import Path

import pypdfium2 as pdfium
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)
from xml.sax.saxutils import escape


ROOT = Path("/home/linux/A2Summ")
SOURCE = ROOT / "report.md"
OUTPUT_DIR = ROOT / "output" / "pdf"
PDF_PATH = OUTPUT_DIR / "report.pdf"
PREVIEW_DIR = OUTPUT_DIR / "report_preview"


def register_fonts():
    font_path = Path("/mnt/c/Windows/Fonts/simhei.ttf")
    pdfmetrics.registerFont(TTFont("SimHei", str(font_path)))


def build_styles():
    styles = getSampleStyleSheet()
    base = ParagraphStyle(
        "BaseCN",
        parent=styles["BodyText"],
        fontName="SimHei",
        fontSize=10.5,
        leading=17,
        textColor=colors.HexColor("#202124"),
        spaceAfter=6,
    )
    return {
        "base": base,
        "title": ParagraphStyle(
            "TitleCN",
            parent=base,
            fontSize=20,
            leading=28,
            alignment=TA_CENTER,
            spaceAfter=18,
            textColor=colors.HexColor("#0f172a"),
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCN",
            parent=base,
            fontSize=11,
            leading=17,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#475569"),
            spaceAfter=24,
        ),
        "h1": ParagraphStyle(
            "H1CN",
            parent=base,
            fontSize=16,
            leading=24,
            spaceBefore=10,
            spaceAfter=10,
            textColor=colors.HexColor("#111827"),
        ),
        "h2": ParagraphStyle(
            "H2CN",
            parent=base,
            fontSize=13,
            leading=20,
            spaceBefore=8,
            spaceAfter=8,
            textColor=colors.HexColor("#1f2937"),
        ),
        "bullet": ParagraphStyle(
            "BulletCN",
            parent=base,
            leftIndent=12,
            firstLineIndent=0,
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "CodeCN",
            parent=base,
            fontName="Courier",
            fontSize=8.5,
            leading=12,
            backColor=colors.HexColor("#f8fafc"),
            borderColor=colors.HexColor("#cbd5e1"),
            borderWidth=0.5,
            borderPadding=6,
            borderRadius=2,
            spaceBefore=6,
            spaceAfter=10,
        ),
    }


def md_links_to_text(text: str) -> str:
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)


def inline_markup(text: str) -> str:
    text = md_links_to_text(text)
    text = escape(text)
    text = re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def add_page_number(canvas, doc):
    canvas.setFont("SimHei", 9)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawRightString(190 * mm, 10 * mm, f"{doc.page}")


def parse_markdown(lines, styles):
    story = []
    in_code = False
    code_lines = []
    bullet_buffer = []
    para_buffer = []

    def flush_paragraph():
        nonlocal para_buffer
        if not para_buffer:
            return
        text = " ".join(x.strip() for x in para_buffer).strip()
        if text:
            story.append(Paragraph(inline_markup(text), styles["base"]))
        para_buffer = []

    def flush_bullets():
        nonlocal bullet_buffer
        if not bullet_buffer:
            return
        items = [
            ListItem(Paragraph(inline_markup(item), styles["bullet"]))
            for item in bullet_buffer
        ]
        story.append(
            ListFlowable(
                items,
                bulletType="bullet",
                start="circle",
                leftIndent=10,
                bulletFontName="SimHei",
            )
        )
        story.append(Spacer(1, 4))
        bullet_buffer = []

    def flush_code():
        nonlocal code_lines
        if not code_lines:
            return
        story.append(Preformatted("\n".join(code_lines), styles["code"]))
        code_lines = []

    for raw in lines:
        line = raw.rstrip("\n")

        if line.strip().startswith("```"):
            flush_paragraph()
            flush_bullets()
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not line.strip():
            flush_paragraph()
            flush_bullets()
            story.append(Spacer(1, 4))
            continue

        if line.startswith("# "):
            flush_paragraph()
            flush_bullets()
            story.append(Paragraph(inline_markup(line[2:].strip()), styles["title"]))
            story.append(
                Paragraph("A2Summ 端到端视频摘要系统说明与实现报告", styles["subtitle"])
            )
            continue

        if line.startswith("## "):
            flush_paragraph()
            flush_bullets()
            story.append(Paragraph(inline_markup(line[3:].strip()), styles["h1"]))
            continue

        if line.startswith("### "):
            flush_paragraph()
            flush_bullets()
            story.append(Paragraph(inline_markup(line[4:].strip()), styles["h2"]))
            continue

        if line.strip() == "---":
            flush_paragraph()
            flush_bullets()
            story.append(Spacer(1, 8))
            continue

        if re.match(r"^\s*-\s+", line):
            flush_paragraph()
            bullet_buffer.append(re.sub(r"^\s*-\s+", "", line).strip())
            continue

        para_buffer.append(line)

    flush_paragraph()
    flush_bullets()
    flush_code()
    return story


def build_pdf():
    register_fonts()
    styles = build_styles()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = SOURCE.read_text(encoding="utf-8").splitlines()

    story = parse_markdown(lines, styles)
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title="A2Summ 端到端视频摘要处理技术报告",
        author="OpenAI Codex",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def render_preview(max_pages: int = 3):
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(PDF_PATH))
    page_count = min(len(pdf), max_pages)
    for idx in range(page_count):
        page = pdf[idx]
        bitmap = page.render(scale=2.0)
        image = bitmap.to_pil()
        image.save(PREVIEW_DIR / f"page_{idx + 1:02d}.png")


def main():
    build_pdf()
    render_preview()
    print(PDF_PATH)


if __name__ == "__main__":
    main()
