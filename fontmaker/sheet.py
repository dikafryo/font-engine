"""손글씨 양식 PDF 생성. 칸 좌표는 layout.py 가 정한다.

사용: python -m fontmaker.sheet [출력경로]   (기본 out/sheet_v2.pdf)
"""
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.colors import Color, black
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

from . import layout as L

ROOT = Path(__file__).resolve().parent.parent
FONT = "NotoSansKR"
pdfmetrics.registerFont(TTFont(FONT, str(ROOT / "assets/NotoSansKR-Regular.ttf")))

# 스캔 후 이진화에서 사라지도록 연한 파랑, 안내 글자는 연한 회색
GUIDE = Color(0.62, 0.78, 0.96)
GUIDE_SOFT = Color(0.80, 0.89, 0.98)
LABEL = Color(0.55, 0.55, 0.55)

TIPS = (
    "· 평소 쓰는 검정 펜 하나로 끝까지 (0.5~0.7mm 권장). 회색 글자를 보고 바로 아래 파란 칸 안에 평소 모양 그대로 쓰세요.",
    "· 칸 가장자리에 닿지 않게 자연스러운 크기로. 틀리면 수정테이프로 지우고 다시. 영문은 파란 기준선 위에 쓰세요.",
    "· 반드시 100%(실제 크기)로 인쇄. 네 모서리 검은 네모와 아래 가운데 쪽 코드는 가리거나 접지 마세요.",
    "· 1~3쪽이 필수, 4쪽(문장)은 선택입니다. 스캔은 300dpi 이상 컬러/그레이, 사진이면 평평하게·그늘 없이·정면에서.",
)


def y(top_mm: float) -> float:
    """좌상단 기준 mm → reportlab 좌하단 기준 pt."""
    return (L.PAGE_H - top_mm) * mm


def draw_frame(c: Canvas, page: L.Page) -> None:
    c.setFillColor(black)
    for cx, cy in L.MARK_CENTERS:
        c.rect((cx - L.MARK / 2) * mm, y(cy + L.MARK / 2), L.MARK * mm, L.MARK * mm, stroke=0, fill=1)
    for bit, (x, top) in zip(L.code_bits(page.number), L.code_squares()):
        if bit:
            c.rect(x * mm, y(top + L.CODE_SQ), L.CODE_SQ * mm, L.CODE_SQ * mm, stroke=0, fill=1)
    c.setFont(FONT, 12)
    c.drawString(20 * mm, y(17), f"손글씨 폰트 양식 — {page.title}")
    c.setFont(FONT, 8)
    c.setFillColor(LABEL)
    c.drawRightString(190 * mm, y(17), f"font.sw4u.kr · {L.VERSION} · {page.number}/{len(L.pages())} 쪽")


def draw_box(c: Canvas, cell: L.Cell) -> None:
    x, top, s = cell.x, cell.y, cell.size
    c.setStrokeColor(GUIDE)
    c.setLineWidth(0.5)
    c.rect(x * mm, y(top + s), s * mm, s * mm, stroke=1, fill=0)
    c.setStrokeColor(GUIDE_SOFT)
    c.setLineWidth(0.3)
    c.setDash(1.2, 1.8)
    c.line((x + s / 2) * mm, y(top), (x + s / 2) * mm, y(top + s))
    if cell.kind == "hangul":
        c.line(x * mm, y(top + s / 2), (x + s) * mm, y(top + s / 2))
    else:
        c.line(x * mm, y(top + s * L.LATIN_CAP), (x + s) * mm, y(top + s * L.LATIN_CAP))
    c.setDash()
    if cell.kind == "latin":
        c.setStrokeColor(GUIDE)
        c.setLineWidth(0.5)
        c.line(x * mm, y(top + s * L.LATIN_BASELINE), (x + s) * mm, y(top + s * L.LATIN_BASELINE))


def draw_cells(c: Canvas, page: L.Page) -> None:
    for cell in page.cells:
        c.setFillColor(LABEL)
        c.setFont(FONT, 9)
        c.drawString((cell.x + 0.5) * mm, y(cell.y - 0.9), cell.char)
        c.setFont(FONT, 5)
        c.drawRightString((cell.x + cell.size - 0.5) * mm, y(cell.y - 0.9), str(cell.index))
        draw_box(c, cell)


def draw_tips(c: Canvas) -> None:
    c.setFont(FONT, 7.8)
    c.setFillColor(black)
    for i, tip in enumerate(TIPS):
        c.drawString(20 * mm, y(25 + 4.6 * i), tip)


def draw_ruled(c: Canvas, top: float, h: float, width: float) -> None:
    c.setStrokeColor(GUIDE)
    c.setLineWidth(0.4)
    c.line(20 * mm, y(top + h), (20 + width) * mm, y(top + h))
    c.setStrokeColor(GUIDE_SOFT)
    c.setLineWidth(0.3)
    c.setDash(1.2, 1.8)
    c.line(20 * mm, y(top + h / 2), (20 + width) * mm, y(top + h / 2))
    c.setDash()


def draw_sentences(c: Canvas) -> None:
    top = 25.0
    c.setFont(FONT, 9)
    c.setFillColor(black)
    c.drawString(20 * mm, y(top), "회색 문장을 아래 줄에 평소처럼 써 주세요. 폰트에는 쓰이지 않고 글씨 분석에만 씁니다.")
    top += 5
    for s in L.SENTENCES:
        c.setFillColor(LABEL)
        c.setFont(FONT, 8.5)
        c.drawString(20 * mm, y(top + 3), s)
        top += 5
        for h in (12.0, 8.0):
            draw_ruled(c, top, h, 170)
            top += h + 2
        top += 2
    c.setFillColor(black)
    c.setFont(FONT, 9)
    c.drawString(20 * mm, y(top + 2), "자유롭게 몇 줄")
    top += 5
    while top + 9 < 276:
        draw_ruled(c, top, 9.0, 170)
        top += 9


def render(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    c = Canvas(str(out), pagesize=A4)
    c.setTitle("손글씨 폰트 양식")
    for page in L.pages():
        draw_frame(c, page)
        if page.number == 1:
            draw_tips(c)
        if page.cells:
            draw_cells(c, page)
        else:
            draw_sentences(c)
        c.showPage()
    c.save()


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "out/sheet_v2.pdf"
    render(target)
    counts = [len(p.cells) for p in L.pages()]
    print(f"{target} — 칸 {counts} (한글 {len(L.hangul_chars())}자)")
