"""외곽선(칸 픽셀) → OTF(CFF) + WOFF2."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.ttLib import TTFont

from .vector import Outline, Point, affine, bounds

UPM = 1000
ASCENT, DESCENT = 880, -120
HANGUL_MID = 380  # 한글 몸통 세로 중심 (폰트 단위)
LATIN_SB = 50
SPACE_W = 300


@dataclass(frozen=True)
class GlyphSrc:
    char: str
    outline: Outline  # 칸 픽셀 좌표
    kind: str  # "hangul" | "latin"


@dataclass(frozen=True)
class Metrics:
    cell_px: float
    baseline_px: float  # 영문 칸 기준선 (칸 픽셀)
    scale: float  # 폰트 단위 / 픽셀
    hangul_adv: int


def place_hangul(outline: Outline, m: Metrics) -> Outline:
    """가로는 가운데 정렬, 세로는 쓴 위치의 절반만 살린다(들쭉날쭉 완화)."""
    b = bounds(outline)
    by = (b.y0 + b.y1) / 2
    keep = (by - m.cell_px / 2) * 0.5
    ty = HANGUL_MID + (by - keep) * m.scale
    tx = m.hangul_adv / 2 - (b.x0 + b.x1) / 2 * m.scale
    return affine(outline, m.scale, -m.scale, tx, ty)


def place_latin(outline: Outline, m: Metrics) -> tuple[Outline, int]:
    b = bounds(outline)
    adv = round(b.w * m.scale + 2 * LATIN_SB)
    return affine(outline, m.scale, -m.scale, LATIN_SB - b.x0 * m.scale, m.baseline_px * m.scale), adv


def _r(p: Point) -> tuple[int, int]:
    return round(p[0]), round(p[1])


def charstring(outline: Outline, width: int):
    pen = T2CharStringPen(width, None)
    for c in outline:
        pen.moveTo(_r(c.start))
        for s in c.segs:
            if s[0] == "L":
                pen.lineTo(_r(s[1]))
            else:
                pen.curveTo(_r(s[1]), _r(s[2]), _r(s[3]))
        pen.closePath()
    return pen.getCharString()


def ps_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text)[:20] or "Handwriting"


def build(glyphs: list[GlyphSrc], m: Metrics, family: str, uid: str, out_otf: Path, names: dict[str, str] | None = None) -> Path:
    """names: 사이트가 job.json 에 넣어 준 copyright / license_text / license_url (없으면 기본 문구)"""
    order = [".notdef", "space"]
    cmap = {0x20: "space", 0x3000: "space"}
    strings = {".notdef": charstring([], 500), "space": charstring([], SPACE_W)}
    metrics = {".notdef": (500, 0), "space": (SPACE_W, 0)}
    for g in glyphs:
        if not g.outline or ord(g.char) in cmap:
            continue
        name = f"uni{ord(g.char):04X}"
        if g.kind == "latin":
            placed, adv = place_latin(g.outline, m)
        else:
            placed, adv = place_hangul(g.outline, m), m.hangul_adv
        order.append(name)
        cmap[ord(g.char)] = name
        strings[name] = charstring(placed, adv)
        metrics[name] = (adv, round(bounds(placed).x0))

    psn = f"{ps_name(family)}-{uid[:8]}"
    fb = FontBuilder(UPM, isTTF=False)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(cmap)
    fb.setupCFF(psn, {"FullName": psn}, strings, {})  # CFF 이름은 latin-1 만 허용 — 한글 이름은 name 테이블에
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ASCENT, descent=DESCENT)
    names = names or {}
    fb.setupNameTable(
        {"familyName": {"en": family, "ko": family}, "styleName": "Regular", "uniqueFontIdentifier": f"font.sw4u.kr:{uid}",
         "fullName": {"en": family, "ko": family}, "psName": psn, "version": "Version 1.000",
         "manufacturer": "font.sw4u.kr", "description": "손글씨로 만든 폰트 (font.sw4u.kr)",
         "copyright": names.get("copyright") or "© font.sw4u.kr. 만든 사람과 공동 저작권.",
         "licenseDescription": names.get("license_text") or "저작권은 만든 사람과 font.sw4u.kr 이 공동으로 보유합니다.",
         "licenseInfoURL": names.get("license_url") or "https://font.sw4u.kr/terms"},
        mac=False,
    )
    fb.setupOS2(sTypoAscender=ASCENT, sTypoDescender=DESCENT, sTypoLineGap=200,
                usWinAscent=1000, usWinDescent=280, achVendID="SW4U", fsType=0)
    fb.setupPost()
    font = fb.font
    font["OS/2"].recalcUnicodeRanges(font)
    font["OS/2"].recalcCodePageRanges(font)
    out_otf.parent.mkdir(parents=True, exist_ok=True)
    font.save(str(out_otf))
    return out_otf


def to_woff2(otf: Path, out: Path) -> Path:
    f = TTFont(str(otf))
    f.flavor = "woff2"
    f.save(str(out))
    return out
