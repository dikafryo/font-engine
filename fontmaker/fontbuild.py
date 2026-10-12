"""외곽선(칸 픽셀) → OTF(CFF) + WOFF2."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.ttLib import TTFont

from .vector import Outline, Point, affine, bounds, scale_about

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


CAPS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")  # 대문자선까지
ASCENDERS = set("bdfhklt")  # 대문자선까지 (t 는 조금 낮게)
XHEIGHT = set("acemnorsuvwxzi")  # x 높이까지 (i 는 몸통 기준, 점은 따라 움직인다)
DESCENDERS = set("gjpqy")  # 전체 높이는 어센더만큼, 몸통 꼭대기를 x 높이에 두어 꼬리가 기준선 아래로 내려간다
SHORTER = {"t": 0.85}  # 어센더 무리 안에서 낮게 쓰는 글자
EVEN_RANGE = (0.6, 1.6)  # 한 글자를 키우거나 줄이는 한계 (측정이 어긋난 글자가 폭주하지 않게)


def _main_top(outline: Outline) -> float:
    """몸통(경계 상자가 가장 큰 윤곽)의 꼭대기 y. i·j 의 점, 떨어진 조각은 뺀다."""
    best, top = -1.0, 0.0
    for c in outline:
        b = bounds([c])
        if b and b.w * b.h > best:
            best, top = b.w * b.h, b.y0
    return top


def even_latin_heights(latin: dict[str, Outline], baseline_px: float) -> dict[str, Outline]:
    """영문·숫자 높이를 고르게: 같은 무리(대문자·숫자·어센더 / x 높이·디센더)는 그 무리의 중앙값 높이로 기준선을 축으로
    확대·축소한다. 글쓴이의 비율(대문자 대 x 높이)은 중앙값이 그대로 지키고, 글자마다 들쭉날쭉한 것만 고른다.
    기호·문장부호는 손대지 않는다. 기준선에서 뜬 글자는 바닥을 기준선에 붙인다(디센더는 꼬리가 바닥이라 제외)."""
    heights: dict[str, float] = {}  # 글자 → 기준선 위 높이 (픽셀). 무리 중앙값(목표)을 재는 데만 쓴다
    for ch, o in latin.items():
        if ch in CAPS or ch in ASCENDERS:
            heights[ch] = baseline_px - bounds(o).y0
        elif ch in XHEIGHT:
            heights[ch] = baseline_px - _main_top(o)
    cap = _median([h for c, h in heights.items() if c in CAPS or c in ASCENDERS])
    xh = _median([h for c, h in heights.items() if c in XHEIGHT])
    out = {}
    for ch, o in latin.items():
        b = bounds(o)
        if ch in DESCENDERS and cap and xh:
            # 많은 분이 p·q·g·y 를 기준선 아래로 내리지 않고 칸 위쪽에 통째로 쓴다. 전체 높이를 어센더 높이로 맞춘 뒤
            # 몸통 꼭대기를 x 높이에 두면 꼬리가 기준선 아래로 내려간다
            k = _clamp(cap / max(b.h, 1))
            o = scale_about(o, k, k, b.x0, b.y0)
            out[ch] = affine(o, 1, 1, 0, baseline_px - xh - _main_top(o))
            continue
        target = (cap * SHORTER.get(ch, 1.0)) if (ch in CAPS or ch in ASCENDERS) else (xh if ch in XHEIGHT else None)
        if not target or not b:
            out[ch] = o  # 기호·문장부호
            continue
        o = affine(o, 1, 1, 0, baseline_px - b.y1)  # 바닥을 기준선에
        h = baseline_px - (bounds(o).y0 if ch in CAPS or ch in ASCENDERS else _main_top(o))
        k = _clamp(target / max(h, 1))
        out[ch] = scale_about(o, k, k, bounds(o).x0, baseline_px)
    return out


def _median(xs: list[float]) -> float:
    xs = sorted(x for x in xs if x > 0)
    return xs[len(xs) // 2] if xs else 0.0


def _clamp(k: float) -> float:
    return min(max(k, EVEN_RANGE[0]), EVEN_RANGE[1])


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
        {"familyName": {"en": family, "ko": family}, "styleName": "Regular", "uniqueFontIdentifier": f"neis.me/v1/font:{uid}",
         "fullName": {"en": family, "ko": family}, "psName": psn, "version": "Version 1.000",
         "manufacturer": "neis.me", "description": "손글씨로 만든 폰트 (neis.me/v1/font)",
         "copyright": names.get("copyright") or "© neis.me. 만든 사람과 공동 저작권.",
         "licenseDescription": names.get("license_text") or "저작권은 만든 사람과 neis.me 가 공동으로 보유합니다.",
         "licenseInfoURL": names.get("license_url") or "https://neis.me/v1/font/terms"},
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
