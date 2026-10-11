"""나이스 디미체 — 디지털·미래 느낌의 둥근 고딕을 기하학적 자모 뼈대로 직접 그려 만든다.

손글씨 파이프라인을 쓰지 않는다. 자모마다 중심선(획)을 단위 상자 안에 정의하고, 음절 유형(V/H/M × 받침)에 따라
잉크 상자를 정한 뒤 굵고 둥근 선으로 래스터 → potrace 외곽선 → CFF 글리프. ㅇ·ㅁ·ㅎ 속은 채우고 ㅂ 아래 칸을 채운다.

사용: .venv/bin/python scripts/dimi.py <출력 폴더> [--uid 32hex] [--sample 견본.png]
출력: font.otf · font.woff2 · preview.png (워커 출력과 같은 이름)
"""
from __future__ import annotations

import argparse
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from fontTools.fontBuilder import FontBuilder
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fontmaker.fontbuild import SPACE_W, UPM, charstring, ps_name, to_woff2  # noqa: E402
from fontmaker.hangul import CHO, JONG, JUNG, all_syllables, decompose  # noqa: E402
from fontmaker.pipeline import preview  # noqa: E402
from fontmaker.vector import Outline, affine, bounds, trace  # noqa: E402

# ── 설계 좌표: 가로 1000 em, 세로는 위에서 아래로(yd). 글꼴 y = 800 - yd (기준선 yd=800) ──
W = 110         # 한글 획 굵기
LW = 100        # 영문·숫자 획 굵기
S = 2           # 래스터 배율 (px / 단위)
GAP = 50        # 쌍자음·겹받침 사이 잉크 간격
T = 170         # 모음 가지 길이 (중심선)
G2 = 190        # ㅐ·ㅔ 두 세로획 간격
HL, HR = 70, 930  # 한글 잉크 좌우
ASCENT, DESCENT = 880, -200

Stroke = tuple  # ("L", [(u,v)…]) 둥근 선 | ("A", cu,cv,ru,rv,deg0,deg1) 호 | ("D", u0,v0,u1,v1) 채운 원 | ("R", u0,v0,u1,v1) 채운 둥근 네모 | ("P", u,v) 점


def L(*pts) -> Stroke:
    return ("L", list(pts))


# ── 자음 (단위 상자, u 오른쪽·v 아래) ──
STEM_L, STEM_R = L((0, 0), (0, 1)), L((1, 0), (1, 1))
CONS: dict[str, list[Stroke]] = {
    "ㄱ": [L((0, 0), (1, 0), (1, 1))],
    "ㄴ": [L((0, 0), (0, 1), (1, 1))],
    "ㄷ": [L((1, 0), (0, 0), (0, 1), (1, 1))],
    "ㄹ": [L((0, 0), (1, 0), (1, 0.5), (0, 0.5), (0, 1), (1, 1))],
    "ㅁ": [("R", 0.13, 0.13, 0.87, 0.87)],  # 채운 네모는 상자보다 작게 — 획과 무게를 맞춘다
    "ㅂ": [STEM_L, STEM_R, ("R", 0, 0.5, 1, 1)],
    "ㅅ": [L((0.5, 0), (0, 1)), L((0.5, 0.1), (1, 1))],
    "ㅇ": [("D", 0.1, 0.1, 0.9, 0.9)],
    "ㅈ": [L((0, 0), (1, 0)), L((0.5, 0), (0.03, 1)), L((0.5, 0.14), (0.97, 1))],
    "ㅊ": [L((0.3, 0), (0.7, 0)), L((0, 0.3), (1, 0.3)), L((0.5, 0.3), (0.03, 1)), L((0.5, 0.44), (0.97, 1))],
    "ㅋ": [L((0, 0), (1, 0), (1, 1)), L((0, 0.5), (1, 0.5))],
    "ㅌ": [L((1, 0), (0, 0), (0, 1), (1, 1)), L((0, 0.5), (1, 0.5))],
    "ㅍ": [L((0, 0), (1, 0)), L((0, 1), (1, 1)), L((0.27, 0), (0.27, 1)), L((0.73, 0), (0.73, 1))],
    "ㅎ": [L((0.3, 0), (0.7, 0)), L((0, 0.28), (1, 0.28)), ("D", 0.1, 0.44, 0.9, 1)],
}
PAIRS = {"ㄲ": "ㄱㄱ", "ㄸ": "ㄷㄷ", "ㅃ": "ㅂㅂ", "ㅆ": "ㅅㅅ", "ㅉ": "ㅈㅈ", "ㄳ": "ㄱㅅ", "ㄵ": "ㄴㅈ", "ㄶ": "ㄴㅎ",
         "ㄺ": "ㄹㄱ", "ㄻ": "ㄹㅁ", "ㄼ": "ㄹㅂ", "ㄽ": "ㄹㅅ", "ㄾ": "ㄹㅌ", "ㄿ": "ㄹㅍ", "ㅀ": "ㄹㅎ", "ㅄ": "ㅂㅅ"}
THIN = {"ㄹ": 0.27, "ㅌ": 0.27, "ㅍ": 0.27, "ㅂ": 0.28, "ㅊ": 0.28, "ㅎ": 0.28, "ㅋ": 0.32, "ㄷ": 0.32, "ㅈ": 0.32}  # 획이 겹치지 않게 상자 크기에 비례해 가늘게
ASPECT = {"ㅇ": 1.0, "ㅁ": 1.15}  # 너비 ≤ 높이×비율 (나머지는 유형별 기본값)

# ── 모음 세로 부분: (자연 너비, 획) — 상자 u=0..1 이 자연 너비, v=0..1 이 높이 ──
_A = T / (T + G2)
VERT: dict[str, tuple[int, list[Stroke]]] = {
    "ㅣ": (0, [STEM_L]),
    "ㅏ": (T, [STEM_L, L((0, 0.5), (1, 0.5))]),
    "ㅑ": (T, [STEM_L, L((0, 0.36), (1, 0.36)), L((0, 0.64), (1, 0.64))]),
    "ㅓ": (T, [L((0, 0.5), (1, 0.5)), STEM_R]),
    "ㅕ": (T, [L((0, 0.36), (1, 0.36)), L((0, 0.64), (1, 0.64)), STEM_R]),
    "ㅐ": (G2, [STEM_L, L((0, 0.5), (1, 0.5)), STEM_R]),
    "ㅒ": (G2, [STEM_L, L((0, 0.36), (1, 0.36)), L((0, 0.64), (1, 0.64)), STEM_R]),
    "ㅔ": (T + G2, [L((0, 0.5), (_A, 0.5)), L((_A, 0), (_A, 1)), STEM_R]),
    "ㅖ": (T + G2, [L((0, 0.36), (_A, 0.36)), L((0, 0.64), (_A, 0.64)), L((_A, 0), (_A, 1)), STEM_R]),
}
M_PARTS = {"ㅘ": ("ㅗ", "ㅏ"), "ㅙ": ("ㅗ", "ㅐ"), "ㅚ": ("ㅗ", "ㅣ"), "ㅝ": ("ㅜ", "ㅓ"), "ㅞ": ("ㅜ", "ㅔ"), "ㅟ": ("ㅜ", "ㅣ"), "ㅢ": ("ㅡ", "ㅣ")}

# ── 음절 유형별 잉크 상자 (yd) ──
# 세로 모음: cho (x1 은 모음 왼쪽 - 간격), 모음 y, 받침
VLAY = {"V0": dict(cho_y=(40, 790), vow_y=(30, 800), jong=None, cho_gap=70),
        "V1": dict(cho_y=(40, 510), vow_y=(30, 520), jong=(90, 590, 910, 820), cho_gap=70)}
# 가로 모음: bar_y 중심선, tick 길이, 받침
HLAY = {"H0": dict(cho=(150, 30, 850, 420), up=(610, 100), down=(560, 120), flat=610, jong=None),
        "H1": dict(cho=(150, 30, 850, 310), up=(470, 90), down=(430, 90), flat=470, jong=(90, 600, 910, 820))}
MLAY = {"M0": dict(cho_y=(30, 400), vow_y=(30, 800), up=(580, 100), down=(540, 120), flat=580, jong=None, cho_gap=50),
        "M1": dict(cho_y=(30, 290), vow_y=(30, 520), up=(440, 80), down=(400, 90), flat=440, jong=(90, 600, 910, 820), cho_gap=50)}
MAX_ASPECT = {"V": 0.78, "H": 1.6, "M": 1.4, "J": 2.0}  # 자음 너비 ≤ 높이×비율


def arc_points(cu, cv, ru, rv, d0, d1, n=28):
    return [(cu + ru * math.cos(math.radians(d)), cv + rv * math.sin(math.radians(d))) for d in np.linspace(d0, d1, n)]


def render(strokes: list[Stroke], ink: tuple[float, float, float, float], w: float) -> Outline:
    """잉크 상자(x0,y0,x1,y1, yd)에 획을 그려 외곽선(설계 좌표)으로. 상자 u/v=0..1 은 중심선 범위(잉크에서 w/2 안쪽)."""
    x0, y0, x1, y1 = ink
    cx0, cy0, cw, ch = x0 + w / 2, y0 + w / 2, max(x1 - x0 - w, 0), max(y1 - y0 - w, 0)

    def m(p):
        return (cx0 + p[0] * cw, cy0 + p[1] * ch)

    items = []  # (종류, 점들/상자)
    for s in strokes:
        if s[0] == "L":
            items.append(("L", [m(p) for p in s[1]]))
        elif s[0] == "A":
            items.append(("L", [m(p) for p in arc_points(*s[1:])]))
        elif s[0] in ("D", "R"):
            items.append((s[0], (*m((s[1], s[2])), *m((s[3], s[4])))))
        elif s[0] == "P":
            items.append(("P", m((s[1], s[2]))))
    xs, ys = [], []
    for kind, data in items:
        pts = data if kind == "L" else ([data] if kind == "P" else [(data[0], data[1]), (data[2], data[3])])
        xs += [p[0] for p in pts]
        ys += [p[1] for p in pts]
    if not xs:
        return []
    pad = w
    ox, oy = min(xs) - pad, min(ys) - pad
    size = (int((max(xs) - ox + pad) * S) + 2, int((max(ys) - oy + pad) * S) + 2)
    img = Image.new("L", size, 0)
    d = ImageDraw.Draw(img)

    def px(p):
        return ((p[0] - ox) * S, (p[1] - oy) * S)

    r = w / 2 * S
    for kind, data in items:
        if kind == "L":
            pts = [px(p) for p in data]
            if len(pts) > 1:
                d.line(pts, fill=255, width=int(round(w * S)), joint="curve")
            for p in pts:
                d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=255)
        elif kind == "D":
            a, b = px((data[0], data[1])), px((data[2], data[3]))
            d.ellipse((a[0] - r, a[1] - r, b[0] + r, b[1] + r), fill=255)
        elif kind == "R":
            a, b = px((data[0], data[1])), px((data[2], data[3]))
            d.rounded_rectangle((a[0] - r, a[1] - r, b[0] + r, b[1] + r), radius=r * 1.1, fill=255)
        elif kind == "P":
            p, rp = px(data), r * 1.15
            d.ellipse((p[0] - rp, p[1] - rp, p[0] + rp, p[1] + rp), fill=255)
    mask = np.array(img) > 127
    return affine(trace(mask), 1 / S, 1 / S, ox, oy)


# ── 자음 배치 ──
def fit_box(ink, max_aspect: float):
    x0, y0, x1, y1 = ink
    h, w = y1 - y0, x1 - x0
    if w > h * max_aspect:
        cx = (x0 + x1) / 2
        x0, x1 = cx - h * max_aspect / 2, cx + h * max_aspect / 2
    return (x0, y0, x1, y1)


@lru_cache(maxsize=None)
def consonant(ch: str, ink: tuple, role: str) -> tuple:
    """자음 하나(쌍자음·겹받침 포함)를 잉크 상자에. 결과는 외곽선(튜플로 캐시)."""
    if ch in PAIRS:
        a, b = PAIRS[ch]
        x0, y0, x1, y1 = ink
        xm = (x0 + x1) / 2
        return consonant(a, (x0, y0, xm - GAP / 2, y1), role) + consonant(b, (xm + GAP / 2, y0, x1, y1), role)
    ink = fit_box(ink, ASPECT.get(ch, MAX_ASPECT[role]))
    if ch == "ㅇ":  # 정원
        x0, y0, x1, y1 = ink
        s = min(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ink = (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)
    x0, y0, x1, y1 = ink
    w = min(W, THIN.get(ch, 0.36) * min(x1 - x0, y1 - y0))
    return tuple(render(CONS[ch], ink, w))


@lru_cache(maxsize=None)
def vertical(ch: str, y0: float, y1: float) -> tuple[tuple, float]:
    """세로 모음(ㅏ…ㅣ)을 오른쪽에 붙여 그린다. (외곽선, 잉크 왼쪽 x)"""
    nw, strokes = VERT[ch]
    ink = (HR - W - nw, y0, HR, y1)
    return tuple(render(strokes, ink, W)), ink[0]


@lru_cache(maxsize=None)
def horizontal(ch: str, x0: float, x1: float, yb: float, tl: float) -> tuple:
    """가로 모음: 중심선 yb 의 가로획 + 가지(위/아래, 길이 tl). x0..x1 은 잉크."""
    strokes = [L((0, 0.5), (1, 0.5))]
    if ch in "ㅗㅛ":
        ys = (0, 0.5)
    elif ch in "ㅜㅠ":
        ys = (0.5, 1)
    else:
        ys = None
    if ys:
        for u in ((0.5,) if ch in "ㅗㅜ" else (0.33, 0.67)):
            strokes.append(L((u, ys[0]), (u, ys[1])))
    ink = (x0, yb - tl - W / 2, x1, yb + tl + W / 2)
    return tuple(render(strokes, ink, W))


def syllable(ch: str) -> Outline:
    syl = decompose(ch)
    cho, jung, jong = CHO[syl.cho], JUNG[syl.jung], JONG[syl.jong]
    t = syl.type
    out: list = []
    if t[0] == "V":
        lay = VLAY[t]
        vow, vl = vertical(jung, *lay["vow_y"])
        out += vow
        out += consonant(cho, (HL, lay["cho_y"][0], vl - lay["cho_gap"], lay["cho_y"][1]), "V")
    elif t[0] == "H":
        lay = HLAY[t]
        out += consonant(cho, lay["cho"], "H")
        key = "up" if jung in "ㅗㅛ" else "down" if jung in "ㅜㅠ" else "flat"
        yb, tl = (lay[key], 0) if key == "flat" else lay[key]
        out += horizontal(jung, HL, HR, yb, tl)
    else:
        lay = MLAY[t]
        hpart, vpart = M_PARTS[jung]
        vow, vl = vertical(vpart, *lay["vow_y"])
        out += vow
        right = vl - lay["cho_gap"]
        out += consonant(cho, (HL, lay["cho_y"][0], right, lay["cho_y"][1]), "M")
        key = "up" if hpart == "ㅗ" else "down" if hpart == "ㅜ" else "flat"
        yb, tl = (lay[key], 0) if key == "flat" else lay[key]
        out += horizontal(hpart, HL, right, yb, tl)
    if jong != " ":
        out += consonant(jong, lay["jong"], "J")
    return list(out)


# ── 영문·숫자·문장부호 (상자 v=0 윗선, v=1 기준선; 소문자는 x-높이 기준, 위아래로 넘길 수 있다) ──
CAP_INK, X_INK, BASE = 110, 330, 800
ASC = -(BASE - CAP_INK - LW) / (BASE - X_INK - LW)  # 소문자 상자에서 어센더 꼭대기 v
DESC = 1 + 118 / (BASE - X_INK - LW)  # 디센더 바닥 v
CIRC = ("A", 0.5, 0.5, 0.5, 0.5, 0, 360)
LATIN: dict[str, tuple[str, float, list[Stroke]]] = {  # 글자: (상자 종류 cap|x, 너비/높이, 획)
    "A": ("cap", 0.78, [L((0, 1), (0.5, 0), (1, 1)), L((0.17, 0.66), (0.83, 0.66))]),
    "B": ("cap", 0.62, [STEM_L, L((0, 0), (0.55, 0)), ("A", 0.55, 0.25, 0.45, 0.25, -90, 90), L((0.55, 0.5), (0, 0.5), (0.58, 0.5)), ("A", 0.58, 0.75, 0.42, 0.25, -90, 90), L((0.58, 1), (0, 1))]),
    "C": ("cap", 0.74, [("A", 0.5, 0.5, 0.5, 0.5, 40, 320)]),
    "D": ("cap", 0.7, [STEM_L, L((0, 0), (0.4, 0)), ("A", 0.4, 0.5, 0.6, 0.5, -90, 90), L((0.4, 1), (0, 1))]),
    "E": ("cap", 0.58, [L((1, 0), (0, 0), (0, 1), (1, 1)), L((0, 0.5), (0.85, 0.5))]),
    "F": ("cap", 0.56, [L((1, 0), (0, 0), (0, 1)), L((0, 0.5), (0.8, 0.5))]),
    "G": ("cap", 0.76, [("A", 0.5, 0.5, 0.5, 0.5, 40, 360), L((0.55, 0.5), (1, 0.5), (1, 0.75))]),
    "H": ("cap", 0.68, [STEM_L, STEM_R, L((0, 0.5), (1, 0.5))]),
    "I": ("cap", 0, [STEM_L]),
    "J": ("cap", 0.55, [L((1, 0), (1, 0.7)), ("A", 0.5, 0.7, 0.5, 0.3, 0, 180)]),
    "K": ("cap", 0.64, [STEM_L, L((1, 0), (0, 0.55)), L((0.3, 0.38), (1, 1))]),
    "L": ("cap", 0.55, [L((0, 0), (0, 1), (1, 1))]),
    "M": ("cap", 0.86, [L((0, 1), (0, 0), (0.5, 0.7), (1, 0), (1, 1))]),
    "N": ("cap", 0.7, [L((0, 1), (0, 0), (1, 1), (1, 0))]),
    "O": ("cap", 0.8, [CIRC]),
    "P": ("cap", 0.6, [L((0, 1), (0, 0), (0.55, 0)), ("A", 0.55, 0.27, 0.45, 0.27, -90, 90), L((0.55, 0.54), (0, 0.54))]),
    "Q": ("cap", 0.8, [CIRC, L((0.62, 0.68), (1.05, 1.08))]),
    "R": ("cap", 0.62, [L((0, 1), (0, 0), (0.55, 0)), ("A", 0.55, 0.27, 0.45, 0.27, -90, 90), L((0.55, 0.54), (0, 0.54)), L((0.45, 0.54), (1, 1))]),
    "S": ("cap", 0.62, [("A", 0.5, 0.26, 0.5, 0.26, 90, 320), ("A", 0.5, 0.74, 0.5, 0.26, 270, 500)]),
    "T": ("cap", 0.66, [L((0, 0), (1, 0)), L((0.5, 0), (0.5, 1))]),
    "U": ("cap", 0.68, [L((0, 0), (0, 0.6)), ("A", 0.5, 0.6, 0.5, 0.4, 0, 180), L((1, 0), (1, 0.6))]),
    "V": ("cap", 0.76, [L((0, 0), (0.5, 1), (1, 0))]),
    "W": ("cap", 1.06, [L((0, 0), (0.25, 1), (0.5, 0.15), (0.75, 1), (1, 0))]),
    "X": ("cap", 0.7, [L((0, 0), (1, 1)), L((1, 0), (0, 1))]),
    "Y": ("cap", 0.7, [L((0, 0), (0.5, 0.5), (1, 0)), L((0.5, 0.5), (0.5, 1))]),
    "Z": ("cap", 0.65, [L((0, 0), (1, 0), (0, 1), (1, 1))]),
    "a": ("x", 0.95, [CIRC, STEM_R]),
    "b": ("x", 0.95, [L((0, ASC), (0, 1)), CIRC]),
    "c": ("x", 0.9, [("A", 0.5, 0.5, 0.5, 0.5, 40, 320)]),
    "d": ("x", 0.95, [L((1, ASC), (1, 1)), CIRC]),
    "e": ("x", 0.95, [("A", 0.5, 0.5, 0.5, 0.5, 40, 360), L((0, 0.5), (1, 0.5))]),
    "f": ("x", 0.6, [L((0.1, 1), (0.1, -0.2)), ("A", 0.6, -0.2, 0.5, 0.37, 180, 300), L((-0.1, 0), (0.8, 0))]),
    "g": ("x", 0.95, [CIRC, L((1, 0), (1, 0.95)), ("A", 0.5, 0.95, 0.5, 0.36, 0, 180)]),
    "h": ("x", 0.85, [L((0, ASC), (0, 1)), ("A", 0.5, 0.5, 0.5, 0.5, 180, 360), L((1, 0.5), (1, 1))]),
    "i": ("x", 0, [STEM_L, ("P", 0, -0.42)]),
    "j": ("x", 0.3, [L((1, 0), (1, 1)), ("A", 0.5, 1, 0.5, 0.3, 0, 180), ("P", 1, -0.42)]),
    "k": ("x", 0.75, [L((0, ASC), (0, 1)), L((1, 0), (0, 0.6)), L((0.3, 0.42), (1, 1))]),
    "l": ("x", 0, [L((0, ASC), (0, 1))]),
    "m": ("x", 1.5, [STEM_L, ("A", 0.25, 0.4, 0.25, 0.4, 180, 360), L((0.5, 0.4), (0.5, 1)), ("A", 0.75, 0.4, 0.25, 0.4, 180, 360), L((1, 0.4), (1, 1))]),
    "n": ("x", 0.85, [STEM_L, ("A", 0.5, 0.5, 0.5, 0.5, 180, 360), L((1, 0.5), (1, 1))]),
    "o": ("x", 0.95, [CIRC]),
    "p": ("x", 0.95, [L((0, 0), (0, DESC)), CIRC]),
    "q": ("x", 0.95, [L((1, 0), (1, DESC)), CIRC]),
    "r": ("x", 0.55, [STEM_L, ("A", 0.5, 0.5, 0.5, 0.5, 180, 280)]),
    "s": ("x", 0.8, [("A", 0.5, 0.26, 0.5, 0.26, 90, 320), ("A", 0.5, 0.74, 0.5, 0.26, 270, 500)]),
    "t": ("x", 0.6, [L((0.4, -0.35), (0.4, 0.65)), ("A", 0.7, 0.65, 0.3, 0.35, 60, 180), L((0, 0), (0.85, 0))]),
    "u": ("x", 0.85, [L((0, 0), (0, 0.5)), ("A", 0.5, 0.5, 0.5, 0.5, 0, 180), STEM_R]),
    "v": ("x", 0.85, [L((0, 0), (0.5, 1), (1, 0))]),
    "w": ("x", 1.3, [L((0, 0), (0.25, 1), (0.5, 0.1), (0.75, 1), (1, 0))]),
    "x": ("x", 0.8, [L((0, 0), (1, 1)), L((1, 0), (0, 1))]),
    "y": ("x", 0.85, [L((0, 0), (0.5, 1)), L((1, 0), (0.3, DESC))]),
    "z": ("x", 0.75, [L((0, 0), (1, 0), (0, 1), (1, 1))]),
    "0": ("cap", 0.68, [CIRC]),
    "1": ("cap", 0.4, [L((0, 0.3), (0.6, 0), (0.6, 1))]),
    "2": ("cap", 0.62, [("A", 0.5, 0.3, 0.5, 0.3, 180, 400), L((0.88, 0.49), (0, 1), (1, 1))]),
    "3": ("cap", 0.62, [("A", 0.5, 0.26, 0.5, 0.26, 180, 450), ("A", 0.5, 0.74, 0.5, 0.26, 270, 540)]),
    "4": ("cap", 0.7, [L((0.75, 1), (0.75, 0), (0, 0.72), (1, 0.72))]),
    "5": ("cap", 0.62, [L((0.95, 0), (0.1, 0), (0.03, 0.48)), ("A", 0.5, 0.72, 0.5, 0.28, 200, 520)]),
    "6": ("cap", 0.65, [("A", 0.5, 0.7, 0.5, 0.3, 0, 360), ("A", 0.5, 0.55, 0.5, 0.55, 180, 300), L((0, 0.55), (0, 0.7))]),
    "7": ("cap", 0.62, [L((0, 0), (1, 0), (0.35, 1))]),
    "8": ("cap", 0.62, [("A", 0.5, 0.26, 0.44, 0.26, 0, 360), ("A", 0.5, 0.72, 0.5, 0.28, 0, 360)]),
    "9": ("cap", 0.65, [("A", 0.5, 0.3, 0.5, 0.3, 0, 360), L((1, 0.3), (1, 0.45)), ("A", 0.5, 0.45, 0.5, 0.55, 0, 120)]),
    ".": ("cap", 0, [("P", 0, 0.93)]),
    ",": ("cap", 0.1, [L((1, 0.93), (0, 1.2))]),
    "!": ("cap", 0, [L((0, 0), (0, 0.66)), ("P", 0, 0.93)]),
    "?": ("cap", 0.62, [("A", 0.5, 0.3, 0.5, 0.3, 180, 450), L((0.5, 0.6), (0.5, 0.7)), ("P", 0.5, 0.93)]),
    "-": ("cap", 0.45, [L((0, 0.55), (1, 0.55))]),
    ":": ("cap", 0, [("P", 0, 0.45), ("P", 0, 0.93)]),
    ";": ("cap", 0.1, [("P", 0.5, 0.45), L((1, 0.93), (0, 1.2))]),
    "'": ("cap", 0, [L((0, 0), (0, 0.22))]),
    '"': ("cap", 0.28, [L((0, 0), (0, 0.22)), L((1, 0), (1, 0.22))]),
    "(": ("cap", 0.4, [("A", 1, 0.5, 1, 0.62, 110, 250)]),
    ")": ("cap", 0.4, [("A", 0, 0.5, 1, 0.62, 290, 430)]),
    "/": ("cap", 0.55, [L((0, 1.05), (1, -0.05))]),
    "+": ("cap", 0.6, [L((0, 0.55), (1, 0.55)), L((0.5, 0.2), (0.5, 0.9))]),
    "=": ("cap", 0.6, [L((0, 0.4), (1, 0.4)), L((0, 0.7), (1, 0.7))]),
    "_": ("cap", 0.6, [L((0, 1.1), (1, 1.1))]),
    "~": ("cap", 0.7, [("A", 0.25, 0.55, 0.25, 0.12, 180, 360), ("A", 0.75, 0.55, 0.25, 0.12, 0, 180)]),
    "<": ("cap", 0.55, [L((1, 0.2), (0, 0.55), (1, 0.9))]),
    ">": ("cap", 0.55, [L((0, 0.2), (1, 0.55), (0, 0.9))]),
    "|": ("cap", 0, [L((0, -0.05), (0, 1.15))]),
    "*": ("cap", 0.5, [L((0.5, 0.1), (0.5, 0.6)), L((0.07, 0.22), (0.93, 0.48)), L((0.93, 0.22), (0.07, 0.48))]),
}
LATIN_SB = 60


def latin(ch: str) -> tuple[Outline, int]:
    kind, ratio, strokes = LATIN[ch]
    top = CAP_INK if kind == "cap" else X_INK
    ch_c = BASE - top - LW  # 중심선 높이
    ink = (LATIN_SB, top, LATIN_SB + LW + ratio * ch_c, BASE)
    outline = render(strokes, ink, LW)
    b = bounds(outline)
    if b is None:
        return [], 0
    return affine(outline, 1, 1, LATIN_SB - b.x0, 0), round(b.w + 2 * LATIN_SB)


def compat(ch: str) -> Outline:
    """호환 자모 ㄱ~ㅣ: 자음은 가운데 상자, 모음은 세로/가로 자리에."""
    if ch in CONS or ch in PAIRS:
        return list(consonant(ch, (190, 150, 810, 690), "H"))
    if ch in VERT:
        nw = VERT[ch][0]
        out, _ = vertical(ch, 40, 790)
        return affine(list(out), 1, 1, -(HR - (W + nw) / 2) + 500, 0)
    if ch in "ㅗㅛㅜㅠㅡ":
        key = "up" if ch in "ㅗㅛ" else "down" if ch in "ㅜㅠ" else "flat"
        yb, tl = (560, 0) if key == "flat" else ((560, 160) if key == "up" else (480, 160))
        return list(horizontal(ch, 110, 890, yb, tl))
    h, v = M_PARTS[ch]
    out, vl = vertical(v, 30, 800)
    key = "up" if h == "ㅗ" else "down" if h == "ㅜ" else "flat"
    yb, tl = (580, 0) if key == "flat" else ((580, 100) if key == "up" else (540, 120))
    return list(out) + list(horizontal(h, HL, vl - 50, yb, tl))


def to_font(outline: Outline) -> Outline:
    return affine(outline, 1, -1, 0, BASE)


def build(out_otf: Path, family: str, uid: str, names: dict[str, str]) -> dict[str, int]:
    order = [".notdef", "space"]
    cmap = {0x20: "space", 0x3000: "space"}
    strings = {".notdef": charstring([], 500), "space": charstring([], SPACE_W)}
    metrics = {".notdef": (500, 0), "space": (SPACE_W, 0)}
    count = {"hangul": 0, "latin": 0, "jamo": 0}

    def add(ch: str, outline: Outline, adv: int, kind: str):
        if not outline:
            return
        name = f"uni{ord(ch):04X}"
        placed = to_font(outline)
        order.append(name)
        cmap[ord(ch)] = name
        strings[name] = charstring(placed, adv)
        metrics[name] = (adv, round(bounds(placed).x0))
        count[kind] += 1

    for ch in all_syllables():
        add(ch, syllable(ch), UPM, "hangul")
    for ch in LATIN:
        o, adv = latin(ch)
        add(ch, o, adv, "latin")
    for code in range(0x3131, 0x3164):
        ch = chr(code)
        if ch in CONS or ch in PAIRS or ch in VERT or ch in "ㅗㅛㅜㅠㅡ" or ch in M_PARTS:
            add(ch, compat(ch), UPM, "jamo")

    psn = f"{ps_name(family)}-{uid[:8]}"
    fb = FontBuilder(UPM, isTTF=False)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(cmap)
    fb.setupCFF(psn, {"FullName": psn}, strings, {})
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ASCENT, descent=DESCENT)
    fb.setupNameTable(
        {"familyName": {"en": family, "ko": family}, "styleName": "Regular", "uniqueFontIdentifier": f"font.sw4u.kr:{uid}",
         "fullName": {"en": family, "ko": family}, "psName": psn, "version": "Version 1.000",
         "manufacturer": "font.sw4u.kr", "description": "디지털·미래 느낌의 둥근 고딕 — 기하학적 자모 뼈대로 그린 폰트 (font.sw4u.kr)",
         "copyright": names["copyright"], "licenseDescription": names["license_text"], "licenseInfoURL": names["license_url"]},
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
    return count


SAMPLE = ("나이스 디미체 — 디지털과 미래", "다람쥐 헌 쳇바퀴에 타고파", "꿈꾸는 빛의 숲, 밝은 뜰에 앉아 쉼표를 찍다",
          "값 뷁 쾅 뚫 흙 뭉 쁘 쒸 왜 쥐 랩 홈 뗏 릎 쵸 휙 괜 봤", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz",
          "0123456789 .,!?-:;'\"()/+=_~<>|*", "The quick brown fox jumps over the lazy dog")


def sample(otf: Path, out: Path) -> None:
    font = ImageFont.truetype(str(otf), 56)
    img = Image.new("RGB", (1500, 90 * len(SAMPLE) + 200), "white")
    d = ImageDraw.Draw(img)
    for i, line in enumerate(SAMPLE):
        d.text((30, 25 + i * 90), line, font=font, fill=(20, 20, 30))
    small = "작은 크기에서도 읽기 편한지 봅니다. 한글 11,172자와 영문·숫자가 모두 들어 있어요. 2026년 10월 11일, font.sw4u.kr"
    y = 90 * len(SAMPLE) + 50
    for size in (28, 20, 14):
        d.text((30, y), small, font=ImageFont.truetype(str(otf), size), fill=(20, 20, 30))
        y += size + 18
    img = img.crop((0, 0, 1500, y + 10))
    img.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--uid", default="dimi0001" * 4)
    ap.add_argument("--family", default="나이스 디미체")
    ap.add_argument("--author", default="Claude")
    ap.add_argument("--sample", type=Path)
    a = ap.parse_args()
    names = {"copyright": f"© 2026 {a.author} & font.sw4u.kr. 공동 저작권.",
             "license_text": "저작권은 만든 사람과 font.sw4u.kr 이 공동으로 보유합니다. 누구나 상업적 이용을 포함해 자유롭게 사용·배포할 수 있습니다. 폰트 파일 자체의 판매는 금지합니다.",
             "license_url": "https://font.sw4u.kr/terms#public_commercial"}
    otf = a.out / "font.otf"
    count = build(otf, a.family, a.uid, names)
    to_woff2(otf, a.out / "font.woff2")
    preview(otf, a.out / "preview.png")
    if a.sample:
        sample(otf, a.sample)
    print(f"한글 {count['hangul']} · 영문/기호 {count['latin']} · 자모 {count['jamo']} → {otf} ({otf.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()
