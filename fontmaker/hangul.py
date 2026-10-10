"""한글 음절 ↔ 자모, 손글씨 음절을 초·중·종성 부품으로 나누기.

부품 분리는 연결요소(획 덩어리) 단위로 위치 규칙에 따라 배정한다. 자모끼리 붙여 쓴 글자는
분리에 실패할 수 있고, 그 글자는 통째로만 쓰고 부품 공급원에서는 빠진다.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"

V_CLASS = {0: "V", 1: "V", 2: "V", 3: "V", 4: "V", 5: "V", 6: "V", 7: "V", 20: "V",
           8: "H", 12: "H", 13: "H", 17: "H", 18: "H",
           9: "M", 10: "M", 11: "M", 14: "M", 15: "M", 16: "M", 19: "M"}
DOWN_TICK = {13, 17}  # ㅜ ㅠ: 가로획 아래로 내려오는 점
TWO_STROKES = {1, 3, 5, 7}  # ㅐ ㅒ ㅔ ㅖ
ROLES = ("cho", "jung", "jong")


@dataclass(frozen=True)
class Syllable:
    cho: int
    jung: int
    jong: int  # 0 = 받침 없음

    @property
    def vclass(self) -> str:
        return V_CLASS[self.jung]

    @property
    def type(self) -> str:
        """위치 유형: V0 V1 H0 H1 M0 M1 (모음 방향 + 받침 유무)."""
        return f"{self.vclass}{int(self.jong > 0)}"

    def jamo(self, role: str) -> int:
        return getattr(self, role)

    @property
    def char(self) -> str:
        return chr(0xAC00 + (self.cho * 21 + self.jung) * 28 + self.jong)


def decompose(ch: str) -> Syllable | None:
    code = ord(ch) - 0xAC00
    if not 0 <= code < 11172:
        return None
    return Syllable(code // 588, (code % 588) // 28, code % 28)


def all_syllables() -> list[str]:
    """현대 한글 완성형 11,172자."""
    return [chr(c) for c in range(0xAC00, 0xD7A4)]


# 초성 ↔ 받침 같은 꼴 대응 (부품 대체용)
CHO_TO_JONG = {CHO.index(c): JONG.index(c) for c in CHO if c in JONG}
JONG_TO_CHO = {v: k for k, v in CHO_TO_JONG.items()}


@dataclass(frozen=True, eq=False)
class Comp:
    mask: np.ndarray  # 칸 전체 크기의 bool
    x0: int
    y0: int
    x1: int
    y1: int
    area: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def components(mask: np.ndarray) -> list[Comp]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        out.append(Comp(labels == i, int(x), int(y), int(x + w), int(y + h), int(area)))
    return out


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0


def bbox(comps: list[Comp]) -> Box:
    return Box(min(c.x0 for c in comps), min(c.y0 for c in comps), max(c.x1 for c in comps), max(c.y1 for c in comps))


def bar_row(c: Comp) -> int:
    """가로획 덩어리에서 잉크가 가장 많은 행."""
    return c.y0 + int(c.mask[c.y0:c.y1].sum(1).argmax())


def split_vertical(comps: list[Comp], ink: Box, syl: Syllable) -> dict[str, list[Comp]] | None:
    """세로 모음(ㅏㅓㅣ…): 오른쪽 긴 세로획 = 중성, 그 아래 = 종성, 나머지 = 초성."""
    tall = [c for c in comps if c.h >= 0.35 * ink.h and c.x0 > ink.x0 + 0.3 * ink.w and c.y0 < ink.y0 + 0.3 * ink.h]
    if not tall:
        return None
    stroke = max(tall, key=lambda c: c.x1)
    jung = [stroke]
    jung += [c for c in tall if c is not stroke and c.x1 > stroke.x0 - 0.35 * ink.w and c.h >= 0.4 * stroke.h]  # ㅐㅔ
    left = min(c.x0 for c in jung)
    for c in comps:
        if c in jung:
            continue
        near = c.x1 >= left - 0.12 * ink.w and c.w < 0.3 * ink.w and c.h < 0.25 * ink.h
        if near and stroke.y0 - 0.05 * ink.h <= c.cy <= stroke.y1:
            jung.append(c)
    rest = [c for c in comps if c not in jung]
    if not syl.jong:
        return {"cho": rest, "jung": jung, "jong": []}
    bottom = max(c.y1 for c in jung)
    jong = [c for c in rest if c.y0 >= bottom - 0.08 * ink.h or c.cy > bottom]
    return {"cho": [c for c in rest if c not in jong], "jung": jung, "jong": jong}


def split_horizontal(comps: list[Comp], ink: Box, syl: Syllable) -> dict[str, list[Comp]] | None:
    """가로 모음(ㅗㅜㅡ…): 넓은 가로획 = 중성, 위 = 초성, 아래 = 종성(ㅜㅠ 의 내림 점 제외)."""
    wide = [c for c in comps if c.w >= 0.5 * ink.w and c.cy > ink.y0 + 0.25 * ink.h]
    if not wide:
        return None
    bar = max(wide, key=lambda c: c.w)
    row = bar_row(bar)
    jung = [bar] + [c for c in comps if c is not bar and _is_tick(c, bar, row, ink, syl.jung in DOWN_TICK)]
    rest = [c for c in comps if c not in jung]
    cho = [c for c in rest if c.cy < row]
    below = [c for c in rest if c.cy >= row]
    if syl.jong:
        return {"cho": cho, "jung": jung, "jong": below}
    return {"cho": cho, "jung": jung + below, "jong": []}


def _is_tick(c: Comp, bar: Comp, row: int, ink: Box, down: bool) -> bool:
    """가로획에 붙은 짧은 점(ㅗ 위, ㅜ 아래)."""
    if not (bar.x0 <= c.cx <= bar.x1) or c.w >= 0.22 * ink.w or c.h >= 0.3 * ink.h:
        return False
    up = row - 0.08 * ink.h <= c.y1 <= row + 0.03 * ink.h
    dn = down and row - 0.03 * ink.h <= c.y0 <= row + 0.1 * ink.h
    return up or dn


def split_mixed(comps: list[Comp], ink: Box, syl: Syllable) -> dict[str, list[Comp]] | None:
    """복합 모음(ㅘㅝㅢ…): 오른쪽 세로획 + 가로획 = 중성."""
    tall = [c for c in comps if c.h >= 0.35 * ink.h and c.x0 > ink.x0 + 0.4 * ink.w and c.y0 < ink.y0 + 0.3 * ink.h]
    if not tall:
        return None
    stroke = max(tall, key=lambda c: c.x1)
    wide = [c for c in comps if c is not stroke and c.w >= 0.35 * ink.w and c.cy > ink.y0 + 0.25 * ink.h]
    if stroke.w >= 0.5 * ink.w:  # 가로획이 세로획에 붙어 한 덩어리
        bar = stroke
    elif wide:
        bar = max(wide, key=lambda c: c.w)
    else:
        return None
    row = bar_row(bar)
    jung = [stroke] if bar is stroke else [stroke, bar]
    for c in comps:
        if c in jung:
            continue
        small = c.w < 0.22 * ink.w and c.h < 0.3 * ink.h
        tick_v = small and bar.x0 <= c.cx <= stroke.x0 and abs((c.y1 if c.cy < row else c.y0) - row) < 0.12 * ink.h
        tick_h = small and c.x0 >= stroke.x0 - 0.05 * ink.w  # ㅘ 의 오른쪽 점 / ㅝ 의 왼쪽 점
        if tick_v or tick_h:
            jung.append(c)
    rest = [c for c in comps if c not in jung]
    cho = [c for c in rest if c.cy < row]
    jong = [c for c in rest if c.cy >= row]
    if not syl.jong:
        return {"cho": cho, "jung": jung + jong, "jong": []}
    return {"cho": cho, "jung": jung, "jong": jong}


SPLITTERS = {"V": split_vertical, "H": split_horizontal, "M": split_mixed}


def segment(mask: np.ndarray, syl: Syllable) -> dict[str, np.ndarray] | None:
    """음절 비트맵 → {"cho","jung","jong"} 부품 마스크. 규칙이 안 맞으면 붙은 곳을 잘라 다시 시도, 끝내 안 되면 None."""
    for cut in _cut_variants(mask, syl):
        parts = _assign(cut, syl)
        if parts is not None:
            return parts
    return None


def _cut_variants(mask: np.ndarray, syl: Syllable):
    """원본 → 자모가 맞닿은 가장 가는 곳을 1~3군데 자른 변형들."""
    yield mask
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = y1 - y0, x1 - x0
    rows, cols = mask.sum(1), mask.sum(0)
    lines: list[tuple[str, int]] = []
    if syl.vclass in "HM":
        bar = _argext(rows, y0 + 0.35 * h, y1, max)
        lines.append(("row", _argext(rows, y0 + 0.2 * h, bar - 0.04 * h, min)))
    if syl.vclass in "VM":
        strong = np.nonzero(cols[int(x0 + 0.4 * w):x1] >= 0.75 * cols.max())[0]
        stroke = int(x0 + 0.4 * w) + int(strong[0]) if len(strong) else _argext(cols, x0 + 0.45 * w, x1, max)
        lines.append(("col", _argext(cols, x0 + 0.25 * w, stroke - 0.04 * w, min)))
    if syl.jong:
        lines.append(("row", _argext(rows, y0 + 0.45 * h, y1 - 0.15 * h, min)))
    for n in range(1, len(lines) + 1):
        for combo in _combos(lines, n):
            m = mask.copy()
            for axis, at in combo:
                if axis == "row":
                    m[max(at - 1, 0):at + 1, :] = False
                else:
                    m[:, max(at - 1, 0):at + 1] = False
            yield m


def _argext(arr: np.ndarray, lo: float, hi: float, fn) -> int:
    lo_i, hi_i = int(lo), max(int(hi), int(lo) + 1)
    seg = arr[lo_i:hi_i]
    if len(seg) == 0:
        return lo_i
    return lo_i + int(np.argmax(seg) if fn is max else np.argmin(seg))


def _combos(items: list, n: int) -> list[list]:
    from itertools import combinations

    return [list(c) for c in combinations(items, n)]


def _assign(mask: np.ndarray, syl: Syllable) -> dict[str, np.ndarray] | None:
    comps = components(mask)
    if len(comps) < 2:
        return None
    ink = bbox(comps)
    parts = SPLITTERS[syl.vclass](comps, ink, syl)
    if parts is None or not parts["cho"] or not parts["jung"] or bool(syl.jong) != bool(parts["jong"]):
        return None
    if not _plausible(parts, syl):
        return None
    out = {}
    for role, cs in parts.items():
        if cs:
            m = np.zeros(mask.shape, dtype=bool)
            for c in cs:
                m |= c.mask
            out[role] = m
    return out


def _plausible(parts: dict[str, list[Comp]], syl: Syllable) -> bool:
    cho, jung = bbox(parts["cho"]), bbox(parts["jung"])
    if syl.vclass == "V" and cho.x0 > jung.x0:
        return False
    if syl.jung in TWO_STROKES and _vertical_runs(parts["jung"], jung) < 2:
        return False  # ㅐㅔㅒㅖ 는 세로획 둘
    if syl.vclass == "H" and cho.y0 > jung.y0 + jung.h * 0.5:
        return False
    if parts["jong"]:
        jong = bbox(parts["jong"])
        if jong.y0 < cho.y0 + cho.h * 0.5:
            return False
    return True


def _vertical_runs(comps: list[Comp], box: Box) -> int:
    """세로로 긴 획(칸 높이 35% 이상 잉크가 쌓인 열 묶음)의 개수."""
    m = np.zeros_like(comps[0].mask)
    for c in comps:
        m |= c.mask
    strong = m.sum(0) >= 0.35 * box.h
    return int(np.count_nonzero(strong[1:] & ~strong[:-1]) + strong[0])
