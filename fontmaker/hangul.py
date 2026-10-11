"""한글 음절 ↔ 자모, 손글씨 음절을 초·중·종성 부품으로 나누기.

부품 분리는 여러 후보(획 덩어리 규칙 배정·기준 상자 배정·붙은 곳 자르기·픽셀 가르기)를 만들고
기준 배치(reference.py, Noto Sans KR Thin)와 가장 닮은 것을 고른다. 끝내 못 나눈 글자는 통째로만 쓰고
부품 공급원에서는 빠진다.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import reference

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
CUT_COST = 0.04  # 자른 곳 하나당 점수 깎기 (덜 자른 후보 우선)
PIXEL_COST = 0.08  # 기준 상자로 픽셀을 가르는 최후 수단의 점수 깎기
MIN_SCORE = 0.3  # 이보다 기준 배치와 안 닮으면 분리 실패로 본다


def segment(mask: np.ndarray, syl: Syllable) -> dict[str, np.ndarray] | None:
    """음절 비트맵 → {"cho","jung","jong"} 부품 마스크. 끝내 못 나누면 None.

    후보: (원본 + 붙은 곳을 1~3군데 자른 변형) × (모음 방향 규칙 분리, 기준 상자 겹침 배정) + 기준 상자로 픽셀 가르기.
    그중 역할별 상자가 기준 배치(reference, 같은 음절의 Noto Thin 배치)와 가장 닮은 후보를 고른다.
    """
    ref = _ref_boxes(syl)
    best, best_score = None, MIN_SCORE
    for cut, n_cuts in _cut_variants(mask, syl):
        comps = components(cut)
        if len(comps) < 2:
            continue
        ink = bbox(comps)
        for parts in (SPLITTERS[syl.vclass](comps, ink, syl), _by_boxes(comps, ink, ref, syl)):
            if parts is None or not _complete(parts, syl) or not _plausible(parts, syl):
                continue
            masks = _union(parts, mask.shape)
            score = _score(masks, ref) - CUT_COST * n_cuts
            if score > best_score:
                best, best_score = masks, score
    pixel = _by_pixels(mask, ref, syl)
    if pixel is not None and _score(pixel, ref) - PIXEL_COST > best_score:
        best = pixel
    return best


def _ref_boxes(syl: Syllable) -> dict[str, np.ndarray]:
    """기준 배치의 역할 상자를 잉크 전체 상자 기준 0~1 로."""
    boxes = {r: b for r in ROLES if (b := reference.box(syl.char, r)) is not None}
    allb = np.array(list(boxes.values()))
    x0, y0 = allb[:, :2].min(0)
    x1, y1 = allb[:, 2:].max(0)
    scale = np.array([x1 - x0, y1 - y0, x1 - x0, y1 - y0])
    return {r: (b - [x0, y0, x0, y0]) / scale for r, b in boxes.items()}


def _norm_box(mask: np.ndarray, ink: Box) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.array([(xs.min() - ink.x0) / ink.w, (ys.min() - ink.y0) / ink.h,
                     (xs.max() + 1 - ink.x0) / ink.w, (ys.max() + 1 - ink.y0) / ink.h])


def _iou(a: np.ndarray, b: np.ndarray, pad: float = 0.06) -> float:
    """상자 IoU. 둘 다 획 굵기쯤(pad, 잉크 상자 비율) 넓혀서 견준다 — ㅡ 처럼 얇은 상자가 굵기 차이만으로 0 이 되지 않게."""
    a = a + [-pad, -pad, pad, pad]
    b = b + [-pad, -pad, pad, pad]
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _score(masks: dict[str, np.ndarray], ref: dict[str, np.ndarray]) -> float:
    """역할 상자와 기준 상자의 IoU 평균 (잉크 전체 상자 기준으로 맞춘 뒤)."""
    allm = np.zeros_like(next(iter(masks.values())))
    for m in masks.values():
        allm |= m
    ys, xs = np.nonzero(allm)
    ink = Box(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    return float(np.mean([_iou(_norm_box(masks[r], ink), ref[r]) for r in ref]))


def _complete(parts: dict[str, list[Comp]], syl: Syllable) -> bool:
    return bool(parts["cho"]) and bool(parts["jung"]) and bool(syl.jong) == bool(parts["jong"])


def _union(parts: dict[str, list[Comp]], shape: tuple[int, int]) -> dict[str, np.ndarray]:
    out = {}
    for role, cs in parts.items():
        if cs:
            m = np.zeros(shape, dtype=bool)
            for c in cs:
                m |= c.mask
            out[role] = m
    return out


def _to_ink(ref: dict[str, np.ndarray], ink: Box) -> dict[str, np.ndarray]:
    return {r: b * [ink.w, ink.h, ink.w, ink.h] + [ink.x0, ink.y0, ink.x0, ink.y0] for r, b in ref.items()}


def _role_cost(shape: tuple[int, int], boxes: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """픽셀마다 역할 비용: 상자 밖이면 떨어진 거리, 안이면 상자 넓이(작은 상자 우선 — 큰 모음 상자가 초성을 삼키지 않게)."""
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]] + 0.5
    out = {}
    for r, b in boxes.items():
        gap = np.maximum.reduce([b[0] - xs, xs - b[2], b[1] - ys, ys - b[3], np.zeros(shape)])
        out[r] = gap * 50 + (b[2] - b[0]) * (b[3] - b[1]) / 100
    return out


def _by_boxes(comps: list[Comp], ink: Box, ref: dict[str, np.ndarray], syl: Syllable) -> dict[str, list[Comp]] | None:
    """획 덩어리마다 비용이 가장 낮은 역할 (덩어리 픽셀 합)."""
    cost = _role_cost(comps[0].mask.shape, _to_ink(ref, ink))
    parts: dict[str, list[Comp]] = {r: [] for r in ROLES}
    for c in comps:
        parts[min(cost, key=lambda r: cost[r][c.mask].sum())].append(c)
    return parts


def _by_pixels(mask: np.ndarray, ref: dict[str, np.ndarray], syl: Syllable) -> dict[str, np.ndarray] | None:
    """최후 수단: 픽셀을 기준 상자로 가른다. 몇 픽셀만 넘친 획 덩어리는 통째로 주된 역할에."""
    comps = components(mask)
    if not comps:
        return None
    ink = bbox(comps)
    cost = _role_cost(mask.shape, _to_ink(ref, ink))
    roles = list(cost)
    near = np.argmin(np.stack([cost[r] for r in roles]), axis=0)
    out = {r: np.zeros(mask.shape, bool) for r in roles}
    for c in comps:
        votes = np.bincount(near[c.mask], minlength=len(roles))
        minority = votes.sum() - votes.max()
        if minority <= 0.1 * votes.sum() and minority <= 0.02 * mask.sum():  # 가장자리 몇 픽셀만 넘친 덩어리
            out[roles[int(votes.argmax())]] |= c.mask
        else:
            for k, r in enumerate(roles):
                out[r] |= c.mask & (near == k)
    if not all(out[r].sum() > 0.03 * mask.sum() for r in roles):
        return None
    return out


def _cut_variants(mask: np.ndarray, syl: Syllable):
    """(변형, 자른 곳 수): 원본 → 자모가 맞닿은 가장 가는 곳을 1~3군데 자른 변형들."""
    yield mask, 0
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
            yield m, n


def _argext(arr: np.ndarray, lo: float, hi: float, fn) -> int:
    lo_i, hi_i = int(lo), max(int(hi), int(lo) + 1)
    seg = arr[lo_i:hi_i]
    if len(seg) == 0:
        return lo_i
    return lo_i + int(np.argmax(seg) if fn is max else np.argmin(seg))


def _combos(items: list, n: int) -> list[list]:
    from itertools import combinations

    return [list(c) for c in combinations(items, n)]


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
