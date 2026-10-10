"""직접 쓴 음절에서 뽑은 부품으로 나머지 음절을 조합한다.

부품은 (역할, 자모, 위치유형) 별로 모은다. 같은 유형 부품이 있으면 그 자리 그대로 쓰고,
없으면 다른 유형 부품을 '역할 평균 상자' 사이의 변환으로 옮겨 쓴다.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import median

import numpy as np

from .hangul import CHO, CHO_TO_JONG, JONG, JONG_TO_CHO, Syllable
from .vector import Bounds, Outline, affine, bounds, fit, scale_about, trace

TYPES = ("V0", "V1", "H0", "H1", "M0", "M1")
PAIRS = {"ㄲ": "ㄱㄱ", "ㄸ": "ㄷㄷ", "ㅃ": "ㅂㅂ", "ㅆ": "ㅅㅅ", "ㅉ": "ㅈㅈ", "ㄳ": "ㄱㅅ", "ㄵ": "ㄴㅈ", "ㄶ": "ㄴㅎ",
         "ㄺ": "ㄹㄱ", "ㄻ": "ㄹㅁ", "ㄼ": "ㄹㅂ", "ㄽ": "ㄹㅅ", "ㄾ": "ㄹㅌ", "ㄿ": "ㄹㅍ", "ㅀ": "ㄹㅎ", "ㅄ": "ㅂㅅ"}

# 손글씨 자료가 없을 때 쓰는 역할 상자 (칸 크기 대비 비율)
DEFAULT_BOXES: dict[tuple[str, str], tuple[float, float, float, float]] = {
    ("cho", "V0"): (0.12, 0.22, 0.55, 0.78), ("jung", "V0"): (0.58, 0.10, 0.86, 0.90),
    ("cho", "V1"): (0.14, 0.12, 0.55, 0.50), ("jung", "V1"): (0.58, 0.06, 0.86, 0.62),
    ("jong", "V1"): (0.22, 0.62, 0.80, 0.92),
    ("cho", "H0"): (0.25, 0.10, 0.75, 0.50), ("jung", "H0"): (0.10, 0.48, 0.90, 0.86),
    ("cho", "H1"): (0.27, 0.06, 0.73, 0.36), ("jung", "H1"): (0.10, 0.34, 0.90, 0.60),
    ("jong", "H1"): (0.25, 0.62, 0.75, 0.93),
    ("cho", "M0"): (0.12, 0.10, 0.52, 0.48), ("jung", "M0"): (0.08, 0.08, 0.88, 0.88),
    ("cho", "M1"): (0.12, 0.06, 0.50, 0.36), ("jung", "M1"): (0.08, 0.04, 0.88, 0.62),
    ("jong", "M1"): (0.22, 0.64, 0.80, 0.93),
}


@dataclass(frozen=True)
class Part:
    outline: Outline
    box: Bounds
    type: str
    source: str  # 부품을 뽑은 음절


def mask_bounds(mask: np.ndarray) -> Bounds:
    ys, xs = np.nonzero(mask)
    return Bounds(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


class Library:
    def __init__(self, cell_px: int) -> None:
        self.cell = cell_px
        self.parts: dict[tuple[str, int, str], list[Part]] = defaultdict(list)

    def add(self, syl: Syllable, masks: dict[str, np.ndarray]) -> None:
        for role, mask in masks.items():
            part = Part(trace(mask), mask_bounds(mask), syl.type, syl.char)
            self.parts[(role, syl.jamo(role), syl.type)].append(part)

    def role_boxes(self) -> dict[tuple[str, str], Bounds]:
        """역할·유형별 평균(중앙값) 상자. 자료가 없으면 기본 비율."""
        acc: dict[tuple[str, str], list[Bounds]] = defaultdict(list)
        for (role, _, typ), parts in self.parts.items():
            acc[(role, typ)] += [p.box for p in parts]
        out = {}
        for key, frac in DEFAULT_BOXES.items():
            boxes = acc.get(key)
            if boxes:
                out[key] = Bounds(*(median(getattr(b, a) for b in boxes) for a in ("x0", "y0", "x1", "y1")))
            else:
                out[key] = Bounds(*(f * self.cell for f in frac))
        return out


class Composer:
    def __init__(self, lib: Library) -> None:
        self.lib = lib
        self.boxes = lib.role_boxes()
        self.gap = lib.cell * 0.025

    def _find(self, role: str, jamo: int, typ: str) -> tuple[Part, str] | None:
        """(부품, 부품의 원래 역할). 같은 유형 → 같은 모음방향 → 받침 유무 같은 유형 → 아무 유형 순.
        끝내 없으면 초성↔받침 같은 꼴 자음을 빌린다."""
        order = sorted(TYPES, key=lambda t: (t != typ, t[0] != typ[0], t[1] != typ[1]))
        for t in order:
            found = self.lib.parts.get((role, jamo, t))
            if found and (role, t) in self.boxes:
                return found[0], role
        alt = {"cho": ("jong", CHO_TO_JONG), "jong": ("cho", JONG_TO_CHO)}.get(role)
        if alt and jamo in alt[1]:
            other_role, table = alt
            for t in order:
                found = self.lib.parts.get((other_role, table[jamo], t))
                if found and (other_role, t) in self.boxes:
                    return found[0], other_role
        return None

    def _place(self, role: str, jamo: int, typ: str) -> Outline | None:
        hit = self._find(role, jamo, typ)
        if hit is None:
            return self._place_pair(role, jamo, typ)
        part, src_role = hit
        if part.type == typ and src_role == role:
            return part.outline
        return fit(part.outline, self.boxes[(src_role, part.type)], self.boxes[(role, typ)])

    def _place_pair(self, role: str, jamo: int, typ: str) -> Outline | None:
        """겹받침·쌍자음 샘플이 없으면 홑자음 둘을 상자 왼쪽·오른쪽에 나란히."""
        names = CHO if role == "cho" else JONG
        pair = PAIRS.get(names[jamo])
        if not pair:
            return None
        dst = self.boxes[(role, typ)]
        half = dst.w * 0.5
        out: Outline = []
        for i, single in enumerate(pair):
            outline = self._place(role, names.index(single), typ)
            src = bounds(outline) if outline else None
            if src is None:
                return None
            x0 = dst.x0 + i * half
            out += fit(outline, src, Bounds(x0 + half * 0.04, dst.y0, x0 + half * 0.96, dst.y1))
        return out

    def compose(self, syl: Syllable) -> Outline | None:
        typ = syl.type
        cho = self._place("cho", syl.cho, typ)
        jung = self._place("jung", syl.jung, typ)
        jong = self._place("jong", syl.jong, typ) if syl.jong else []
        if cho is None or jung is None or jong is None:
            return None
        cho = self._avoid_overlap(cho, jung, syl)
        if jong:
            jong = self._push_down(jong, cho + jung)
        return cho + jung + jong

    def _avoid_overlap(self, cho: Outline, jung: Outline, syl: Syllable) -> Outline:
        cb, jb = bounds(cho), bounds(jung)
        if cb is None or jb is None:
            return cho
        if syl.vclass == "V" and cb.x1 > jb.x0 - self.gap and cb.w > 0:
            k = max(0.6, (jb.x0 - self.gap - cb.x0) / cb.w)
            return scale_about(cho, k, 1.0, cb.x0, cb.y0)
        if syl.vclass == "H" and cb.y1 > jb.y0 - self.gap and cb.h > 0:
            k = max(0.6, (jb.y0 - self.gap - cb.y0) / cb.h)
            return scale_about(cho, 1.0, k, cb.x0, cb.y0)
        return cho

    def _push_down(self, jong: Outline, above: Outline) -> Outline:
        """받침이 윗부분과 겹치면 아래로 내리고, 칸을 넘으면 납작하게."""
        jb, ab = bounds(jong), bounds(above)
        if jb is None or ab is None:
            return jong
        top = ab.y1 + self.gap
        if jb.y0 >= top:
            return jong
        room = self.lib.cell * 0.98 - top
        if room <= jb.h * 0.5:
            return jong
        k = min(1.0, room / jb.h)
        squeezed = scale_about(jong, 1.0, k, jb.x0, jb.y0)
        return affine(squeezed, 1, 1, 0, top - jb.y0)


def median_height(outlines: list[Outline]) -> float:
    hs = [b.h for b in map(bounds, outlines) if b]
    return median(hs) if hs else 1.0
