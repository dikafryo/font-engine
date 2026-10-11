"""직접 쓴 음절에서 뽑은 부품으로 나머지 음절을 조합한다.

부품은 (역할, 자모, 위치유형) 별로 모은다. 대상 음절의 자모 하나를 채울 때
① 같은 자모 부품 중 기준 배치(reference)에서 원래 음절의 상자가 대상 음절의 상자와 가장 비슷한 것을 고르고,
② 그 부품의 손글씨 상자에 '기준 글꼴에서 두 음절 사이 그 자모가 움직인 만큼'(변 단위, 손글씨 크기로 환산)을 더한 곳에 놓는다.
쓴 사람의 버릇(크기·위치)은 살리고, 문맥(받침 유무·겹받침·모음 모양)에 따른 변화만 기준 글꼴을 따른다.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from math import log

import numpy as np

from . import reference
from .hangul import CHO, CHO_TO_JONG, JONG, JONG_TO_CHO, Syllable, decompose
from .vector import Bounds, Outline, bounds, fit, trace

TYPES = ("V0", "V1", "H0", "H1", "M0", "M1")
SIBLING = 0.5  # 원래 음절이 대상과 다른 자모도 같으면 가산점 — 초성·중성 맞물림을 한 음절에서 함께 가져온다 (평가 +0.007)
DENOISE = 0.5  # 부품을 놓을 때 그 샘플 한 번의 어긋남 대신 그 자모의 평소 버릇(후보들의 중앙값)을 섞는 비율
# (실제 손글씨 교차 검증 IoU 0.294 → 0.313, 흔들림 합성 0.566 → 0.599)
FOLLOW = 0.5  # 기준 글꼴의 문맥 이동을 얼마나 따를지 (1=그대로). 손글씨 6종 평가에서 0.5 가 1.0 보다 나았다
PAIRS = {"ㄲ": "ㄱㄱ", "ㄸ": "ㄷㄷ", "ㅃ": "ㅂㅂ", "ㅆ": "ㅅㅅ", "ㅉ": "ㅈㅈ", "ㄳ": "ㄱㅅ", "ㄵ": "ㄴㅈ", "ㄶ": "ㄴㅎ",
         "ㄺ": "ㄹㄱ", "ㄻ": "ㄹㅁ", "ㄼ": "ㄹㅂ", "ㄽ": "ㄹㅅ", "ㄾ": "ㄹㅌ", "ㄿ": "ㄹㅍ", "ㅀ": "ㄹㅎ", "ㅄ": "ㅂㅅ"}

# 역할별 대략의 자리 (칸 비율) — scripts/make_reference.py 가 기준 글꼴을 나눌 때 위치 사전확률로 쓴다
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
    box: Bounds  # 손글씨 칸 픽셀 상자
    type: str
    source: str  # 부품을 뽑은 음절
    role: str  # 원래 역할 (초성 자리 부품을 받침에 빌려 쓸 때 기준 상자를 찾는 데 쓴다)


def mask_bounds(mask: np.ndarray) -> Bounds:
    ys, xs = np.nonzero(mask)
    return Bounds(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


class Library:
    def __init__(self, cell_px: int) -> None:
        self.cell = cell_px
        self.parts: dict[tuple[str, int, str], list[Part]] = defaultdict(list)

    def add(self, syl: Syllable, masks: dict[str, np.ndarray]) -> None:
        for role, mask in masks.items():
            part = Part(trace(mask), mask_bounds(mask), syl.type, syl.char, role)
            self.parts[(role, syl.jamo(role), syl.type)].append(part)

    def of(self, role: str, jamo: int) -> list[Part]:
        return [p for t in TYPES for p in self.parts.get((role, jamo, t), [])]

    def scale(self) -> tuple[float, float]:
        """기준 배치 1(칸 비율) = 손글씨 몇 픽셀인지 (가로, 세로). 부품 상자 크기 비의 중앙값."""
        kx, ky = [], []
        for parts in self.parts.values():
            for p in parts:
                b = reference.box(p.source, p.role)
                kx.append(p.box.w / max(b[2] - b[0], 0.02))
                ky.append(p.box.h / max(b[3] - b[1], 0.02))
        return (median(kx), median(ky)) if kx else (self.cell, self.cell)


class Composer:
    def __init__(self, lib: Library) -> None:
        self.lib = lib
        self.kx, self.ky = lib.scale()

    def _candidates(self, role: str, jamo: int) -> list[Part]:
        """같은 역할 부품, 없으면 초성↔받침 같은 꼴 자음."""
        own = self.lib.of(role, jamo)
        if own:
            return own
        alt = {"cho": ("jong", CHO_TO_JONG), "jong": ("cho", JONG_TO_CHO)}.get(role)
        if alt and jamo in alt[1]:
            return self.lib.of(alt[0], alt[1][jamo])
        return []

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        """기준 상자 둘이 얼마나 다른가 (크기 비 로그 + 중심 거리)."""
        wa, ha, wb, hb = a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]
        size = abs(log(max(wa, 0.02) / max(wb, 0.02))) + abs(log(max(ha, 0.02) / max(hb, 0.02)))
        move = abs((a[0] + a[2]) - (b[0] + b[2])) + abs((a[1] + a[3]) - (b[1] + b[3]))
        return size + 2 * move

    def _place(self, role: str, jamo: int, target: str) -> Outline | None:
        want = reference.box(target, role)
        parts = self._candidates(role, jamo)
        if not parts or want is None:
            return self._place_pair(role, jamo, target, want)
        goal = decompose(target)

        def cost(p: Part) -> float:
            src = decompose(p.source)
            shared = sum(src.jamo(r) == goal.jamo(r) for r in ("cho", "jung", "jong") if r != role)
            return self._distance(reference.box(p.source, p.role), want) - SIBLING * shared

        part = min(parts, key=cost)
        usual = np.median(np.array([self._residual(p) for p in parts]), axis=0)
        offset = (1 - DENOISE) * self._residual(part) + DENOISE * usual
        return fit(part.outline, part.box, self._moved(part.box, reference.box(part.source, part.role), want, offset))

    def _to_ref(self, box: Bounds) -> np.ndarray:
        ox, oy = self._origin()
        return np.array([(box.x0 - ox) / self.kx, (box.y0 - oy) / self.ky, (box.x1 - ox) / self.kx, (box.y1 - oy) / self.ky])

    def _residual(self, p: Part) -> np.ndarray:
        """부품이 기준 상자에서 어긋난 정도 (기준 좌표, 4변) — 쓴 사람의 버릇 + 그 글자 한 번의 흔들림."""
        return self._to_ref(p.box) - reference.box(p.source, p.role)

    def _moved(self, box: Bounds, ref_from: np.ndarray, ref_to: np.ndarray, offset: np.ndarray | None = None) -> Bounds:
        """손글씨 상자 box 를, 기준 상자 ref_from → ref_to 의 비례 변환으로 옮긴다 (기준 좌표에서 계산).
        offset: 기준 상자에서의 어긋남(없으면 box 자신의 것). 쓴 사람의 버릇은 비율로 유지된다."""
        goal = ref_from + FOLLOW * (ref_to - ref_from)
        ox, oy = self._origin()
        u = ref_from + (self._to_ref(box) - ref_from if offset is None else offset)
        size_from = np.maximum(ref_from[2:] - ref_from[:2], 0.02)
        k = np.clip((goal[2:] - goal[:2]) / size_from, 0.35, 3.0)  # 지나친 찌그러짐 방지
        lo = goal[:2] + (u[:2] - ref_from[:2]) * k
        hi = goal[:2] + (u[2:] - ref_from[:2]) * k
        if (hi - lo).min() <= 0 and offset is not None:  # 평소 버릇을 섞었더니 뒤집히면 샘플 그대로
            return self._moved(box, ref_from, ref_to)
        return Bounds(ox + lo[0] * self.kx, oy + lo[1] * self.ky, ox + hi[0] * self.kx, oy + hi[1] * self.ky)

    def _place_pair(self, role: str, jamo: int, target: str, want: np.ndarray | None) -> Outline | None:
        """겹받침·쌍자음 샘플이 없으면 홑자음 둘을 대상 상자 왼쪽·오른쪽에 나란히."""
        names = CHO if role == "cho" else JONG
        pair = PAIRS.get(names[jamo])
        if not pair or want is None:
            return None
        dst = self._absolute(want)
        half = dst.w / 2
        out: Outline = []
        for i, single in enumerate(pair):
            parts = self._candidates(role, names.index(single))
            if not parts:
                return None
            aspect = half / max(dst.h, 1)
            part = min(parts, key=lambda p: abs(log(max(p.box.w, 1) / max(p.box.h, 1) / aspect)))
            x0 = dst.x0 + i * half
            out += fit(part.outline, part.box, Bounds(x0 + half * 0.04, dst.y0, x0 + half * 0.96, dst.y1))
        return out

    def _absolute(self, ref_box: np.ndarray) -> Bounds:
        """기준 상자 → 손글씨 칸 픽셀 (글자 전체 자리는 직접 쓴 글자들의 평균 자리에 맞춘다)."""
        ox, oy = self._origin()
        return Bounds(ox + ref_box[0] * self.kx, oy + ref_box[1] * self.ky, ox + ref_box[2] * self.kx, oy + ref_box[3] * self.ky)

    def _origin(self) -> tuple[float, float]:
        if not hasattr(self, "_org"):
            xs, ys = [], []
            for parts in self.lib.parts.values():
                for p in parts:
                    b = reference.box(p.source, p.role)
                    xs.append(p.box.x0 - b[0] * self.kx)
                    ys.append(p.box.y0 - b[1] * self.ky)
            self._org = (median(xs), median(ys)) if xs else (0.0, 0.0)
        return self._org

    def compose(self, syl: Syllable) -> Outline | None:
        ch = syl.char
        cho = self._place("cho", syl.cho, ch)
        jung = self._place("jung", syl.jung, ch)
        jong = self._place("jong", syl.jong, ch) if syl.jong else []
        if cho is None or jung is None or jong is None:
            return None
        return cho + jung + jong
