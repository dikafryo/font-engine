"""비트맵 → 베지어 외곽선(potrace), 그리고 외곽선 변환.

좌표는 칸 픽셀(y 아래로 증가). 폰트 단위 변환은 fontbuild.py 에서 한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import potrace

Point = tuple[float, float]
Segment = tuple  # ("L", p) 또는 ("C", c1, c2, p)


@dataclass(frozen=True)
class Contour:
    start: Point
    segs: tuple[Segment, ...]


Outline = list[Contour]


@dataclass(frozen=True)
class Bounds:
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


def trace(mask: np.ndarray) -> Outline:
    """True=잉크 비트맵을 외곽선으로. potracer 는 0(False)을 잉크로 보므로 뒤집어 넘긴다."""
    path = potrace.Bitmap(~mask).trace(turdsize=4, alphamax=1.0, opticurve=True, opttolerance=0.3)
    out = []
    for curve in path:
        segs: list[Segment] = []
        for s in curve.segments:
            if s.is_corner:
                segs.append(("L", (s.c.x, s.c.y)))
                segs.append(("L", (s.end_point.x, s.end_point.y)))
            else:
                segs.append(("C", (s.c1.x, s.c1.y), (s.c2.x, s.c2.y), (s.end_point.x, s.end_point.y)))
        out.append(Contour((curve.start_point.x, curve.start_point.y), tuple(segs)))
    return out


def points(outline: Outline):
    for c in outline:
        yield c.start
        for s in c.segs:
            yield from s[1:]


def bounds(outline: Outline) -> Bounds | None:
    """제어점 포함 경계 — 곡선은 제어점 볼록껍질 안에 있으므로 약간 넉넉한 근사."""
    pts = list(points(outline))
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return Bounds(min(xs), min(ys), max(xs), max(ys))


def affine(outline: Outline, sx: float, sy: float, tx: float, ty: float) -> Outline:
    """x' = sx·x + tx, y' = sy·y + ty"""
    def f(p: Point) -> Point:
        return (p[0] * sx + tx, p[1] * sy + ty)

    return [Contour(f(c.start), tuple((s[0], *map(f, s[1:])) for s in c.segs)) for c in outline]


def fit(outline: Outline, src: Bounds, dst: Bounds) -> Outline:
    """src 상자를 dst 상자로 옮기는 변환을 적용."""
    sx = dst.w / src.w if src.w else 1.0
    sy = dst.h / src.h if src.h else 1.0
    return affine(outline, sx, sy, dst.x0 - src.x0 * sx, dst.y0 - src.y0 * sy)


def scale_about(outline: Outline, kx: float, ky: float, ox: float, oy: float) -> Outline:
    return affine(outline, kx, ky, ox - ox * kx, oy - oy * ky)
