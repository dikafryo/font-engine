"""기준 배치 자료 만들기: Noto Sans KR Thin 의 11,172자를 초·중·종성으로 나눠 fontmaker/data/reference.npz 에 저장.

사용: python -m scripts.make_reference <NotoSansKR[wght].ttf>
  글꼴: https://github.com/google/fonts/raw/main/ofl/notosanskr/NotoSansKR%5Bwght%5D.ttf (OFL)

나누는 법 — Thin 굵기는 자모끼리 닿지 않으므로 겹치는 윤곽선끼리 묶은 덩어리가 곧 자모(또는 그 일부)다.
  ① 윤곽선 모양(시작점 기준 상대좌표)이 똑같이 나오는 음절들에서 어느 자모가 늘 같은지 투표,
  ② 역할별 위치 사전확률(compose.DEFAULT_BOXES), ③ 닫힌 고리(ㅇㅁㅂㅎ…)는 모음이 될 수 없음,
  을 더해 덩어리→역할 배정을 전수 탐색(필요한 역할은 모두 채움).
  그래도 이상하거나(배치 상식 위반) 비슷한 글자들의 중앙값 상자에서 크게 벗어나면 — 기준 글꼴이 두 자모를
  한 윤곽선으로 합친 경우(곽·계 등) — 비슷한 글자들의 중앙값으로 메운다.

저장 내용
  boxes  float32 (11172, 3, 4)  역할(cho·jung·jong)별 상자 x0,y0,x1,y1 — 글자 칸(1000 단위, 위=0) 비율. 없는 받침은 nan
"""
from __future__ import annotations

import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont

from fontmaker.compose import DEFAULT_BOXES
from fontmaker.hangul import ROLES, all_syllables, decompose
from fontmaker.reference import ASCENT, EM, PATH

RES = 200  # 덩어리 판정용 래스터 해상도


class FlatPen(BasePen):
    """곡선을 잘게 꺾은선으로."""

    def __init__(self) -> None:
        super().__init__(None)
        self.polys: list[np.ndarray] = []
        self.cur: list[tuple[float, float]] = []

    def _moveTo(self, p):
        self.cur = [p]

    def _lineTo(self, p):
        self.cur.append(p)

    def _curveToOne(self, a, b, c):
        p0 = self.cur[-1]
        for t in np.linspace(0.1, 1, 10):
            m = 1 - t
            self.cur.append(tuple(m ** 3 * p0[i] + 3 * m * m * t * a[i] + 3 * m * t * t * b[i] + t ** 3 * c[i] for i in (0, 1)))

    def _qCurveToOne(self, a, b):
        p0 = self.cur[-1]
        for t in np.linspace(0.125, 1, 8):
            m = 1 - t
            self.cur.append(tuple(m * m * p0[i] + 2 * m * t * a[i] + t * t * b[i] for i in (0, 1)))

    def _closePath(self):
        self.polys.append(np.array(self.cur, float))
        self.cur = []

    _endPath = _closePath


def split_contours(rec: RecordingPen) -> list[list]:
    out, cur = [], []
    for op, args in rec.value:
        cur.append((op, args))
        if op in ("closePath", "endPath"):
            out.append(cur)
            cur = []
    return out


def signature(contour: list) -> tuple:
    pts = [p for _, args in contour for p in args]
    x0, y0 = pts[0]
    return tuple((round(x - x0), round(y - y0)) for x, y in pts)


def flatten(contour: list) -> np.ndarray:
    pen = FlatPen()
    for op, args in contour:
        getattr(pen, op)(*args)
    p = pen.polys[0]
    return np.c_[p[:, 0] / EM, (ASCENT - p[:, 1]) / EM]  # 칸 비율, 위=0


def raster(polys: list[np.ndarray], size: int) -> np.ndarray:
    """0 이 아닌 감김 수 규칙으로 채우기 (겹치는 획·ㅇ 구멍 모두 바르게)."""
    wind = np.zeros((size, size), np.int32)
    for p in polys:
        m = np.zeros((size, size), np.uint8)
        cv2.fillPoly(m, [np.round(p * size * 8).astype(np.int32)], 1, shift=3)
        area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
        wind += int(np.sign(area)) * m.astype(np.int32)
    return wind != 0


def inside(a: np.ndarray, b: np.ndarray) -> bool:
    """a 가 거의 b 안에 든다 (ㅇ 의 안쪽 윤곽 등)."""
    return (a & ~b).sum() < 0.05 * a.sum()


def groups_of(masks: list[np.ndarray], touching: bool) -> list[list[int]]:
    """touching=True: 겹치는 윤곽선끼리 묶음. False: 구멍(감싸인 윤곽선)만 바깥 윤곽선에 붙임."""
    parent = list(range(len(masks)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in itertools.combinations(range(len(masks)), 2):
        if (masks[i] & masks[j]).any() if touching else inside(masks[i], masks[j]) or inside(masks[j], masks[i]):
            parent[find(i)] = find(j)
    out: dict[int, list[int]] = defaultdict(list)
    for i in range(len(masks)):
        out[find(i)].append(i)
    return list(out.values())


UP_TICK = {8, 9, 10, 11, 12}  # ㅗㅘㅙㅚㅛ — 가로획 위로 꼭지가 선다
WIDE_JUNG = {1, 3, 4, 5, 6, 7}  # ㅐㅒㅓㅔㅕㅖ — 세로획 왼쪽으로 꼭지·둘째 획이 있어 중성 자리가 넓다


def prior(box: np.ndarray, syl, role: str) -> float:
    frac = DEFAULT_BOXES.get((role, syl.type))
    if frac is None:
        return -5.0
    if role == "jung" and syl.jung in WIDE_JUNG:
        frac = (0.4, *frac[1:])
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    dx = max(frac[0] - cx, 0, cx - frac[2])
    dy = max(frac[1] - cy, 0, cy - frac[3])
    return -8 * (dx + dy)


def label(ch: str, contours: list, where: dict, prior_weight: float = 1.0, touching: bool = True) -> tuple[list[str], float]:
    """(윤곽선별 역할, 배정된 덩어리 중 가장 나쁜 위치 점수)"""
    syl = decompose(ch)
    polys = [flatten(c) for c in contours]
    masks = [raster([p], RES) for p in polys]
    roles = [r for r in ROLES if r != "jong" or syl.jong]
    groups = groups_of(masks, touching)
    if len(groups) < len(roles):
        groups = groups_of(masks, False)
    scores, priors = [], []
    for g in groups:
        s = np.zeros(3)
        for i in g:
            seen = where[signature(contours[i])]  # (n, 3) 그 모양이 나오는 음절들의 초·중·종
            s += math.log1p(len(seen)) * (seen == [syl.cho, syl.jung, syl.jong]).mean(0)
        pts = np.vstack([polys[i] for i in g])
        box = np.r_[pts.min(0), pts.max(0)]
        pr = np.array([prior(box, syl, r) for r in ROLES])
        s += prior_weight * pr
        priors.append(pr)
        if any(inside(masks[j], masks[i]) for i in g for j in g if i != j):
            s[1] -= 100  # 닫힌 고리(ㅇㅁㅂㅎ…)는 모음이 아니다
        scores.append(s)
    best = assign_roles([[sc[ROLES.index(r)] for r in roles] for sc in scores], len(roles))
    out = [""] * len(contours)
    for k, g in enumerate(groups):
        for i in g:
            out[i] = roles[best[k]]
    worst = min(priors[k][ROLES.index(roles[a])] for k, a in enumerate(best))
    return out, worst


def assign_roles(score: list[list[float]], n_roles: int) -> list[int]:
    """덩어리마다 점수 최대 역할을 주고, 빈 역할이 있으면 가장 덜 손해 보는 덩어리를 옮겨 채운다.
    (점수가 덩어리별로 독립이라 전수 탐색과 거의 같은 답을 훨씬 빨리 낸다 — 윤곽선이 많으면 3^n 이 폭발한다.)"""
    pick = [int(np.argmax(s)) for s in score]
    for _ in range(n_roles):
        missing = [r for r in range(n_roles) if r not in pick]
        if not missing:
            break
        r = missing[0]
        movable = [k for k in range(len(pick)) if pick.count(pick[k]) > 1]
        if not movable:
            break
        k = min(movable, key=lambda k: score[k][pick[k]] - score[k][r])
        pick[k] = r
    return pick


def suspicious(syl, b: np.ndarray) -> bool:
    """배치가 상식에서 벗어난 글자 (사람이 확인할 목록)."""
    cho, jung, jong = b
    if min(cho[2] - cho[0], cho[3] - cho[1]) < 0.12:  # 자음이 획 하나만 남은 경우 (관 의 ㄱ 이 윗줄만 남는 등)
        return True
    if syl.jong and min(jong[2] - jong[0], jong[3] - jong[1]) < 0.12:
        return True
    if syl.jung in WIDE_JUNG and jung[2] - jung[0] < 0.1:
        return True
    if syl.jung in UP_TICK and cho[3] > jung[1] + 0.05:  # ㅗ 꼭지가 초성에 붙어 간 경우 (고·코 등에서 윤곽선이 합쳐짐)
        return True
    if syl.vclass in "HM" and jung[2] - jung[0] < 0.4:  # 가로획(ㅗㅜㅡ…)이 빠진 중성 (ㅟ+받침에서 ㅜ 가 받침으로 가는 등)
        return True
    if syl.vclass == "V" and (cho[0] + cho[2]) / 2 > (jung[0] + jung[2]) / 2:
        return True
    if syl.vclass == "H" and (cho[1] + cho[3]) / 2 > (jung[1] + jung[3]) / 2:
        return True
    return bool(syl.jong) and (jong[1] + jong[3]) / 2 < (cho[1] + cho[3]) / 2


def outliers(boxes: np.ndarray, limit: float = 0.15) -> set[int]:
    """비슷한 글자들의 중앙값 상자에서 한 변이라도 limit(칸 비율) 넘게 벗어난 글자.
    중성·종성은 (모음, 받침)이 같고 초성만 다른 글자끼리, 초성은 (초성, 모음, 받침 유무)가 같은 글자끼리 견준다."""
    syls = [decompose(chr(0xAC00 + i)) for i in range(len(boxes))]
    out = set()
    for k, role in enumerate(ROLES):
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, s in enumerate(syls):
            if role != "jong" or s.jong:
                groups[(s.cho, s.jung, bool(s.jong)) if role == "cho" else (s.jung, s.jong)].append(i)
        for idx in groups.values():
            med = np.median(boxes[idx, k], axis=0)
            out |= {i for i in idx if np.abs(boxes[i, k] - med).max() > limit}
        wide: dict[tuple, list[int]] = defaultdict(list)  # 무리 전체가 틀린 경우: (자모, 위치 유형) 단위로 한 번 더
        for i, s in enumerate(syls):
            if role != "jong" or s.jong:
                wide[(s.jamo(role), s.type)].append(i)
        for idx in wide.values():
            med = np.median(boxes[idx, k], axis=0)
            out |= {i for i in idx if np.abs(boxes[i, k] - med).max() > limit * 1.4}
            size = med[2:] - med[:2]  # 폭·높이가 무리의 절반도 안 되면 자모 일부만 잡힌 것 (제 의 ㅔ 등)
            out |= {i for i in idx if ((boxes[i, k, 2:] - boxes[i, k, :2]) < 0.5 * size).any()}
    return out


def impute(boxes: np.ndarray, bad: set[int]) -> None:
    """끝내 못 나눈 글자(기준 글꼴이 두 자모를 한 윤곽선으로 합친 경우)는 비슷한 글자들의 상자 중앙값으로 메운다.
    같은 자모 + 같은 모음 + 같은 받침 → 받침 유무 → 위치 유형 → 모음 방향 → 아무 글자 순으로 넓힌다."""
    syls = [decompose(chr(0xAC00 + i)) for i in range(len(boxes))]
    good = [i for i in range(len(boxes)) if i not in bad]
    for i in bad:
        s = syls[i]
        for k, role in enumerate(ROLES):
            if role == "jong" and not s.jong:
                continue
            for same in (lambda t: t.jung == s.jung and t.jong == s.jong,
                         lambda t: t.jung == s.jung and bool(t.jong) == bool(s.jong),
                         lambda t: t.type == s.type,
                         lambda t: t.vclass == s.vclass,
                         lambda t: True):
                near = [boxes[j, k] for j in good if syls[j].jamo(role) == s.jamo(role) and same(syls[j])]
                if near:
                    boxes[i, k] = np.median(near, axis=0)
                    break


def main(vf: Path) -> None:
    font = TTFont(str(vf))
    glyphs = font.getGlyphSet(location={"wght": 100})
    cmap = font.getBestCmap()
    data = {}
    for ch in all_syllables():
        rec = RecordingPen()
        glyphs[cmap[ord(ch)]].draw(rec)
        data[ch] = split_contours(rec)
    seen: dict[tuple, list[tuple[int, int, int]]] = defaultdict(list)
    for ch, cs in data.items():
        syl = decompose(ch)
        for c in cs:
            seen[signature(c)].append((syl.cho, syl.jung, syl.jong))
    where = {k: np.array(v) for k, v in seen.items()}

    boxes = np.full((len(data), 3, 4), np.nan, np.float32)
    odd: list[str] = []
    for n, (ch, cs) in enumerate(data.items()):
        polys = [flatten(c) for c in cs]
        tries = []
        # 투표가 엉뚱하면 위치 사전확률을 키워 다시, 그래도 안 되면 맞닿은 자모를 윤곽선 단위로 풀어서 다시
        for touching, weight in ((True, 1), (True, 6), (False, 1), (False, 6), (False, 30)):
            labs, worst = label(ch, cs, where, weight, touching)
            b = np.full((3, 4), np.nan)
            for k, role in enumerate(ROLES):
                mine = [p for p, r in zip(polys, labs) if r == role]
                if mine:
                    pts = np.vstack(mine)
                    b[k] = np.r_[pts.min(0), pts.max(0)]
            ok = not suspicious(decompose(ch), b)
            tries.append((ok, worst > -1.0, labs, b))
            if ok and worst > -1.0:
                break
        sane = [t for t in tries if t[0]]
        _, _, labs, b = sane[0] if sane else tries[-1]
        if not sane:
            odd.append(ch)
        boxes[n] = b
    bad = {ord(ch) - 0xAC00 for ch in odd} | outliers(boxes)
    impute(boxes, bad)
    odd = [chr(0xAC00 + i) for i in sorted(bad)]
    PATH.parent.mkdir(exist_ok=True)
    np.savez_compressed(PATH, boxes=boxes)
    print(f"{PATH} 저장. 중앙값으로 메운 글자 {len(odd)}개")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("사용: python -m scripts.make_reference <NotoSansKR[wght].ttf>")
    main(Path(sys.argv[1]))
