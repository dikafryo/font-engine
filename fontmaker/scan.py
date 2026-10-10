"""스캔·사진 → 정렬된 쪽 이미지 → 칸별 잉크 비트맵.

정렬 후 좌표계: PX_PER_MM 배율의 A4 (좌상단 원점).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import layout as L

PX_PER_MM = 12.0
INK_LEVEL = 0.58  # 배경 대비 밝기 비. 연파랑 안내선(~0.75)은 떨어지고 검정 펜(<0.4)은 남는다
MAX_SIDE = 4200  # 너무 큰 사진은 줄여서 처리


class ScanError(Exception):
    pass


@dataclass(frozen=True)
class PageImage:
    number: int
    norm: np.ndarray  # float32, 1.0=종이, 낮을수록 잉크
    source: str


@dataclass(frozen=True)
class CellInk:
    cell: L.Cell
    mask: np.ndarray  # bool, True=잉크. 칸 내부 (BOX mm × PX_PER_MM)
    empty: bool


def load_images(path: Path) -> list[np.ndarray]:
    """jpg/png 는 한 장, pdf 는 쪽마다 한 장. cv2.imread 가 EXIF 방향을 반영하고, 나머지 회전은 쪽 코드로 판정한다."""
    if path.suffix.lower() == ".pdf":
        import pymupdf

        out = []
        with pymupdf.open(path) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
                out.append(cv2.cvtColor(np.ascontiguousarray(arr[:, :, :3]), cv2.COLOR_RGB2BGR))
        return out
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ScanError(f"{path.name}: 이미지를 열 수 없습니다")
    return [img]


def downscale(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    k = MAX_SIDE / max(h, w)
    return cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else img


def normalize(gray: np.ndarray) -> np.ndarray:
    """조명 얼룩 제거: 종이 밝기(큰 커널 닫힘)로 나눈다."""
    k = max(15, (min(gray.shape) // 40) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    return np.clip(gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1), 0, 1)


def find_marks(norm: np.ndarray) -> np.ndarray:
    """모서리 검은 네모 4개의 중심. 반환 순서는 이미지 기준 좌상·우상·우하·좌하."""
    ink = (norm < 0.5).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(ink, connectivity=8)
    min_area = norm.size * 2e-5
    cands = []
    for i in range(1, n):
        _, _, w, h, area = stats[i]
        if area < min_area or not 0.6 < w / h < 1.6 or area / (w * h) < 0.8:
            continue
        cands.append((area, cents[i]))
    if len(cands) < 4:
        raise ScanError("모서리 검은 네모 4개를 찾지 못했습니다")
    top_area = max(a for a, _ in cands)
    pts = np.array([c for a, c in cands if a > top_area * 0.35])
    if len(pts) < 4:
        raise ScanError("모서리 검은 네모 4개를 찾지 못했습니다")
    s, d = pts.sum(1), pts[:, 0] - pts[:, 1]
    quad = np.array([pts[s.argmin()], pts[d.argmax()], pts[s.argmax()], pts[d.argmin()]], np.float32)
    if len({tuple(p) for p in quad.tolist()}) < 4 or cv2.contourArea(quad) < norm.size * 0.15:
        raise ScanError("모서리 네모의 배치가 이상합니다 (종이 전체가 나오게 찍어 주세요)")
    return quad


def warp(norm: np.ndarray, quad: np.ndarray, rot: int) -> np.ndarray:
    dst = np.array([[x * PX_PER_MM, y * PX_PER_MM] for x, y in L.MARK_CENTERS], np.float32)
    src = np.roll(quad, -rot, axis=0)
    m = cv2.getPerspectiveTransform(src, dst)
    size = (int(L.PAGE_W * PX_PER_MM), int(L.PAGE_H * PX_PER_MM))
    return cv2.warpPerspective(norm, m, size, flags=cv2.INTER_LINEAR, borderValue=1.0)


def read_code(page: np.ndarray) -> int | None:
    bits = []
    inset = L.CODE_SQ * 0.25
    for x, y in L.code_squares():
        x0, y0 = int((x + inset) * PX_PER_MM), int((y + inset) * PX_PER_MM)
        x1, y1 = int((x + L.CODE_SQ - inset) * PX_PER_MM), int((y + L.CODE_SQ - inset) * PX_PER_MM)
        bits.append(int(page[y0:y1, x0:x1].mean() < 0.5))
    return L.decode_bits(bits)


def align(img: np.ndarray, source: str) -> PageImage:
    gray = cv2.cvtColor(downscale(img), cv2.COLOR_BGR2GRAY)
    norm = normalize(gray)
    quad = find_marks(norm)
    for rot in range(4):
        page = warp(norm, quad, rot)
        number = read_code(page)
        if number is not None:
            return PageImage(number, page, source)
    raise ScanError("쪽 코드(아래 가운데 작은 네모)를 읽지 못했습니다 — font.sw4u.kr 양식이 맞는지 확인해 주세요")


def extract_cells(page: PageImage) -> list[CellInk]:
    spec = L.pages()[page.number - 1]
    inset = 0.7  # 칸 테두리 안쪽으로 (mm)
    size = int((L.BOX - 2 * inset) * PX_PER_MM)
    out = []
    for cell in spec.cells:
        x0, y0 = int((cell.x + inset) * PX_PER_MM), int((cell.y + inset) * PX_PER_MM)
        mask = clean(page.norm[y0:y0 + size, x0:x0 + size] < INK_LEVEL)
        out.append(CellInk(cell, mask, int(mask.sum()) < PX_PER_MM ** 2 * 1.5))
    return out


def clean(mask: np.ndarray) -> np.ndarray:
    """작은 티끌 제거 + 획 가장자리 정리."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= PX_PER_MM ** 2 * 0.12
    return keep[labels]
