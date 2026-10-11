"""기준 배치 자료 (Noto Sans KR Thin 에서 뽑은 11,172자의 자모별 상자). 만드는 법은 scripts/make_reference.py.

좌표는 글자 칸 비율 (가로 0~1, 세로 0=위 ~ 1=아래). 손글씨 분리(hangul.segment)와 조합(compose)이 함께 쓴다.
"""
from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np

PATH = Path(__file__).parent / "data/reference.npz"
EM, ASCENT = 1000, 880  # 기준 글꼴 단위 → 칸 비율 환산
ROLE_INDEX = {"cho": 0, "jung": 1, "jong": 2}


@cache
def _load() -> np.ndarray:
    with np.load(PATH) as z:
        return z["boxes"]


def box(ch: str, role: str) -> np.ndarray | None:
    """음절 ch 에서 역할의 상자 [x0, y0, x1, y1] (칸 비율). 없는 받침이면 None."""
    b = _load()[ord(ch) - 0xAC00, ROLE_INDEX[role]]
    return None if np.isnan(b[0]) else b.astype(float)
