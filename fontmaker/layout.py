"""양식 정의 — 칸 좌표의 단일 출처. sheet.py(인쇄)와 scan.py(인식)가 함께 쓴다.

좌표는 mm, 좌상단 원점.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache

PAGE_W, PAGE_H = 210.0, 297.0
SHEET_KIND = 1  # 쪽 코드에 박히는 양식 종류. 칸 구성을 바꾸면 올린다.
VERSION = "v2"

MARK = 5.0
MARK_CENTERS: tuple[tuple[float, float], ...] = ((12.0, 12.0), (198.0, 12.0), (198.0, 285.0), (12.0, 285.0))

# 쪽 코드: 아래 가운데 8칸 = [1, kind 3비트, page 3비트, 1]
CODE_BITS, CODE_SQ, CODE_GAP, CODE_Y = 8, 3.0, 1.5, 285.0
CODE_X0 = (PAGE_W - (CODE_BITS * CODE_SQ + (CODE_BITS - 1) * CODE_GAP)) / 2

COLS, CELL_W, BOX, LABEL_H, ROW_GAP = 9, 19.0, 17.0, 4.0, 2.0
GRID_X0 = (PAGE_W - COLS * CELL_W) / 2 + (CELL_W - BOX) / 2
ROW_H = LABEL_H + BOX + ROW_GAP
GRID_BOTTOM = 279.0
LATIN_BASELINE = 0.70  # 영문 칸 기준선 (칸 높이 대비, 위에서부터)
LATIN_CAP = 0.22

# 위치 유형(세로·가로·복합 모음 × 받침 유무)마다 초·중·종성이 고루 나오도록 고른 글자. 중복은 첫 번째만 남긴다.
HANGUL_GROUPS: tuple[str, ...] = (
    "가까나다따라마바빠사싸아자짜차카타파하",
    "고꼬노도또로모보뽀소쏘오조쪼초코토포호",
    "애야얘어에여예와왜외요우워웨위유으의이",
    "거너더러머버서저처커터퍼허",
    "구누두루무부수주추쿠투푸후",
    "각간갇갈감갑갓강갖갗같갚갛갔",
    "난단람밥삼잡찬칸탄판한",
    "곡논돋롤몸봅솟옹좇촌콕톤폭홍",
    "국눈둘문북숨운준춤",
    "관권원왕횡될괜과뒤쥐화",
    "닭삶넓없앉많읽핥밖엌넋읊싫곬",
    "는을은있것그들서지기리정시일상대인전장제",
)
LATIN = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    ".,?!'\"()-~:;/&@#%+="
)
SENTENCES: tuple[str, ...] = (
    "다람쥐 헌 쳇바퀴에 타고파",
    "키스의 고유조건은 입술끼리 만나야 하고 특별한 기술은 필요치 않다",
    "오늘 날씨가 참 맑아서 산책하기 좋았습니다.",
    "2026년 10월 10일, 커피 두 잔과 빵 3개를 샀다!",
    "The quick brown fox jumps over the lazy dog.",
)


@dataclass(frozen=True)
class Cell:
    char: str
    kind: str  # "hangul" | "latin"
    page: int
    index: int  # 전체 일련번호 (1부터)
    x: float
    y: float
    size: float = BOX


@dataclass(frozen=True)
class Page:
    number: int
    title: str
    grid_top: float
    cells: tuple[Cell, ...]


def hangul_chars() -> list[str]:
    seen: dict[str, None] = {}
    for group in HANGUL_GROUPS:
        for ch in group:
            seen.setdefault(ch, None)
    return list(seen)


def _rows(top: float) -> int:
    return int((GRID_BOTTOM - top + ROW_GAP) // ROW_H)


def _place(chars: list[str], kind: str, page: int, top: float, start: int) -> tuple[Cell, ...]:
    cells = []
    for i, ch in enumerate(chars):
        r, c = divmod(i, COLS)
        cells.append(Cell(ch, kind, page, start + i, round(GRID_X0 + c * CELL_W, 3), round(top + r * ROW_H + LABEL_H, 3)))
    return tuple(cells)


FIRST_GRID_TOP = 47.0  # 1쪽은 안내문 아래부터
OTHER_GRID_TOP = 25.0


@cache
def pages() -> tuple[Page, ...]:
    chars = hangul_chars()
    cap1 = _rows(FIRST_GRID_TOP) * COLS
    cap2 = _rows(OTHER_GRID_TOP) * COLS
    if len(chars) > cap1 + cap2:
        raise ValueError(f"한글 칸이 {len(chars) - cap1 - cap2}개 모자랍니다")
    if len(LATIN) > cap2:
        raise ValueError("영문 칸이 모자랍니다")
    p1 = _place(chars[:cap1], "hangul", 1, FIRST_GRID_TOP, 1)
    p2 = _place(chars[cap1:], "hangul", 2, OTHER_GRID_TOP, len(p1) + 1)
    p3 = _place(list(LATIN), "latin", 3, OTHER_GRID_TOP, len(p1) + len(p2) + 1)
    return (
        Page(1, "한글 (1)", FIRST_GRID_TOP, p1),
        Page(2, "한글 (2)", OTHER_GRID_TOP, p2),
        Page(3, "영문·숫자·기호", OTHER_GRID_TOP, p3),
        Page(4, "문장 (선택)", OTHER_GRID_TOP, ()),
    )


def code_bits(page: int) -> list[int]:
    kind = [(SHEET_KIND >> b) & 1 for b in (2, 1, 0)]
    num = [(page >> b) & 1 for b in (2, 1, 0)]
    return [1, *kind, *num, 1]


def decode_bits(bits: list[int]) -> int | None:
    """쪽 코드 → 쪽 번호. 양식 종류가 다르거나 깨졌으면 None."""
    if len(bits) != CODE_BITS or bits[0] != 1 or bits[-1] != 1:
        return None
    kind = bits[1] * 4 + bits[2] * 2 + bits[3]
    page = bits[4] * 4 + bits[5] * 2 + bits[6]
    if kind != SHEET_KIND or not 1 <= page <= len(pages()):
        return None
    return page


def code_squares() -> list[tuple[float, float]]:
    """쪽 코드 각 칸의 좌상단 (mm)."""
    return [(CODE_X0 + i * (CODE_SQ + CODE_GAP), CODE_Y - CODE_SQ / 2) for i in range(CODE_BITS)]
