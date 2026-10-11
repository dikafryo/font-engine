"""양식 정의 — 칸 좌표의 단일 출처. sheet.py(인쇄)와 scan.py(인식)가 함께 쓴다.

좌표는 mm, 좌상단 원점.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache

PAGE_W, PAGE_H = 210.0, 297.0
SHEET_KIND = 2  # 쪽 코드에 박히는 양식 종류. 칸 구성을 바꾸면 올린다. (2 = v3, 한글 4쪽)
VERSION = "v3"

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
# 더 받는 글자 (v3): 위 기본 글자만으로 조합할 때 부품을 가장 멀리 옮겨야 하는 (자모 × 자리) 빈틈을
# 가장 많이 메우는 순서로 2,350자 안에서 고른 것(일상 문장에 나오는 글자 우선). 앞쪽일수록 효과가 크다.
# 실제 손글씨 검증에서 샘플이 늘수록 조합 정확도가 꾸준히 올랐다(35→140자: IoU 0.21→0.31).
EXTRA_HANGUL = (
    "깩뗌뼘큅썅쁑쭁폡쩟틤꿱눴겠쇘껸뢴쵯챙섈켭떡뵉쓺뜻댄넷뫈"
    "끓퍅잽쐈벤띌쬡렸명툿죈퓜꾐멜탰틱챦븃햄뎀떪쌤낏옌뼈묩켁"
    "쩝얜럇뱃렬칟듕뉠솥놓맺쩽꽂굵룡탓쉥켯셈덕퓰줆빽표휙릎콸"
    "쌓곌젊펏갠헤혐랭딪샨렀뷴웰굽앱님빰폐넌똥업싶큠범툴쾌렇"
    "췽깥몇휑퇸녜흖별캣걷몲늘뫄봐테끅땝훈팔냠힉줏째쉿퀄씀폣"
    "셉껙쩠케찾짙뇌뙤렉른룔쫓래꽤퇴뽑배녁빻젝켸놨쬐쀼솨푯섭"
    "흄숯뒹댁짓퐈읜셰흼털늦횻웁꿨밧갤뎐겪채싼런떼멀든헬겟볶"
    "롸쭙꿀쐐입컥젤짧웡림쎄붉푠민깜펭메개딱닳닦댈텅"
)
# 추가 양식(종류 3, 한글 4쪽): 이미 만든 폰트에 더 써서 올리는 글자. v3 387자를 다 쓴 뒤에도 남는 빈틈을
# 가장 많이 메우는 순서로 2,350자에서 고른 것(빈틈 비용 0.034 → 0.012).
ADDON_HANGUL = (
    "퀭통넹혈쥔몫싱꿋쑨끝큘췐땟걸능웩균큭튑함푿첫쮸펫헥삥깬"
    "냥넬렷훗촤크춥쪄휼뤽곗흡켑냘뚬템빌퓔솅썼띱뀔뱀찡몌쌩괩"
    "랄옰충탤둣편뺀퐝먁끄횅캘컴펼죡뭏근뷩얗쵸덞꼐쩐였뜀엶규"
    "꽃갱뺍느뚤쎌벗섰둡컁옇뤠닝횰핍륙땃깨엠뙨건윷형눼뇩뿐씜"
    "녘뭣셨뎄큔쥣딧협쏟읖줘햇촁귤된뺙읫뒈뱉뒝풀랫뎠휄뭬칡척"
    "틸췌퉜뱐욈쳇긔슉잿짱샌윙륨띠굳썹옅돎곱퐁경줴휵젖띳혭멨"
    "녑엑락끔뛸씔궤면륜붸딸늄김섧꿰섕뎁쉐숀퀘끗찜쌘쒜좋쳅뒬"
    "퉁쭉펠쨉톱롑씻넝냈엇흴왑휫펴텬뚜훑욜춈첩깸붤되속퉤짖쉠"
    "륩불썽웹셌성넘촉겹란드뼁쟎훽슘삠쫍싻룅친맬뚫밍떴멕넛퀴"
    "삔꼇쉈뿟끊쪘맏뾰늬얹출쳤법쟐첼꿜쇗댓트훨펀켤듸헉힁숌띄"
    "떽욕늴쿱텡믿왓쨌뫘낀찔올륫벋땠힌얇혓랩쨔닐뮬텀쟌흩넣롓"
    "봔뇟웝깡짭쭹앵훤꿍꼲륏쓰쟨뗍뺏훰항닻뭡헒언렐렵폄밞탔읗"
    "퓸옆백긺쯔빼빤뷜닉신굅겉즉쥠니룩휴치펍읔말켄쉭끌빛홑꺅"
    "짊꿸귀뻬붇할딜뜁뇹높솔캠랜얠헵펨젬뱍극륄듄띨갬넥틔뀝견"
    "삘훼쫀랖킥뱅끎덖쿡"
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


ADDON_KIND = 3  # 추가 양식의 쪽 코드 종류
HANGUL_PAGES = {SHEET_KIND: 4, ADDON_KIND: 4}  # 양식 종류 → 한글 쪽 수. v2(종류 1)는 2026-10-11 폐기
REQUIRED_HANGUL_PAGES = (1, 2)  # 본 양식(SHEET_KIND)의 필수 쪽


def hangul_chars(kind: int = SHEET_KIND) -> list[str]:
    """본 양식: 기본 글자(필수 1·2쪽) 다음에 더 받는 글자(권장 3·4쪽). 추가 양식: ADDON_HANGUL."""
    seen: dict[str, None] = {}
    for group in ((ADDON_HANGUL,) if kind == ADDON_KIND else (*HANGUL_GROUPS, EXTRA_HANGUL)):
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
def pages(kind: int = SHEET_KIND) -> tuple[Page, ...]:
    """양식 종류별 쪽 구성 (HANGUL_PAGES 에 있는 종류만 읽는다)."""
    chars = hangul_chars(kind)
    n_hangul = HANGUL_PAGES[kind]
    caps = [_rows(FIRST_GRID_TOP) * COLS] + [_rows(OTHER_GRID_TOP) * COLS] * (n_hangul - 1)
    if len(chars) > sum(caps):
        raise ValueError(f"한글 칸이 {len(chars) - sum(caps)}개 모자랍니다")
    if len(LATIN) > caps[1]:
        raise ValueError("영문 칸이 모자랍니다")
    addon = kind == ADDON_KIND
    out, start, at = [], 1, 0
    for n, cap in enumerate(caps, 1):
        cells = _place(chars[at:at + cap], "hangul", n, FIRST_GRID_TOP if n == 1 else OTHER_GRID_TOP, start)
        if addon:
            title = f"추가 양식 ({n})"
        else:
            title = f"한글 ({n})" if n <= len(REQUIRED_HANGUL_PAGES) else f"한글 ({n}) — 권장"
        out.append(Page(n, title, FIRST_GRID_TOP if n == 1 else OTHER_GRID_TOP, cells))
        start, at = start + len(cells), at + cap
    if addon:
        return tuple(out)
    latin = n_hangul + 1
    out.append(Page(latin, "영문·숫자·기호", OTHER_GRID_TOP, _place(list(LATIN), "latin", latin, OTHER_GRID_TOP, start)))
    out.append(Page(latin + 1, "문장 (선택)", OTHER_GRID_TOP, ()))
    return tuple(out)


def code_bits(page: int, sheet: int = SHEET_KIND) -> list[int]:
    kind = [(sheet >> b) & 1 for b in (2, 1, 0)]
    num = [(page >> b) & 1 for b in (2, 1, 0)]
    return [1, *kind, *num, 1]


def decode_bits(bits: list[int]) -> tuple[int, int] | None:
    """쪽 코드 → (양식 종류, 쪽 번호). 모르는 양식 종류거나 깨졌으면 None."""
    if len(bits) != CODE_BITS or bits[0] != 1 or bits[-1] != 1:
        return None
    kind = bits[1] * 4 + bits[2] * 2 + bits[3]
    page = bits[4] * 4 + bits[5] * 2 + bits[6]
    if kind not in HANGUL_PAGES or not 1 <= page <= len(pages(kind)):
        return None
    return kind, page


def code_squares() -> list[tuple[float, float]]:
    """쪽 코드 각 칸의 좌상단 (mm)."""
    return [(CODE_X0 + i * (CODE_SQ + CODE_GAP), CODE_Y - CODE_SQ / 2) for i in range(CODE_BITS)]
