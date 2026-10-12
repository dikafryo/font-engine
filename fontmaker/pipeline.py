"""스캔 파일들 → 폰트. worker.py 가 작업마다 호출한다."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import layout as L
from .compose import TYPES, Composer, Library
from .fontbuild import GlyphSrc, Metrics, build, even_latin_heights, to_woff2
from .hangul import CHO, JONG, JUNG, all_syllables, decompose, segment
from .scan import PX_PER_MM, PageImage, ScanError, align, extract_cells, load_images
from .vector import Bounds, Outline, bounds, fit, trace

Progress = Callable[[str, int], None]
REQUIRED_PAGES = L.REQUIRED_HANGUL_PAGES
INSET_MM = 0.7  # scan.extract_cells 와 같은 값
COMPAT_CONS = "ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
PREVIEW_LINES = ("다람쥐 헌 쳇바퀴에 타고파", "내 손글씨로 만든 폰트입니다.", "ABC abc 0123456789 !?")


class PipelineError(Exception):
    def __init__(self, code: str, message: str, details: list[str] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or []


@dataclass
class Report:
    pages: list[int] = field(default_factory=list)  # 본 양식에서 읽은 쪽
    addon_pages: list[int] = field(default_factory=list)  # 추가 양식에서 읽은 쪽
    samples: int = 0  # 직접 쓴 한글 칸 수 (같은 글자를 여러 번 썼으면 모두 센다)
    file_errors: list[str] = field(default_factory=list)
    direct: int = 0
    composed: int = 0
    latin: int = 0
    not_made: int = 0
    edited: int = 0  # 글자 편집(그림판·한 글자 사진)으로 덮어쓴 글자 수
    empty_cells: list[dict] = field(default_factory=list)  # {char, page, index, kind}
    unsplit: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def read_pages(inputs: list[Path], report: Report) -> list[PageImage]:
    """올린 순서대로 모든 쪽. 같은 쪽을 여러 번 올렸으면(스캔 추가) 모두 돌려준다."""
    pages: list[PageImage] = []
    for path in inputs:
        try:
            images = load_images(path)
        except (ScanError, RuntimeError) as e:
            report.file_errors.append(f"{path.name}: {e}")
            continue
        for i, img in enumerate(images, 1):
            label = path.name if len(images) == 1 else f"{path.name} {i}쪽"
            try:
                pages.append(align(img, label))
            except ScanError as e:
                report.file_errors.append(f"{label}: {e}")
    report.pages = sorted({p.number for p in pages if p.kind == L.SHEET_KIND})
    report.addon_pages = sorted({p.number for p in pages if p.kind == L.ADDON_KIND})
    missing = [p for p in REQUIRED_PAGES if p not in report.pages]
    if missing:
        names = ", ".join(f"{p}쪽" for p in missing)
        raise PipelineError("missing_pages", f"양식 {names}을 찾지 못했습니다", report.file_errors)
    return pages


def run(inputs: list[Path], family: str, uid: str, out: Path, progress: Progress, names: dict[str, str] | None = None,
        glyphs: dict[str, Path] | None = None) -> Report:
    """glyphs: 글자 → 편집한 글자 PNG(검은 획·흰 바탕, 정사각 칸). 스캔에서 읽은 같은 글자를 덮어쓴다."""
    report = Report()
    progress("scan", 5)
    pages = read_pages(inputs, report)

    progress("recognize", 20)
    cell_px = int((L.BOX - 2 * INSET_MM) * PX_PER_MM)
    lib = Library(cell_px)
    direct: dict[str, Outline] = {}
    latin: dict[str, Outline] = {}
    cells = [(page.kind, c) for page in pages for c in extract_cells(page)]
    split_ok: set[str] = set()
    tried: set[str] = set()
    for i, (_, ink) in enumerate(cells):
        if i % 20 == 0:
            progress("recognize", 20 + 30 * i // len(cells))
        if ink.empty:
            continue
        outline = trace(ink.mask)
        if ink.cell.kind == "latin":
            latin[ink.cell.char] = outline  # 여러 번 썼으면 나중에 올린 것
            continue
        direct[ink.cell.char] = outline
        report.samples += 1
        tried.add(ink.cell.char)
        syl = decompose(ink.cell.char)
        parts = segment(ink.mask, syl)
        if parts:
            lib.add(syl, parts)  # 같은 글자의 여러 샘플은 모두 부품 후보 (평소 버릇의 중앙값이 안정된다)
            split_ok.add(ink.cell.char)
    for ch, path in (glyphs or {}).items():  # 편집한 글자: 스캔보다 우선. 한글이면 부품 후보로도 넣는다
        mask = load_glyph(path, cell_px)
        if mask is None:
            continue
        outline = trace(mask)
        syl = decompose(ch)
        if syl is None:
            latin[ch] = outline
        else:
            direct[ch] = outline
            tried.add(ch)
            parts = segment(mask, syl)
            if parts:
                lib.add(syl, parts)
                split_ok.add(ch)
        report.edited += 1
    report.unsplit = sorted(tried - split_ok)
    seen = set(direct) | set(latin)
    # 빈칸 알림은 본 양식만 (추가 양식은 쓴 만큼만 올려도 된다)
    blank = {ink.cell.char: ink.cell for kind, ink in cells if kind == L.SHEET_KIND and ink.empty and ink.cell.char not in seen}
    report.empty_cells = [{"char": c.char, "page": c.page, "index": c.index, "kind": c.kind} for c in blank.values()]
    if len(direct) < 40:
        raise PipelineError("too_few", f"인식된 한글이 {len(direct)}자뿐입니다. 글씨가 너무 흐리거나 칸 밖에 쓰지 않았는지 확인해 주세요")

    progress("compose", 55)
    composer = Composer(lib)
    glyphs = [GlyphSrc(ch, o, "hangul") for ch, o in direct.items()]
    for ch in all_syllables():
        if ch in direct:
            continue
        outline = composer.compose(decompose(ch))
        if outline:
            glyphs.append(GlyphSrc(ch, outline, "hangul"))
            report.composed += 1
        else:
            report.not_made += 1
    glyphs += compat_jamo(lib, cell_px)
    metrics = make_metrics(list(direct.values()), cell_px)
    latin = even_latin_heights(latin, metrics.baseline_px)  # 대문자·숫자·소문자 높이를 무리별로 고르게
    glyphs += [GlyphSrc(ch, o, "latin") for ch, o in latin.items()]
    report.direct, report.latin = len(direct), len(latin)

    progress("build", 85)
    otf = build(glyphs, metrics, family, uid, out / "font.otf", names)
    to_woff2(otf, out / "font.woff2")
    preview(otf, out / "preview.png")
    progress("build", 100)
    return report


def load_glyph(path: Path, cell_px: int) -> np.ndarray | None:
    """편집한 글자 PNG → 칸 크기 bool 마스크(True=잉크). 투명 바탕은 흰색으로 깔고, 잉크가 거의 없으면 None."""
    try:
        img = Image.open(path).convert("RGBA")
    except OSError:
        return None
    flat = Image.new("RGBA", img.size, (255, 255, 255, 255))
    flat.alpha_composite(img)
    gray = flat.convert("L").resize((cell_px, cell_px), Image.LANCZOS)
    mask = np.array(gray) < 128
    return mask if mask.sum() >= PX_PER_MM ** 2 else None


def make_metrics(hangul: list[Outline], cell_px: int) -> Metrics:
    bs = [b for b in map(bounds, hangul) if b]
    scale = 800 / median(b.h for b in bs)
    adv = round(min(1100, max(820, median(b.w for b in bs) * scale + 150)))
    baseline = (L.BOX * L.LATIN_BASELINE - INSET_MM) * PX_PER_MM
    return Metrics(cell_px, baseline, scale, adv)


def compat_jamo(lib: Library, cell_px: int) -> list[GlyphSrc]:
    """호환 자모 ㄱ~ㅣ (ㅋㅋ, ㅠㅠ 용). 부품을 칸 가운데에 키워 놓는다."""
    out = []
    for i, ch in enumerate(COMPAT_CONS):
        keys = []
        if ch in CHO:
            keys.append(("cho", CHO.index(ch)))
        if ch in JONG:
            keys.append(("jong", JONG.index(ch)))
        outline = _first_part(lib, keys)
        if outline:
            out.append(GlyphSrc(chr(0x3131 + i), _center(outline, cell_px, 0.62), "hangul"))
    for i in range(len(JUNG)):
        outline = _first_part(lib, [("jung", i)])
        if outline:
            out.append(GlyphSrc(chr(0x314F + i), _center(outline, cell_px, 0.8), "hangul"))
    return out


def _first_part(lib: Library, keys: list[tuple[str, int]]) -> Outline | None:
    for role, jamo in keys:
        for t in TYPES:
            found = lib.parts.get((role, jamo, t))
            if found:
                return found[0].outline
    return None


def _center(outline: Outline, cell: int, height: float) -> Outline:
    b = bounds(outline)
    k = min(height * cell / max(b.h, 1), 0.8 * cell / max(b.w, 1))
    w, h = b.w * k, b.h * k
    x0, y0 = (cell - w) / 2, (cell - h) / 2
    return fit(outline, b, Bounds(x0, y0, x0 + w, y0 + h))


def preview(otf: Path, out: Path) -> None:
    font = ImageFont.truetype(str(otf), 64)
    img = Image.new("RGB", (1400, 110 * len(PREVIEW_LINES) + 40), "white")
    d = ImageDraw.Draw(img)
    for i, line in enumerate(PREVIEW_LINES):
        d.text((40, 30 + i * 110), line, font=font, fill=(20, 20, 30))
    img.save(out)
