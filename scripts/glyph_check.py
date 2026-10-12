#!/usr/bin/env python3
"""진단: 폰트의 글자 표와 스캔 칸 마스크를 그림으로 모아 눈으로 점검한다.

사용:
  .venv/bin/python scripts/glyph_check.py font <font.otf> <out.png>          — 영문·숫자(기준선 표시) + 한글 직접·조합 글자 표
  .venv/bin/python scripts/glyph_check.py cells <out_dir> <스캔 파일…>         — 쪽마다 칸 마스크 모음(빨간 틀 = 빈칸 판정)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fontmaker import layout as L  # noqa: E402
from fontmaker.scan import PX_PER_MM, align, extract_cells, load_images  # noqa: E402

SAMPLE_HANGUL = "가나다라마바사아자차카타파하 각난달람맙삽앙잦찾캌탙팦핳 괜뷁쀍훑 를ㅋㅋㅠㅠ"


def font_sheet(otf: Path, out: Path) -> None:
    size = 72
    f = ImageFont.truetype(str(otf), size)
    cols = 20
    rows_latin = (len(L.LATIN) + cols - 1) // cols
    hangul = [c for c in L.hangul_chars() + list(SAMPLE_HANGUL) if c != " "]
    rows_h = (len(hangul) + cols - 1) // cols
    cw, ch = 90, 110
    img = Image.new("RGB", (cols * cw + 20, (rows_latin + rows_h) * ch + 60), "white")
    d = ImageDraw.Draw(img)
    asc = f.getmetrics()[0]
    for i, c in enumerate(L.LATIN):
        r, k = divmod(i, cols)
        x, y = 10 + k * cw, 10 + r * ch
        base = y + asc
        d.line([(x, base), (x + cw, base)], fill=(255, 120, 120))  # 기준선
        d.line([(x, base - 700 * size / 1000), (x + cw, base - 700 * size / 1000)], fill=(160, 200, 255))  # 대문자선(700)
        d.line([(x, base - 500 * size / 1000), (x + cw, base - 500 * size / 1000)], fill=(220, 235, 255))  # x 높이(500)
        d.text((x + 10, y), c, font=f, fill="black")
    y0 = 20 + rows_latin * ch
    for i, c in enumerate(hangul):
        r, k = divmod(i, cols)
        x, y = 10 + k * cw, y0 + r * ch
        d.rectangle([x, y, x + cw - 4, y + ch - 4], outline=(230, 230, 230))
        d.text((x + 8, y), c, font=f, fill="black")
    img.save(out)
    print("저장:", out)


def cell_sheet(out_dir: Path, inputs: list[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in inputs:
        for i, raw in enumerate(load_images(path), 1):
            try:
                page = align(raw, f"{path.name} {i}")
            except Exception as e:  # noqa: BLE001
                print(f"{path.name} {i}쪽: {e}")
                continue
            cells = extract_cells(page)
            if not cells:
                continue
            n = cells[0].mask.shape[0]
            cols = L.COLS
            rows = (len(cells) + cols - 1) // cols
            pad = 6
            img = Image.new("RGB", (cols * (n + pad), rows * (n + pad)), "white")
            d = ImageDraw.Draw(img)
            for j, c in enumerate(cells):
                r, k = divmod(j, cols)
                x, y = k * (n + pad), r * (n + pad)
                m = Image.fromarray(np.where(c.mask, 0, 255).astype(np.uint8)).convert("RGB")
                img.paste(m, (x, y))
                d.rectangle([x, y, x + n - 1, y + n - 1], outline=(255, 0, 0) if c.empty else (200, 200, 200))
                if c.cell.kind == "latin":
                    b = int((L.BOX * L.LATIN_BASELINE - 0.7) * PX_PER_MM)
                    d.line([(x, y + b), (x + n, y + b)], fill=(255, 150, 150))
                d.text((x + 2, y + 2), c.cell.char, fill=(0, 120, 255))
            dest = out_dir / f"{path.stem}_{i}_p{page.number}.png"
            img.save(dest)
            print("저장:", dest, "빈칸", sum(c.empty for c in cells))


if __name__ == "__main__":
    if sys.argv[1] == "font":
        font_sheet(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        cell_sheet(Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]])
