"""시험용 가짜 스캔 만들기: 양식 칸에 Noto 글꼴로 글자를 그려 넣고, 사진처럼 기울이고 돌리고 얼룩을 입힌다.

사용: python -m tests.fake_scan [출력폴더]   (기본 out/fake)
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont

from fontmaker import layout as L
from fontmaker.sheet import ROOT, render

DPI = 200
PX = DPI / 25.4
DESK = (70, 80, 90)


def draw_page(page: L.Page, base: Image.Image, font: ImageFont.FreeTypeFont, rng: random.Random) -> Image.Image:
    img = base.convert("RGB")
    d = ImageDraw.Draw(img)
    for cell in page.cells:
        cx = (cell.x + cell.size / 2 + rng.uniform(-0.8, 0.8)) * PX
        if cell.kind == "hangul":
            cy = (cell.y + cell.size / 2 + rng.uniform(-0.8, 0.8)) * PX
            d.text((cx, cy), cell.char, font=font, fill=(20, 20, 25), anchor="mm")
        else:
            by = (cell.y + cell.size * L.LATIN_BASELINE) * PX
            d.text((cx, by), cell.char, font=font, fill=(20, 20, 25), anchor="ms")
    return img


def photo_effect(img: Image.Image, rng: random.Random, rot180: bool) -> np.ndarray:
    a = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = a.shape[:2]
    pad = 120
    canvas = np.full((h + 2 * pad, w + 2 * pad, 3), DESK, np.uint8)
    canvas[pad:pad + h, pad:pad + w] = a
    src = np.float32([[pad, pad], [pad + w, pad], [pad + w, pad + h], [pad, pad + h]])
    dst = src + np.float32([[rng.uniform(-60, 60), rng.uniform(-60, 60)] for _ in range(4)])
    m = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(canvas, m, (canvas.shape[1], canvas.shape[0]), borderValue=DESK)
    shade = np.linspace(0.75, 1.0, out.shape[1], dtype=np.float32)[None, :, None]  # 한쪽 그늘
    noise = np.random.default_rng(1).normal(0, 4, out.shape)
    out = np.clip(out * shade + noise, 0, 255).astype(np.uint8)
    return cv2.rotate(out, cv2.ROTATE_180) if rot180 else out


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = ROOT / "out/sheet_v2.pdf"
    render(pdf)
    font = ImageFont.truetype(str(ROOT / "assets/NotoSansKR-Regular.ttf"), int(12 * PX))
    rng = random.Random(7)
    with pymupdf.open(pdf) as doc:
        for page in L.pages()[:3]:
            pix = doc[page.number - 1].get_pixmap(dpi=DPI)
            base = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            photo = photo_effect(draw_page(page, base, font, rng), rng, rot180=page.number == 2)
            cv2.imwrite(str(out_dir / f"page{page.number}.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{out_dir}: page1~3.jpg")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "out/fake")
