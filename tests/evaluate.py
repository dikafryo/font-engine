"""품질 평가: 기준 글꼴로 가짜 스캔을 만들어 폰트를 뽑고, 조합된 글자를 기준 글꼴의 같은 글자와 비교한다.

사용: python -m tests.evaluate <기준글꼴.ttf> [...] [--out out/eval] [--sample 600]

지표
- split: 직접 쓴 한글 중 부품 분리에 성공한 비율
- iou:   조합 글자 vs 기준 글자 (두 글꼴의 직접 쓴 글자 높이 중앙값으로 크기를 맞추고, 중심을 맞춘 뒤 획을 조금 불린 IoU)
- --jitter: 글자마다 크기·기울기·굵기를 흔든다 (사람 손글씨의 들쭉날쭉함 흉내)
- worst: IoU 가 가장 낮은 글자들 (규칙 조정의 출발점). 비교 그림 out/eval/<글꼴>/compare*.png (위=만든 폰트, 아래=기준)
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from fontmaker import layout as L
from fontmaker.pipeline import run
from fontmaker.sheet import ROOT

from .fake_scan import main as fake_scan

PX = 96
COMMON = "가나다라마바사아자차카타파하각난닫랄맘밥삿앙잦찿캌탙팦핳개걔게계과괘괴교구궈궤귀규그긔기깨꺼께꼬꾀꿔뀌끄끼" \
         "낚넋닳많않얹읊짧흙했었겠됐봤왔줬쳤폈혔훨휘흐희힘"


def ink(font: ImageFont.FreeTypeFont, ch: str) -> np.ndarray | None:
    img = Image.new("L", (PX * 2, PX * 2), 0)
    ImageDraw.Draw(img).text((PX, PX), ch, font=font, fill=255, anchor="mm")
    a = np.array(img) > 96
    return a if a.any() else None


def box(a: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(a)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def normalize(a: np.ndarray, k: float, size: int = 96) -> np.ndarray:
    x0, y0, x1, y1 = box(a)
    crop = a[y0:y1, x0:x1].astype(np.uint8) * 255
    w, h = max(1, min(size, round((x1 - x0) * k))), max(1, min(size, round((y1 - y0) * k)))
    crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA) > 96
    out = np.zeros((size, size), bool)
    oy, ox = (size - h) // 2, (size - w) // 2
    out[oy:oy + h, ox:ox + w] = crop
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    kern = np.ones((5, 5), np.uint8)
    a = cv2.dilate(a.astype(np.uint8), kern) > 0
    b = cv2.dilate(b.astype(np.uint8), kern) > 0
    return float((a & b).sum() / max(1, (a | b).sum()))


def scale_of(font: ImageFont.FreeTypeFont, chars: list[str]) -> float:
    hs = [box(a)[3] - box(a)[1] for ch in chars if (a := ink(font, ch)) is not None]
    return 60 / median(hs)


def evaluate(ref_path: Path, out_root: Path, sample: int, jitter: bool = False) -> dict:
    name = ref_path.stem
    work = out_root / name
    scans = work / "scans"
    if not (scans / "page3.jpg").exists():
        fake_scan(scans, ref_path, jitter)
    rep = run(sorted(scans.glob("page*.jpg")), name, "eval" + name, work, lambda s, p: None)
    built = ImageFont.truetype(str(work / "font.otf"), PX)
    ref = ImageFont.truetype(str(ref_path), PX)
    direct = L.hangul_chars()
    k_b, k_r = scale_of(built, direct), scale_of(ref, direct)

    rng = random.Random(1)
    skip = set(direct)
    have = TTFont(str(ref_path)).getBestCmap()  # 2,350자만 있는 기준 글꼴도 있다
    pool = [chr(c) for c in range(0xAC00, 0xD7A4) if chr(c) not in skip and c in have]
    chars = [c for c in COMMON if c not in skip and ord(c) in have]
    chars += rng.sample([c for c in pool if c not in chars], min(sample, len(pool)))
    scores: list[tuple[float, str]] = []
    for ch in chars:
        a, b = ink(built, ch), ink(ref, ch)
        scores.append((iou(normalize(a, k_b), normalize(b, k_r)) if a is not None and b is not None else 0.0, ch))
    scores.sort()
    compare(scores[:40], built, ref, k_b, k_r, work / "compare.png")
    mid = len(scores) // 2
    compare(scores[mid - 20:mid + 20], built, ref, k_b, k_r, work / "compare_mid.png")
    return {
        "font": name,
        "split": 1 - len(rep.unsplit) / max(1, rep.direct),
        "unsplit": "".join(rep.unsplit),
        "not_made": rep.not_made,
        "iou": mean(s for s, _ in scores),
        "p10": scores[len(scores) // 10][0],
        "worst": "".join(c for _, c in scores[:25]),
    }


def compare(items: list[tuple[float, str]], built, ref, k_b: float, k_r: float, out: Path) -> None:
    """위: 만든 폰트(조합), 아래: 기준 글꼴. 8글자씩 한 줄."""
    cols, cell = 8, 100
    rows = (len(items) + cols - 1) // cols
    img = np.full((rows * cell * 2, cols * cell), 255, np.uint8)
    for i, (s, ch) in enumerate(items):
        r, c = divmod(i, cols)
        for j, (font, k) in enumerate(((built, k_b), (ref, k_r))):
            a = ink(font, ch)
            if a is None:
                continue
            y = (r * 2 + j) * cell + 2
            img[y:y + 96, c * cell + 2:c * cell + 98][normalize(a, k)] = 0
        cv2.rectangle(img, (c * cell, r * 2 * cell), (c * cell + cell - 1, r * 2 * cell + 2 * cell - 1), 200, 1)
        cv2.putText(img, f"{s:.2f}", (c * cell + 3, r * 2 * cell + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 120, 1)
    cv2.imwrite(str(out), img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fonts", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "out/eval")
    ap.add_argument("--sample", type=int, default=600)
    ap.add_argument("--jitter", action="store_true", help="글자마다 크기·기울기·굵기를 흔든 손글씨풍 스캔")
    args = ap.parse_args()
    results = [evaluate(f, args.out, args.sample, args.jitter) for f in args.fonts]
    for r in results:
        print(f"{r['font']:28s} split {r['split']:.2f}  iou {r['iou']:.3f}  p10 {r['p10']:.3f}  not_made {r['not_made']}")
        print(f"   unsplit: {r['unsplit']}")
        print(f"   worst:   {r['worst']}")
    print(f"{'평균':28s} split {mean(r['split'] for r in results):.3f}  iou {mean(r['iou'] for r in results):.3f}")


if __name__ == "__main__":
    main()
