# fontmaker — 손글씨 양식 스캔 → 한글 폰트

font.sw4u.kr(`sw4u/font`) 의 폰트 엔진. 웹 서빙 대상 아님. 사이트와는 작업 폴더 규약으로만 연결된다(`sw4u/font/CLAUDE.md`).

## 저장소 (GitHub, 비공개)
- 엔진(여기): https://github.com/dikafryo/font-engine
- 사이트: https://github.com/dikafryo/font
- 원격은 SSH(`git@github.com:dikafryo/…`). `.venv/`·`out/`·`scans/` 는 올리지 않는다
- 작업 폴더 규약(`job.json`·`status.json`)을 바꾸면 사이트·엔진 두 저장소에 각각 커밋·푸시한다

## 흐름
1. `layout.py` — 양식 칸 좌표의 단일 출처 (A4, mm). v3 6쪽: 한글 1·2쪽 필수(기본 175자+14) · 3·4쪽 권장(`EXTRA_HANGUL`, 총 387자), 5쪽 영문·숫자·기호 81칸, 6쪽 문장(선택). `SHEET_KIND=2` 만 읽는다(v2 폐기)
2. `sheet.py` — 양식 PDF 생성. 모서리 5mm 마커 4개 + 아래 가운데 쪽 코드 8비트 `[1, kind×3, page×3, 1]`
3. `scan.py` — 조명 보정 → 마커 4개로 원근 보정 → 4방향 중 쪽 코드가 읽히는 방향 선택 → 칸별 잉크 마스크
4. `hangul.py` — 음절을 초·중·종성 부품으로 분리. 후보(획 덩어리 규칙 배정·기준 상자 배정 × 붙은 곳 0~3군데 자르기, 기준 상자로 픽셀 가르기) 중
   역할 상자가 기준 배치(`reference.py`)와 가장 닮은 것을 고른다
5. `vector.py` — potrace 로 베지어 외곽선
6. `compose.py` — 11,172자 조합. 자모마다 기준 배치가 대상 음절과 가장 비슷한 원래 음절의 부품을 고르고, 기준 상자 사이 비례 변환을
   `FOLLOW`(0.5)만큼 따라 옮긴다(쓴 사람의 버릇 유지). 없는 겹자음은 홑자음 둘을 나란히, 초성↔받침 같은 꼴 서로 빌림
6-1. `reference.py` + `data/reference.npz` — Noto Sans KR Thin 11,172자의 역할별 상자(칸 비율). 만들기: `scripts/make_reference.py`
7. `fontbuild.py` — OTF(CFF) + WOFF2. 이름표 영어·한국어(0x0412) 모두 사용자 이름
8. `pipeline.py` 가 위를 묶고, `worker.py` 가 cron 에서 대기열을 처리 (lock, 7일 보관 정리)

## 명령
```bash
cd /var/services/web/apps/font
.venv/bin/python -m fontmaker.sheet [출력.pdf]          # 양식 PDF
.venv/bin/python -m tests.fake_scan                     # Noto 글꼴로 채운 가짜 사진 3장 → out/fake/
.venv/bin/python -m fontmaker.worker <jobs 폴더>        # cron 이 1분마다 실행
.venv/bin/python -m tests.evaluate <기준글꼴.ttf>...      # 품질 평가: 가짜 스캔 → 폰트 → 조합 글자 vs 기준 글꼴 IoU, out/eval/<글꼴>/compare*.png
.venv/bin/python -m scripts.make_reference <NotoSansKR[wght].ttf>   # 기준 배치 자료 재생성 (약 1분)
```
- 손글씨풍 평가 글꼴(OFL): google/fonts 의 nanumpenscript·gaegu·himelody·gamjaflower·nanumbrushscript·yeonsung. 저장소에는 넣지 않는다
venv: reportlab, fonttools, brotli, opencv-python-headless, numpy, potracer, pymupdf (Python 3.10, 호스트).

## 알려진 한계
- 자모를 이어 쓰는(흘려 쓰는) 글씨는 부품 분리가 실패할 수 있다. 실패한 글자는 통째로만 쓰이고 부품 공급원에서 빠진다.
- 조합 글자는 다른 음절의 부품을 크기만 맞춰 옮긴 것이라, 직접 쓴 글자보다 어색할 수 있다.
- 실제 손글씨로는 아직 검증 전(손글씨풍 글꼴·흔들림 합성으로만 시험). 첫 실사용 결과를 `tests.evaluate` 비교 그림처럼 확인할 것.

## 2026-10-11 저녁 — 추가 양식 · 스캔 더 넣기
- 추가 양식(쪽 코드 종류 3, `layout.ADDON_HANGUL` 387자, `sheet.py --addon`): v3 387자를 다 쓴 뒤 남는 빈틈을 메우는 순서(빈틈 비용 0.034→0.012)
- `pipeline.read_pages` 는 같은 쪽을 여러 번 받는다(사이트 '스캔 더 넣기'가 job.json files 에 이어 붙임). 같은 글자의 샘플은 모두 부품 후보,
  폰트에 그대로 넣는 글자는 나중에 올린 것. 보고서에 `addon_pages`·`samples` 추가, 빈칸 알림은 본 양식만
- `scan`: 스캔 앱 흑백 필터로 안내선이 펜만큼 진해지는 경우 — 인쇄 줄이 있을 자리(칸 가장자리·영문 기준선) 띠에서 긴 줄·납작한 조각을 지움
- 합성 손글씨: 처음 양식 0.643 → +같은 양식 0.647 → +추가 양식 0.657 → +둘 다 0.658
- `worker.cleanup` 은 created_at 과 touched_at(사이트가 업그레이드·결합 때 씀) 중 늦은 쪽으로 7일을 센다 — 사이트가 원본 보관본에서 작업 폴더를 되살려 다시 만들기 때문

## 2026-10-11 오후 — 양식 v3 (샘플 387자)
첫 실제 손글씨(v2, 가는 펜·흑백 인쇄)로 만든 폰트가 쓸 수 없는 수준이라 개편.
- 실제 글씨 교차 검증(직접 쓴 글자를 가리고 나머지로 조합해 비교): 샘플 35→140자에 IoU 0.21→0.31, 포화 없음 → 한글을 387자로
- 더 받는 212자: 기본 글자로 조합할 때 부품을 가장 멀리 옮겨야 하는 빈틈을 많이 메우는 순으로 2,350자에서 선정(일상 문장 글자 15% 가산)
- `compose.DENOISE=0.5`: 부품을 놓을 때 샘플 한 번의 흔들림 대신 그 자모의 평소 버릇(중앙값)을 섞음 (실제 0.294→0.313)
- `scan`: 흑백 인쇄 안내선(밝기 ~0.6)이 잉크로 잡혀 영문 밑줄이 생기던 문제 — 행·열 중앙값으로 안내선을 찾아 연한 픽셀만 지움. 문장부호 빈칸 기준 0.2mm²
- 흔들림 합성 평가: 0.568(v2) → 0.648(v3). 1·2쪽만 올려도 동작(직접 189자)

## 2026-10-11 — 분리·조합 정확도 개선
손글씨풍 글꼴 7종 평가(tests.evaluate): 분리 성공 95% → 99.9%, 조합 글자 IoU 평균 0.46 → 0.60 (흔들림 스캔 0.57).
글자마다 크기·기울기·굵기를 흔든 스캔(`--jitter`)에서도 분리 100%, IoU 0.56. 직접 쓴 글자는 IoU 0.89~0.92
(스캔→외곽선은 거의 그대로 옮김) — 조합 글자와의 차이는 남의 음절 부품을 옮겨 쓰는 조합 자체의 한계라 샘플을 늘리는 것이 가장 큰 개선책.
- 기준 배치: 윤곽선 모양 공유 투표 + 위치 사전확률 + 고리(ㅇㅁㅂㅎ)는 모음 아님. 기준 글꼴이 두 자모를 한 윤곽선으로 합친 글자(곽·계·고 등)는
  배치 상식 검사(ㅔ·ㅖ 폭, ㅗ 꼭지 위치, 가로모음 폭)와 이웃 중앙값 비교로 걸러 비슷한 글자의 중앙값으로 메운다
- 샘플 수 실험(양식 175자 + 빈틈을 가장 많이 메우는 2,350자 안의 글자): IoU +0 0.598 / +54 0.657 / +108 0.677.
  양식에 한글 쪽을 더하면 품질이 확실히 오른다 — 양식 변경은 `SHEET_KIND` 올리고 PDF 재생성 + `make_reference` 재실행

## 2026-10-10 저녁 — 폰트 메타데이터
사이트가 `job.json` 에 `copyright`·`license_text`·`license_url` 을 넣어 주면 워커가 그대로 name 테이블
ID 0(copyright)·13(licenseDescription)·14(licenseInfoURL)에 기록한다(`fontbuild.build(..., names)`). 없으면 기본 문구.
문구의 기준은 사이트 `sw4u/font/src/License.php` 한 곳이며 엔진은 라이선스 종류를 모른다.

## 글자 편집 덮어쓰기 (2026-10-11)
job.json 의 `"glyphs": {"가": "glyphs/AC00.png"}` 가 있으면 `pipeline.run(..., glyphs=)` 이 그 PNG(흰 바탕·검은 획, 정사각)를 칸 크기로 줄여 마스크로 쓴다.
스캔에서 읽은 같은 글자를 덮어쓰고, 한글이면 `segment` 로 부품도 뽑아 조합에 쓴다. `Report.edited` 로 센다. 사이트의 "글자 하나씩 고치기"가 만든다.
