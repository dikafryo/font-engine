# fontmaker — 손글씨 양식 스캔 → 한글 폰트

font.sw4u.kr(`sw4u/font`) 의 폰트 엔진. 웹 서빙 대상 아님. 사이트와는 작업 폴더 규약으로만 연결된다(`sw4u/font/CLAUDE.md`).

## 흐름
1. `layout.py` — 양식 칸 좌표의 단일 출처 (A4, mm). 4쪽: 한글 175칸(2쪽), 영문·숫자·기호 81칸, 문장(분석용, 선택)
2. `sheet.py` — 양식 PDF 생성. 모서리 5mm 마커 4개 + 아래 가운데 쪽 코드 8비트 `[1, kind×3, page×3, 1]`
3. `scan.py` — 조명 보정 → 마커 4개로 원근 보정 → 4방향 중 쪽 코드가 읽히는 방향 선택 → 칸별 잉크 마스크
4. `hangul.py` — 음절을 초·중·종성 부품으로 분리 (획 덩어리 + 위치 규칙, 붙은 곳은 가장 가는 줄을 잘라 재시도)
5. `vector.py` — potrace 로 베지어 외곽선
6. `compose.py` — 부품을 (역할, 자모, 위치유형 V0/V1/H0/H1/M0/M1) 별로 모아 11,172자 조합. 없는 유형은 역할 평균 상자 사이 변환,
   없는 겹자음은 홑자음 둘을 나란히, 초성↔받침 같은 꼴 서로 빌림
7. `fontbuild.py` — OTF(CFF) + WOFF2. 이름표 영어·한국어(0x0412) 모두 사용자 이름
8. `pipeline.py` 가 위를 묶고, `worker.py` 가 cron 에서 대기열을 처리 (lock, 7일 보관 정리)

## 명령
```bash
cd /var/services/web/apps/font
.venv/bin/python -m fontmaker.sheet [출력.pdf]          # 양식 PDF
.venv/bin/python -m tests.fake_scan                     # Noto 글꼴로 채운 가짜 사진 3장 → out/fake/
.venv/bin/python -m fontmaker.worker <jobs 폴더>        # cron 이 1분마다 실행
```
venv: reportlab, fonttools, brotli, opencv-python-headless, numpy, potracer, pymupdf (Python 3.10, 호스트).

## 알려진 한계
- 자모를 이어 쓰는(흘려 쓰는) 글씨는 부품 분리가 실패할 수 있다. 실패한 글자는 통째로만 쓰이고 부품 공급원에서 빠진다.
- 조합 글자는 다른 음절의 부품을 크기만 맞춰 옮긴 것이라, 직접 쓴 글자보다 어색할 수 있다.
- 실제 손글씨로는 아직 검증 전(합성 글씨로만 시험). 첫 실사용 결과를 보고 `hangul.py` 규칙·`compose.py` 상자를 조정할 것.

## 2026-10-10 저녁 — 폰트 메타데이터
사이트가 `job.json` 에 `copyright`·`license_text`·`license_url` 을 넣어 주면 워커가 그대로 name 테이블
ID 0(copyright)·13(licenseDescription)·14(licenseInfoURL)에 기록한다(`fontbuild.build(..., names)`). 없으면 기본 문구.
문구의 기준은 사이트 `sw4u/font/src/License.php` 한 곳이며 엔진은 라이선스 종류를 모른다.
