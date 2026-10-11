"""작업 대기열 처리기. cron 이 1분마다 실행하고, 이미 돌고 있으면 바로 끝낸다.

사용: python -m fontmaker.worker <jobs 폴더>

작업 폴더 규약 (사이트 sw4u/font 와 공유):
  <jobs>/<id>/job.json     사이트가 씀  {"id", "font_name", "created_at"(ISO8601 UTC), "touched_at"(다시 만든 시각, 선택), "files": ["input/1.jpg", …],
                                         "copyright", "license_text", "license_url"(선택 — 폰트 name 테이블에 그대로 기록)}
  <jobs>/<id>/input/…      사이트가 씀  업로드 원본
  <jobs>/<id>/status.json  워커가 씀    {"state": queued|processing|done|failed, "step", "progress", "code",
                                         "message", "details", "report", "updated_at"}
  <jobs>/<id>/font.otf · font.woff2 · preview.png   워커가 씀 (완료 시)
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .pipeline import PipelineError, run

RETENTION = timedelta(days=7)
MAX_RUN_SECONDS = 50 * 60  # cron 주기와 겹쳐도 lock 이 막아 준다
log = logging.getLogger("fontmaker.worker")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.chmod(tmp, 0o664)
    tmp.replace(path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def created(job: dict) -> datetime:
    try:
        return datetime.strptime(job["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (KeyError, ValueError):
        return datetime.fromtimestamp(0, timezone.utc)


def last_active(job: dict) -> datetime:
    """보관 정리 기준: 올린 시각과 다시 만든 시각(touched_at — 사이트가 업그레이드·결합 때 씀) 중 늦은 쪽."""
    try:
        touched = datetime.strptime(job.get("touched_at") or "", "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return created(job)
    return max(created(job), touched)


def cleanup(root: Path) -> None:
    limit = datetime.now(timezone.utc) - RETENTION
    for d in root.iterdir():
        job = read_json(d / "job.json") if d.is_dir() else None
        if job is not None and last_active(job) < limit:
            shutil.rmtree(d, ignore_errors=True)
            log.info("보관기간 만료 삭제 %s", d.name)


def pending(root: Path) -> list[tuple[Path, dict]]:
    out = []
    for d in root.iterdir():
        job = read_json(d / "job.json") if d.is_dir() else None
        if job is None:
            continue
        status = read_json(d / "status.json")
        if status is None or status.get("state") == "queued":
            out.append((d, job))
    return sorted(out, key=lambda x: created(x[1]))


def inputs_of(d: Path, job: dict) -> list[Path]:
    base = d.resolve()
    files = [(d / f).resolve() for f in job.get("files", []) if isinstance(f, str)]
    return [f for f in files if f.is_file() and f.is_relative_to(base)]


def process(d: Path, job: dict) -> None:
    status_path = d / "status.json"
    write_json(status_path, {"state": "processing", "step": "scan", "progress": 0, "updated_at": now()})
    last = [0.0]

    def progress(step: str, pct: int) -> None:
        if time.monotonic() - last[0] < 1.0 and pct < 100:
            return
        last[0] = time.monotonic()
        write_json(status_path, {"state": "processing", "step": step, "progress": pct, "updated_at": now()})

    name = str(job.get("font_name") or "내 손글씨")[:30]
    started = time.monotonic()
    try:
        names = {k: str(job[k]) for k in ("copyright", "license_text", "license_url") if isinstance(job.get(k), str) and job[k]}
        report = run(inputs_of(d, job), name, d.name, d, progress, names)
    except PipelineError as e:
        write_json(status_path, {"state": "failed", "code": e.code, "message": str(e), "details": e.details[:20], "updated_at": now()})
        log.info("실패 %s: %s", d.name, e)
        return
    except Exception:  # 예상 못 한 오류도 작업을 '실패'로 닫아야 대기열이 멈추지 않는다
        log.error("내부 오류 %s\n%s", d.name, traceback.format_exc())
        write_json(status_path, {"state": "failed", "code": "internal", "message": "처리 중 알 수 없는 오류가 났습니다", "details": [], "updated_at": now()})
        return
    write_json(status_path, {"state": "done", "step": "build", "progress": 100, "report": report.as_dict(), "updated_at": now()})
    log.info("완료 %s (%.1fs) 직접 %d · 조합 %d", d.name, time.monotonic() - started, report.direct, report.composed)


def main(root: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.umask(0o002)
    with open(root / ".worker.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        cleanup(root)
        deadline = time.monotonic() + MAX_RUN_SECONDS
        while time.monotonic() < deadline:
            jobs = pending(root)
            if not jobs:
                break
            process(*jobs[0])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("사용: python -m fontmaker.worker <jobs 폴더>")
    main(Path(sys.argv[1]))
