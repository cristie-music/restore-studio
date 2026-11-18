"""Restore Studio - command line (no browser): full track -> stems, MIDI, loops in projects\\<id>\\.

    restore.bat "D:\\music\\song.flac" [--bpm 120] [--model sw6|rofo4] [--quality high|normal|fast] [--device auto|cuda|cpu]
"""
import argparse
import os
import shutil
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from studio import pipeline as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Restore Studio CLI")
    ap.add_argument("audio")
    ap.add_argument("--bpm", type=float)
    ap.add_argument("--downbeat", type=float)
    ap.add_argument("--model", default="sw6", choices=["sw6", "rofo4"])
    ap.add_argument("--quality", default="high", choices=["high", "normal", "fast"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--only", help="comma list: drums,bass,guitar,piano,other,vocals")
    ap.add_argument("--project", help="re-run an existing project id instead of creating a new one")
    a = ap.parse_args()
    if a.project:
        pid = a.project
    else:
        if not os.path.isfile(a.audio):
            sys.exit(f"Файл не найден: {a.audio}")
        pid = P.create_project(os.path.basename(a.audio), lambda dst: shutil.copyfile(a.audio, dst))
    opts = dict(separator=a.model, quality=a.quality, device=a.device, bpm=a.bpm, downbeat=a.downbeat)
    if a.only:
        on = set(a.only.split(","))
        opts["instruments"] = {k: k in on for k in P.DEFAULT_OPTIONS["instruments"]}
    job = P.Job(pid, opts)
    job.run()
    print(f"\nПроект: {pid}  статус: {job.status}")
    sys.exit(0 if job.status == "done" else 1)


if __name__ == "__main__":
    main()
