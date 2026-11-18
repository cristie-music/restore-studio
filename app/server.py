"""Restore Studio web server: the browser UI + JSON API around the restore pipeline.

    python app/server.py [--host 127.0.0.1] [--port 8790] [--no-browser]

Only the standard library is used for HTTP. One pipeline job runs at a time (the GPU is shared), others wait in a queue.
"""
import argparse
import json
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

APP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP)
WEB = os.path.join(APP, "web")
sys.path.insert(0, APP)

from studio import audio_io as A  # noqa: E402
from studio import device as D  # noqa: E402
from studio import midi_tools  # noqa: E402
from studio import models as M  # noqa: E402
from studio import pipeline as P  # noqa: E402

VERSION = "1.0.0"
os.makedirs(P.PROJECTS, exist_ok=True)
mimetypes.add_type("audio/mpeg", ".mp3")
mimetypes.add_type("audio/flac", ".flac")
mimetypes.add_type("audio/wav", ".wav")
mimetypes.add_type("audio/midi", ".mid")
mimetypes.add_type("text/plain; charset=utf-8", ".txt")
mimetypes.add_type("application/javascript", ".js")

JOBS = {}            # pid -> Job (latest run of that project)
QUEUE = queue.Queue()
DOWNLOADS = {}       # model key -> {done, total, status, error}
SYSTEM = {}


def worker():
    while True:
        job = QUEUE.get()
        if job._cancel:
            job.status = "cancelled"
            continue
        try:
            job.run()
        except Exception:
            traceback.print_exc()


def system_info(refresh=False):
    if refresh or not SYSTEM:
        d = D.detect("auto")
        SYSTEM.update(d, label=D.label(d), version=VERSION, python=sys.version.split()[0],
                      colab="COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ)
    return dict(SYSTEM, models=M.status(), separators={k: dict(title=v["title"], about=v["about"], size_mb=v["size_mb"],
                                                                installed=M.installed(k))
                                                       for k, v in M.SEPARATORS.items()},
                downloads=DOWNLOADS, queue=QUEUE.qsize(), projects_dir=P.PROJECTS)


def project_list():
    out = []
    for pid in sorted(os.listdir(P.PROJECTS), reverse=True):
        try:
            m = P.load_manifest(pid)
        except Exception:
            continue
        j = JOBS.get(pid)
        a = m.get("analysis") or {}
        out.append(dict(id=pid, name=m["name"], created=m["created"], status=j.status if j else m["status"],
                        progress=j.progress if j else None, bpm=(m.get("grid") or {}).get("bpm") or a.get("bpm"),
                        key=a.get("key"), camelot=a.get("camelot"), duration=m.get("duration"),
                        stems=len(m.get("stems") or []),
                        n_midi=sum(len(v["midi"]) for v in (m.get("files") or {}).values()),
                        n_loops=sum(len(v["loops"]) for v in (m.get("files") or {}).values()) + len(m.get("mix_loops") or [])))
    return out


def safe_path(pid, rel):
    base = os.path.realpath(P.project_dir(pid))
    p = os.path.realpath(os.path.join(base, unquote(rel)))
    if not (p == base or p.startswith(base + os.sep)):
        raise PermissionError(rel)
    return p


# ---------------------------------------------------------------- export (zip)
def build_export(pid, what):
    """what: all | fl | stems | midi | loops. Returns path of the zip in projects/<id>/export."""
    m = P.load_manifest(pid)
    pdir = P.project_dir(pid)
    name = re.sub(r"[^\w\- ]+", "_", m["name"], flags=re.U).strip() or "track"
    bpm = (m.get("grid") or {}).get("bpm") or (m.get("analysis") or {}).get("bpm")
    entries = []          # (src_abs, arcname)

    def add(rel, arc):
        src = os.path.join(pdir, rel)
        if os.path.isfile(src):
            entries.append((src, arc))

    files = m.get("files") or {}
    eng = m.get("engines") or {}
    title = {k: re.sub(r"[\s_]+", "_", re.sub(r"[^\w\- ]+", " ", v.get("title", k), flags=re.U).strip())
             for k, v in eng.items()}
    if what in ("all", "fl", "stems"):
        add(m["mix"]["file"], f"01_Stems/{os.path.basename(m['mix']['file'])}")
        for s in m.get("stems") or []:
            if s["active"] or what == "all":
                add(s["file"], f"01_Stems/{os.path.basename(s['file'])}")
        if what != "stems":
            for slot, d in files.items():
                for f in d["stems"]:
                    if "/aligned/" in f:
                        continue
                    add(f, f"01_Stems/{slot}_parts/{os.path.basename(f)}")
    if what in ("all", "fl", "midi"):
        if m.get("arrangement_midi"):
            add(m["arrangement_midi"], f"02_MIDI/00_FULL_ARRANGEMENT_{bpm:g}bpm.mid")
        for slot, d in files.items():
            for f in d["midi"]:
                add(f, f"02_MIDI/{title.get(slot, slot)}/{os.path.basename(f)}")
            if what != "fl":
                for f in d["midi_raw"]:
                    add(f, f"02_MIDI/{title.get(slot, slot)}/raw_unquantized/{os.path.basename(f)}")
    if what in ("all", "fl", "loops"):
        for slot, d in files.items():
            for f in d["loops"]:
                sub = f.split("/loops/", 1)[-1]
                add(f, f"03_Loops/{title.get(slot, slot)}/{sub}")
        for lp in m.get("mix_loops") or []:
            for f in lp["files"]:
                add(f, f"03_Loops/Full_mix/{lp['name']}/{os.path.basename(f)}")
    if what in ("all", "fl"):
        for slot, d in files.items():
            for f in d["oneshots"]:
                add(f, f"04_OneShots/{title.get(slot, slot)}/{os.path.basename(f)}")
            for f in d["docs"]:
                add(f, f"05_Reports/{title.get(slot, slot)}_{os.path.basename(f)}")
            if what == "all":
                for f in d["previews"]:
                    add(f, f"06_MIDI_previews/{title.get(slot, slot)}_{os.path.basename(f)}")
    if not entries:
        raise FileNotFoundError("нечего экспортировать (проект ещё не обработан?)")
    out = os.path.join(pdir, "export", f"{name}_{what}.zip")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    readme = fl_readme(m)
    with zipfile.ZipFile(out + ".tmp", "w", allowZip64=True) as z:
        z.writestr("README_RESTORE_STUDIO.txt", readme.encode("utf-8-sig"), compress_type=zipfile.ZIP_DEFLATED)
        for src, arc in entries:
            ct = zipfile.ZIP_STORED if src.lower().endswith(".wav") else zipfile.ZIP_DEFLATED
            z.write(src, arc, compress_type=ct)
    os.replace(out + ".tmp", out)
    return out


def fl_readme(m):
    a = m.get("analysis") or {}
    g = m.get("grid") or {}
    ch = m.get("chords") or {}
    L = [f"RESTORE STUDIO - {m['name']}", "=" * 60,
         f"Темп: {g.get('bpm', a.get('bpm'))} BPM    Тональность: {a.get('key')} ({a.get('camelot')}, {a.get('open_key')})",
         f"Громкость оригинала: {a.get('lufs')} LUFS", f"Разделение: {(m.get('separator') or {}).get('model')}",
         f"Сдвиг на сетку: {float(g.get('shift_s') or 0) * 1000:+.0f} мс (такт 1 = начало проекта)", "",
         "КАК СОБРАТЬ ПРОЕКТ В FL STUDIO (или любой DAW):",
         f"  1. Новый проект, темп {g.get('bpm', a.get('bpm'))} BPM, размер 4/4.",
         "  2. 01_Stems: перетащите все WAV в Playlist с самого начала (такт 1) - они уже выровнены по сетке.",
         "  3. 02_MIDI/00_FULL_ARRANGEMENT: вся аранжировка одним файлом - File > Import > MIDI file,",
         "     каждая партия ляжет в свой канал. Или берите MIDI по инструментам из подпапок.",
         "     Ударные - раскладка General MIDI (36 бочка, 38 малый, 42/46 хэт...) - для FPC / GetGood Drums / SSD.",
         "  4. 03_Loops: лупы ровно в 1/4/8 тактов на темпе проекта (WAV + MIDI тех же тактов).",
         "  5. 04_OneShots: отдельные удары/ноты/аккорды для Sampler / FPC / DirectWave.",
         "  6. 05_Reports: отчёты по каждой партии - аккорды по тактам, ноты, табы, форма трека.", ""]
    if ch.get("items"):
        bpb = 4
        L.append(f"АККОРДЫ (по партии «{ch.get('source')}», доли от начала, 4 доли = такт):")
        row = []
        for c in ch["items"][:400]:
            row.append(f"т{int(c['beat0'] // bpb) + 1}:{c['name']}")
            if len(row) == 8:
                L.append("  " + "  ".join(row))
                row = []
        if row:
            L.append("  " + "  ".join(row))
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- HTTP
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "RestoreStudio/" + VERSION

    def log_message(self, fmt, *args):
        if "/api/jobs" in (args[0] if args else ""):
            return
        if args and isinstance(args[1], str) and args[1].startswith(("4", "5")):
            sys.stderr.write("  http %s\n" % (fmt % args))

    # helpers ----------------------------------------------------------
    def send_json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def err(self, code, msg):
        self.send_json(dict(error=msg), code)

    def body_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def send_file(self, path, download=False, name=None):
        if not os.path.isfile(path):
            return self.err(404, "файл не найден")
        size = os.path.getsize(path)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-cache")
        if download:
            from urllib.parse import quote
            fn = name or os.path.basename(path)
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(fn)}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    b = f.read(min(1 << 20, left))
                    if not b:
                        break
                    self.wfile.write(b)
                    left -= len(b)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass

    # routes -----------------------------------------------------------
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(u.query).items()}
        path = u.path
        try:
            if path in ("/", "/index.html"):
                return self.send_file(os.path.join(WEB, "index.html"))
            if path.startswith("/static/"):
                p = os.path.realpath(os.path.join(WEB, unquote(path[len("/static/"):])))
                if not p.startswith(os.path.realpath(WEB)):
                    return self.err(403, "forbidden")
                return self.send_file(p)
            if path == "/api/system":
                return self.send_json(system_info(q.get("refresh") == "1"))
            if path == "/api/projects":
                return self.send_json(project_list())
            m = re.fullmatch(r"/api/projects/([\w\-]+)", path)
            if m:
                pid = m.group(1)
                man = P.load_manifest(pid)
                j = JOBS.get(pid)
                man["job"] = j.state(len(j.log)) if j else None
                return self.send_json(man)
            m = re.fullmatch(r"/api/jobs/([\w\-]+)", path)
            if m:
                j = JOBS.get(m.group(1))
                if not j:
                    return self.send_json(dict(status="none"))
                return self.send_json(j.state(int(q.get("since", 0))))
            m = re.fullmatch(r"/files/([\w\-]+)/(.+)", path)
            if m:
                return self.send_file(safe_path(m.group(1), m.group(2)), q.get("dl") == "1")
            m = re.fullmatch(r"/api/peaks/([\w\-]+)/(.+)", path)
            if m:
                return self.send_json(A.peaks_cached(safe_path(m.group(1), m.group(2)), int(q.get("n", 800))))
            m = re.fullmatch(r"/api/midi/([\w\-]+)/(.+)", path)
            if m:
                return self.send_json(midi_tools.notes(safe_path(m.group(1), m.group(2))))
            m = re.fullmatch(r"/api/text/([\w\-]+)/(.+)", path)
            if m:
                p = safe_path(m.group(1), m.group(2))
                return self.send_json(dict(text=open(p, encoding="utf-8-sig", errors="replace").read()))
            m = re.fullmatch(r"/api/export/([\w\-]+)", path)
            if m:
                what = q.get("what", "fl")
                if what not in ("all", "fl", "stems", "midi", "loops"):
                    return self.err(400, "what?")
                z = build_export(m.group(1), what)
                try:
                    return self.send_file(z, True)
                finally:   # archives are rebuilt on demand in seconds - do not keep hundreds of MB on disk
                    try:
                        os.remove(z)
                    except OSError:
                        pass
            return self.err(404, "нет такого адреса")
        except FileNotFoundError as e:
            return self.err(404, str(e))
        except PermissionError:
            return self.err(403, "forbidden")
        except Exception as e:
            traceback.print_exc()
            return self.err(500, f"{type(e).__name__}: {e}")

    def do_PUT(self):
        return self.do_POST()

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(u.query).items()}
        path = u.path
        try:
            if path == "/api/upload":
                name = os.path.basename(unquote(q.get("name", "track.wav")))
                if not name.lower().endswith(A.AUDIO_EXT):
                    return self.err(400, "нужен аудиофайл: " + ", ".join(A.AUDIO_EXT))
                n = int(self.headers.get("Content-Length") or 0)
                if n <= 0:
                    return self.err(400, "пустой файл")

                def write(dst):
                    left = n
                    with open(dst, "wb") as f:
                        while left > 0:
                            b = self.rfile.read(min(1 << 20, left))
                            if not b:
                                raise IOError("загрузка оборвалась")
                            f.write(b)
                            left -= len(b)
                pid = P.create_project(name, write)
                return self.send_json(dict(id=pid))
            if path == "/api/import":   # local path (desktop use / Colab: file already on disk)
                b = self.body_json()
                src = (b.get("path") or "").strip().strip('"')
                if not os.path.isfile(src):
                    return self.err(404, f"файл не найден: {src}")
                pid = P.create_project(os.path.basename(src), lambda dst: shutil.copyfile(src, dst))
                return self.send_json(dict(id=pid))
            m = re.fullmatch(r"/api/projects/([\w\-]+)/run", path)
            if m:
                pid = m.group(1)
                j = JOBS.get(pid)
                if j and j.status in ("queued", "running"):
                    return self.err(409, "уже выполняется")
                opts = self.body_json()
                job = P.Job(pid, opts)
                JOBS[pid] = job
                man = P.load_manifest(pid)
                man["status"] = "queued"
                P.save_manifest(pid, man)
                QUEUE.put(job)
                return self.send_json(dict(ok=True, queued=QUEUE.qsize()))
            m = re.fullmatch(r"/api/projects/([\w\-]+)/cancel", path)
            if m:
                j = JOBS.get(m.group(1))
                if j:
                    j.cancel()
                return self.send_json(dict(ok=True))
            m = re.fullmatch(r"/api/projects/([\w\-]+)/rename", path)
            if m:
                man = P.load_manifest(m.group(1))
                man["name"] = (self.body_json().get("name") or man["name"]).strip()[:120]
                P.save_manifest(m.group(1), man)
                return self.send_json(dict(ok=True))
            m = re.fullmatch(r"/api/projects/([\w\-]+)/delete", path)
            if m:
                pid = m.group(1)
                j = JOBS.get(pid)
                if j and j.status in ("queued", "running"):
                    return self.err(409, "сначала отмените обработку")
                shutil.rmtree(P.project_dir(pid), ignore_errors=True)
                JOBS.pop(pid, None)
                return self.send_json(dict(ok=True))
            m = re.fullmatch(r"/api/projects/([\w\-]+)/open", path)
            if m:   # open the folder in Explorer / Finder (desktop only)
                d = P.project_dir(m.group(1))
                sub = self.body_json().get("sub")
                if sub:
                    d = safe_path(m.group(1), sub)
                    if os.path.isfile(d):
                        d = os.path.dirname(d)
                if os.name == "nt":
                    os.startfile(d)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", d])
                else:
                    subprocess.Popen(["xdg-open", d])
                return self.send_json(dict(ok=True))
            m = re.fullmatch(r"/api/models/(\w+)/download", path)
            if m:
                key = m.group(1)
                if key not in M.ALL:
                    return self.err(404, "нет такой модели")
                if DOWNLOADS.get(key, {}).get("status") == "running":
                    return self.send_json(dict(ok=True))
                st = DOWNLOADS[key] = dict(status="running", done=0, total=0, file="")

                def go():
                    try:
                        M.ensure(key, lambda d, t, fn: st.update(done=d, total=t, file=fn), lambda s: None)
                        st["status"] = "done"
                    except Exception as e:
                        st.update(status="error", error=str(e))
                threading.Thread(target=go, daemon=True).start()
                return self.send_json(dict(ok=True))
            return self.err(404, "нет такого адреса")
        except FileNotFoundError as e:
            return self.err(404, str(e))
        except Exception as e:
            traceback.print_exc()
            return self.err(500, f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    threading.Thread(target=worker, daemon=True).start()
    info = system_info()
    port = a.port
    for _ in range(20):
        try:
            srv = ThreadingHTTPServer((a.host, port), H)
            break
        except OSError:
            port += 1
    else:
        sys.exit("не нашёл свободный порт")
    srv.daemon_threads = True
    url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else a.host}:{port}/"
    print("=" * 64)
    print(f"  RESTORE STUDIO {VERSION}  ->  {url}")
    print(f"  устройство: {info['label']}")
    print(f"  проекты:    {P.PROJECTS}")
    print("  Закройте это окно, чтобы выключить студию.")
    print("=" * 64, flush=True)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
