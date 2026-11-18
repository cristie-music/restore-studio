"""Restore pipeline: full mix -> stems -> per-instrument MIDI / loops / one-shots / tabs on one common bar grid.

  1. read + analyse (tempo, key, loudness: bpm-getter engine)
  2. separate the mix in one pass (BS-RoFormer SW: vocals, drums, bass, guitar, piano, other)
  3. decide which stems really play (levels per second)
  4. engines, each as a subprocess on its stem; the first one that finds the grid becomes the master,
     all others take its grid (--grid-from) so every MIDI / loop lines up bar-for-bar in the DAW:
        drums  -> drum-maker      (MDX23C DrumSep: kick/snare/toms/hh/ride/crash + GM MIDI for GetGood Drums)
        bass   -> bass-maker      (mono bass MIDI, 808/sub, roots, tabs)
        guitar -> guitar-splitter (rhythm/lead, chords, tabs)
        piano  -> piano-maker keys
        other  -> piano-maker synth/pad/pluck/lead  (+ bass-maker on 'other' when the low end is a synth bass)
        vocals -> piano-maker lead (vocal melody as MIDI)
  5. aligned stems (moved like the MIDI: bar 1 = project start), full-mix loops, manifest for the web UI
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

from . import audio_io as A
from . import device as D
from . import models as M

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(APP)
ENGINES = os.path.join(ROOT, "engines")
PROJECTS = os.environ.get("RS_PROJECTS") or os.path.join(ROOT, "projects")
SR = A.SR

STEM_INFO = {
    "vocals": dict(title="Вокал", color="#ec4899", icon="mic"),
    "drums": dict(title="Ударные", color="#f97316", icon="drum"),
    "bass": dict(title="Бас", color="#8b5cf6", icon="bass"),
    "guitar": dict(title="Гитара", color="#eab308", icon="guitar"),
    "piano": dict(title="Клавишные", color="#3b82f6", icon="piano"),
    "other": dict(title="Синты / остальное", color="#10b981", icon="synth"),
    "instrumental": dict(title="Минусовка (без вокала)", color="#a1a1aa", icon="mix"),
}
STEM_ORDER = ["vocals", "drums", "bass", "guitar", "piano", "other", "instrumental"]

ENGINE_SCRIPTS = {"drum-maker": "drum_maker.py", "bass-maker": "bass_maker.py",
                  "guitar-splitter": "guitar_splitter.py", "piano-maker": "piano_maker.py"}

DEFAULT_OPTIONS = dict(
    separator="sw6", device="auto", quality="high",      # quality: high = overlap 4, normal = 2, fast = 1.2
    bpm=None, downbeat=None, quant="auto",
    instruments=dict(drums=True, bass=True, guitar=True, piano=True, other=True, vocals=True),
    bass_mode="bass", guitar_mode="guitar", other_mode="synth", vocal_mode="lead",
    fill_hats=False, synth_bass_from_other=True, mix_loops=True,
)


def new_id(name):
    base = re.sub(r"[^\w\-]+", "_", os.path.splitext(name)[0], flags=re.U).strip("_")[:40] or "track"
    return f"{time.strftime('%Y%m%d-%H%M%S')}_{base}"


def project_dir(pid):
    if not re.fullmatch(r"[\w\-]+", pid, flags=re.U):
        raise ValueError("bad project id")
    return os.path.join(PROJECTS, pid)


def load_manifest(pid):
    p = os.path.join(project_dir(pid), "project.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_manifest(pid, man):
    p = os.path.join(project_dir(pid), "project.json")
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    os.replace(p + ".tmp", p)


def create_project(src_name, write_fn):
    """write_fn(dst_path) stores the uploaded/copied file. -> project id."""
    pid = new_id(src_name)
    d = project_dir(pid)
    os.makedirs(d, exist_ok=True)
    ext = os.path.splitext(src_name)[1].lower() or ".wav"
    dst = os.path.join(d, "source" + ext)
    write_fn(dst)
    man = dict(id=pid, name=os.path.splitext(os.path.basename(src_name))[0], source="source" + ext,
               source_name=os.path.basename(src_name), created=time.strftime("%Y-%m-%d %H:%M:%S"),
               status="new", options=None, stems=[], engines={}, loops=[], analysis=None, grid=None)
    save_manifest(pid, man)
    return pid


class Cancelled(Exception):
    pass


class Job:
    """One pipeline run. Thread-safe state for the web UI: progress 0..1, step text, log lines."""

    def __init__(self, pid, options):
        self.pid = pid
        self.options = {**DEFAULT_OPTIONS, **(options or {})}
        self.options["instruments"] = {**DEFAULT_OPTIONS["instruments"], **(options or {}).get("instruments", {})}
        self.progress = 0.0
        self.step = "В очереди"
        self.steps = []          # [{id, title, state: wait|run|done|skip|error, note}]
        self.log = []
        self.status = "queued"   # queued | running | done | error | cancelled
        self.error = None
        self.started = self.finished = None
        self._cancel = False
        self._proc = None
        self._lock = threading.Lock()

    # ---- state helpers ---------------------------------------------------------------------------
    def say(self, msg):
        with self._lock:
            for line in str(msg).split("\n"):
                self.log.append(line)
            if len(self.log) > 5000:
                del self.log[:1000]
        print(msg, flush=True)

    def set_step(self, sid, state, note=None):
        for s in self.steps:
            if s["id"] == sid:
                s["state"] = state
                if note is not None:
                    s["note"] = note

    def cancel(self):
        self._cancel = True
        p = self._proc
        if p and p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass

    def check(self):
        if self._cancel:
            raise Cancelled()

    def state(self, since=0):
        with self._lock:
            return dict(pid=self.pid, status=self.status, progress=round(self.progress, 4), step=self.step,
                        steps=self.steps, error=self.error, log=self.log[since:], log_len=len(self.log),
                        started=self.started, finished=self.finished)

    # ---- engine subprocess -----------------------------------------------------------------------
    def run_engine(self, tool, wav, bpm, args, out, p0, p1, title):
        script = os.path.join(ENGINES, tool, "app", ENGINE_SCRIPTS[tool])
        cmd = [sys.executable, "-s", "-u", script, wav, f"{bpm:g}", "--out", out] + args
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1", RS_MODELS=M.MODELS,
                   RS_DEVICE=self.options.get("device") or "auto", PYTHONUTF8="1")
        env.pop("PYTHONPATH", None)
        self.say(f"$ {tool} {os.path.basename(wav)} {bpm:g} " + " ".join(a if " " not in a else f'"{a}"' for a in args))
        kw = {}
        if os.name == "nt":
            kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                      cwd=os.path.dirname(script), **kw)
        buf, last_line = b"", ""
        while True:
            ch = self._proc.stdout.read1(4096) if hasattr(self._proc.stdout, "read1") else self._proc.stdout.read(1)
            if not ch:
                break
            buf += ch
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                line = raw.decode("utf-8", "replace").rstrip()
                if not line:
                    continue
                m = re.match(r"\[(\d+)/(\d+)\]\s*(.*)", line)
                if m:
                    k, n = int(m.group(1)), int(m.group(2))
                    self.progress = p0 + (p1 - p0) * (k - 1) / n
                    self.step = f"{title}: {m.group(3)[:80]}"
                m2 = re.search(r"(\d+)/(\d+) \((\d+)%\)", line)
                if m2:   # \r progress of a separation inside the engine: status line only, log 50 / 100 %
                    self.step = f"{title}: {line.strip()[:90]}"
                    pct = int(m2.group(3))
                    if pct >= 100 or (pct >= 50 and last_line != "50"):
                        self.say("  " + line.strip())
                        last_line = "50" if pct < 100 else ""
                    continue
                self.say("  " + line)
            self.check()
        code = self._proc.wait()
        self._proc = None
        self.check()
        res = None
        for d in sorted(os.listdir(out)) if os.path.isdir(out) else []:
            if os.path.isdir(os.path.join(out, d)) and d not in ("_cache", "output") and \
                    any(f.endswith(".txt") for f in os.listdir(os.path.join(out, d))):
                res = os.path.join(out, d)
        return code, res

    # ---- the pipeline ----------------------------------------------------------------------------
    def run(self):
        self.status, self.started = "running", time.strftime("%Y-%m-%d %H:%M:%S")
        man = load_manifest(self.pid)
        man["status"], man["options"] = "running", self.options
        save_manifest(self.pid, man)
        try:
            self._run(man)
            self.status = "done"
            man["status"] = "done"
            self.progress, self.step = 1.0, "Готово"
        except Cancelled:
            self.status = man["status"] = "cancelled"
            self.step = "Отменено"
            self.say("\n! Отменено пользователем")
        except Exception as e:
            traceback.print_exc()
            self.status = man["status"] = "error"
            self.error = man["error"] = f"{type(e).__name__}: {e}"
            self.step = "Ошибка"
            self.say("\n! ОШИБКА: " + self.error)
            self.say(traceback.format_exc())
        self.finished = man["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        man["log_tail"] = self.log[-400:]
        save_manifest(self.pid, man)

    def _run(self, man):
        o = self.options
        pdir = project_dir(self.pid)
        src = os.path.join(pdir, man["source"])
        sep_key = o["separator"]
        inst = o["instruments"]
        self.steps = [dict(id="load", title="Чтение трека", state="wait"),
                      dict(id="analyze", title="Темп и тональность", state="wait"),
                      dict(id="separate", title=f"Разделение на стэмы ({M.SEPARATORS[sep_key]['short']})", state="wait")]
        slots = [("drums", "Ударные → MIDI, кит по частям"), ("bass", "Бас → MIDI, табы"),
                 ("piano", "Клавишные → MIDI, аккорды"), ("guitar", "Гитара → MIDI, аккорды, табы"),
                 ("other", "Синты → MIDI"), ("synthbass", "Синт-бас из «остального»"),
                 ("vocals", "Мелодия вокала → MIDI")]
        self.steps += [dict(id=s, title=t, state="wait") for s, t in slots]
        self.steps.append(dict(id="finish", title="Выравнивание, лупы, экспорт", state="wait"))

        # 1. load ----------------------------------------------------------------------------------
        self.set_step("load", "run")
        self.step = "Чтение трека"
        self.say(f"=== RESTORE STUDIO: {man['source_name']}")
        dinfo = D.detect(o["device"])
        self.say(f"Устройство: {D.label(dinfo)}" + (f"  ({dinfo['warning']})" if dinfo.get("warning") else ""))
        x, sr_in = A.load_stereo(src)
        dur = len(x) / SR
        man.update(duration=round(dur, 3), sr_in=sr_in)
        self.say(f"Длительность {dur:.1f} с, частота {sr_in} Гц")
        self.set_step("load", "done", f"{dur:.0f} с")
        self.progress = 0.02
        self.check()

        # 2. analysis ------------------------------------------------------------------------------
        self.set_step("analyze", "run")
        self.step = "Темп и тональность"
        an_path = os.path.join(pdir, "analysis.json")
        if os.path.exists(an_path):
            an = json.load(open(an_path, encoding="utf-8"))
        else:
            sys.path.insert(0, os.path.join(ENGINES, "bpm-getter", "app"))
            import analyzer
            an = analyzer.analyze(src, lambda msg, p: setattr(self, "step", f"Анализ: {msg}"))
            json.dump(an, open(an_path, "w", encoding="utf-8"), ensure_ascii=False)
        bpm_auto = float(an["tempo"]["bpm"])
        bpm = float(o["bpm"]) if o.get("bpm") else bpm_auto
        man["analysis"] = dict(bpm=bpm_auto, bpm_used=bpm, key=an["key"].get("name"), camelot=an["key"].get("camelot"),
                               open_key=an["key"].get("open_key"), lufs=an["loudness"].get("lufs"),
                               true_peak=an["loudness"].get("true_peak_db", an["loudness"].get("true_peak")),
                               tempo_alts=an["tempo"].get("alternatives") or an["tempo"].get("candidates"),
                               key_conf=an["key"].get("confidence"), relative=an["key"].get("relative"),
                               compatible=an["key"].get("compatible"), chroma=an["key"].get("chroma"),
                               chroma_labels=an["key"].get("chroma_labels"), bpm_conf=an["tempo"].get("confidence"))
        self.say(f"Темп {bpm_auto:g} BPM (используется {bpm:g}), тональность {man['analysis']['key']} "
                 f"({man['analysis']['camelot']}), громкость {man['analysis']['lufs']} LUFS")
        self.set_step("analyze", "done", f"{bpm:g} BPM · {man['analysis']['key']}")
        self.progress = 0.06
        save_manifest(self.pid, man)
        self.check()

        # 3. separation (cached per model + quality) -----------------------------------------------
        self.set_step("separate", "run")
        self.step = "Разделение на стэмы"
        sdir = os.path.join(pdir, "stems")
        stamp_p = os.path.join(sdir, "_separation.json")
        stamp = dict(model=sep_key, quality=o["quality"])
        stems = {}
        if os.path.exists(stamp_p) and json.load(open(stamp_p)).get("model") == sep_key and \
                json.load(open(stamp_p)).get("quality") == o["quality"]:
            self.say("Стэмы уже были разделены этой моделью - беру готовые")
            for f in os.listdir(sdir):
                if f.endswith(".wav"):
                    stems[f[:-4]] = A.read_any(os.path.join(sdir, f))[0]
            sep_info = json.load(open(stamp_p)).get("info", {})
        else:
            def mprog(d, t, fn):
                self.step = f"Загрузка модели {fn}: {d / 2**20:.0f}/{t / 2**20:.0f} МБ"
            M.ensure(sep_key, mprog, self.say)
            from . import separator
            overlap = {"high": 4, "normal": 2, "fast": 1.2}.get(o["quality"], 2)
            if dinfo["kind"] == "cpu" and overlap > 2:
                overlap = 2
                self.say("  на CPU «Максимум» = перекрытие ×2 (×4 заняло бы вдвое дольше почти без выигрыша)")
            self.say(f"Разделение: {M.SEPARATORS[sep_key]['title']}, перекрытие окон {overlap}")

            def sprog(f, msg):
                self.progress = 0.06 + 0.22 * f
                self.step = f"Разделение на стэмы: {msg}"
                self.check()
            raw, sep_info = separator.separate(x.T, sep_key, o["device"], overlap, sprog, self.say)
            shutil.rmtree(sdir, ignore_errors=True)
            for k, v in raw.items():
                stems[k] = v.T
            if "vocals" in stems:
                stems["instrumental"] = x - stems["vocals"]
            for k, v in stems.items():
                A.write_wav(os.path.join(sdir, f"{k}.wav"), v)
            json.dump({**stamp, "info": sep_info}, open(stamp_p, "w"))
            self.say(f"Разделено за {sep_info['seconds']} с ({sep_info['device']})")
        man["separator"] = dict(key=sep_key, **sep_info)
        self.progress = 0.28
        self.check()

        # 4. which stems play --------------------------------------------------------------------
        mix_db = A.rms_db(x)
        stem_meta = {}
        for k in STEM_ORDER:
            if k not in stems:
                continue
            act, lv = A.activity(stems[k], ref_db=mix_db + 12, rel_db=-38)
            stem_meta[k] = dict(id=k, **STEM_INFO[k], level_db=round(A.rms_db(stems[k]), 1),
                                active_s=int(act.sum()), active=bool(act.sum() >= 4), levels=lv)
            self.say(f"  {STEM_INFO[k]['title']:<24} уровень {stem_meta[k]['level_db']:6.1f} дБ, "
                     f"звучит {stem_meta[k]['active_s']:4d} с" + ("" if stem_meta[k]["active"] else "  -> пусто"))
        low_other = None
        if "other" in stems:
            from scipy.signal import butter, sosfiltfilt
            sos = butter(4, 150, "lowpass", fs=SR, output="sos")
            lo = sosfiltfilt(sos, stems["other"].mean(1)[::1]).astype(np.float32)
            lact, _ = A.activity(lo, ref_db=mix_db + 12, rel_db=-36)
            low_other = int(lact.sum())
            self.say(f"  низкие частоты (<150 Гц) в «остальном» звучат {low_other} с")
        self.set_step("separate", "done", f"{sum(1 for v in stem_meta.values() if v['active'])} стэмов со звуком")

        # 5. engines --------------------------------------------------------------------------------
        edir = os.path.join(pdir, "engines")
        engines = {}
        master = None
        bass_empty = not stem_meta.get("bass", {}).get("active")
        plan = []

        def add(slot, stem, tool, args, title):
            plan.append(dict(slot=slot, stem=stem, tool=tool, args=args, title=title))

        q = ["--quant", o["quant"]] if o.get("quant") and o["quant"] != "auto" else []
        if inst.get("drums"):
            fast = dinfo["kind"] == "cpu" or o["quality"] == "fast"
            add("drums", "drums", "drum-maker", q + (["--fill-hats"] if o.get("fill_hats") else [])
                + (["--fast"] if fast else []), "Ударные")
        if inst.get("bass"):
            add("bass", "bass", "bass-maker", ["--no-separate", "--mode", o["bass_mode"]] + q, "Бас")
        if inst.get("piano"):
            add("piano", "piano", "piano-maker", ["--mode", "keys"] + q, "Клавишные")
        if inst.get("guitar"):
            if "guitar" in stems:
                add("guitar", "guitar", "guitar-splitter", ["--no-separate", "--mode", o["guitar_mode"]] + q, "Гитара")
            elif "other" in stems:  # 4-stem model: pull the guitar out of 'other' with the guitar model
                add("guitar", "other", "guitar-splitter", ["--mode", o["guitar_mode"]] + q, "Гитара")
        if inst.get("other"):
            add("other", "other", "piano-maker", ["--mode", o["other_mode"]] + q, "Синты / остальное")
        if inst.get("bass") and o.get("synth_bass_from_other") and bass_empty and (low_other or 0) >= 8:
            add("synthbass", "other", "bass-maker", ["--mode", "synth"] + q, "Синт-бас")
        if inst.get("vocals"):
            add("vocals", "vocals", "piano-maker", ["--mode", o["vocal_mode"]] + q, "Мелодия вокала")
        planned = {p["slot"] for p in plan}
        for s in self.steps:
            if s["id"] in ("drums", "bass", "piano", "guitar", "other", "synthbass", "vocals") and s["id"] not in planned:
                s["state"], s["note"] = "skip", "выключено" if s["id"] != "synthbass" else "не нужно"
        if any(p["tool"] == "guitar-splitter" and "--no-separate" not in p["args"] for p in plan):
            M.ensure("guitar", None, self.say)
        if any(p["tool"] == "drum-maker" for p in plan):
            M.ensure("drumsep", None, self.say)
        if any(p["tool"] == "bass-maker" and "--no-separate" not in p["args"] for p in plan):
            M.ensure("rofo4", None, self.say)

        n_plan = max(1, len(plan))
        for i, p in enumerate(plan):
            self.check()
            p0, p1 = 0.30 + 0.60 * i / n_plan, 0.30 + 0.60 * (i + 1) / n_plan
            self.progress = p0
            slot, stem = p["slot"], p["stem"]
            meta = stem_meta.get(stem)
            if meta is None or not meta["active"]:
                self.set_step(slot, "skip", "в треке нет этого инструмента")
                engines[slot] = dict(status="skip", title=p["title"], stem=stem, note="стэм пустой")
                self.say(f"\n--- {p['title']}: пропуск (стэм пустой)")
                continue
            self.set_step(slot, "run")
            self.step = p["title"]
            self.say(f"\n--- {p['title']} ({p['tool']})")
            out = os.path.join(edir, slot)
            shutil.rmtree(out, ignore_errors=True)
            os.makedirs(out, exist_ok=True)
            args = list(p["args"])
            if master:
                args += ["--grid-from", master]
            elif o.get("downbeat") is not None:
                args += ["--downbeat", f"{float(o['downbeat']):g}"]
            wav = os.path.join(sdir, f"{stem}.wav")
            t0 = time.time()
            code, res = self.run_engine(p["tool"], wav, bpm, args, out, p0, p1, p["title"])
            if code != 0 or not res:
                tail = [line for line in self.log[-6:] if line.strip()]
                note = tail[-1].strip() if tail else f"код {code}"
                self.set_step(slot, "error" if code not in (0, 1) else "skip", note[:140])
                engines[slot] = dict(status="fail", title=p["title"], stem=stem, tool=p["tool"], note=note)
                self.say(f"  ! {p['title']}: не получилось ({note})")
                continue
            engines[slot] = dict(status="done", title=p["title"], stem=stem, tool=p["tool"],
                                 dir=os.path.relpath(res, pdir).replace("\\", "/"), seconds=round(time.time() - t0, 1))
            g = read_grid(res)
            if g and not master:
                master = res
                man["grid"] = dict(g, master=slot)
                self.say(f"  сетка проекта: {g['bpm']:.2f} BPM, первая доля {g['bar_t0'] * 1000:.0f} мс, "
                         f"сдвиг {g['shift_s'] * 1000:+.0f} мс (мастер: {p['title']})")
            self.set_step(slot, "done", f"{time.time() - t0:.0f} с")
            man["engines"] = engines
            save_manifest(self.pid, man)
        man["engines"] = engines

        # 6. aligned stems, previews, mix loops ---------------------------------------------------
        self.check()
        self.set_step("finish", "run")
        self.step = "Выравнивание стэмов и превью"
        self.progress = 0.91
        g = man.get("grid") or dict(bpm=bpm, bar_t0=0.0, shift_s=0.0, master=None)
        if not man.get("grid"):
            man["grid"] = g
        sh = float(g["shift_s"])
        adir = os.path.join(pdir, "aligned")
        pvdir = os.path.join(pdir, "preview")
        shutil.rmtree(adir, ignore_errors=True)
        shutil.rmtree(pvdir, ignore_errors=True)
        mix_al = A.shift(x, sh)
        A.write_wav(os.path.join(adir, "00_full_mix.wav"), mix_al)
        A.write_mp3(os.path.join(pvdir, "mix.mp3"), mix_al)
        out_stems = []
        for n, k in enumerate(STEM_ORDER, 1):
            if k not in stems:
                continue
            y = A.shift(stems[k], sh)
            fn = f"{n:02d}_{k}.wav"
            A.write_wav(os.path.join(adir, fn), y)
            A.write_mp3(os.path.join(pvdir, f"{k}.mp3"), y)
            m = stem_meta[k]
            m.update(file=f"aligned/{fn}", raw=f"stems/{k}.wav", preview=f"preview/{k}.mp3", peaks=A.peaks(y, 1200))
            out_stems.append(m)
        man["stems"] = out_stems
        man["mix"] = dict(file="aligned/00_full_mix.wav", preview="preview/mix.mp3", peaks=A.peaks(mix_al, 1200),
                          source=man["source"])
        man["duration_aligned"] = round(len(mix_al) / SR, 3)
        self.say(f"\nСтэмы выровнены по сетке (сдвиг {sh * 1000:+.0f} мс): aligned\\")

        if o.get("mix_loops"):
            self.step = "Лупы из полного микса"
            man["mix_loops"] = mix_loops(pdir, mix_al, {k: A.shift(stems[k], sh) for k in stems
                                                        if stem_meta[k]["active"] and k != "instrumental"},
                                         g, self.say)
        # chords / key from the melodic engines
        man["chords"] = None
        tracks = {}
        for slot in ("piano", "guitar", "other"):
            e = engines.get(slot)
            if e and e["status"] == "done":
                gj = os.path.join(pdir, e["dir"], "_cache", "grid.json")
                if os.path.exists(gj):
                    d = json.load(open(gj, encoding="utf-8"))
                    if d.get("chords"):
                        tracks[slot] = dict(source=slot, key=d.get("key"), items=d["chords"])
        if tracks:   # the part with the richest harmony is the main chord chart, the others are kept too
            best = max(tracks.values(), key=lambda t: len({c["name"] for c in t["items"]}) * 1000 + len(t["items"]))
            man["chords"] = dict(best, all=tracks)
        for e in engines.values():   # engine caches (float WAVs of their separations) are big, not needed any more
            if e.get("status") == "done":
                edone = os.path.join(pdir, e["dir"])
                for f in os.listdir(edone):   # listening previews of the MIDI: MP3 is plenty
                    if f.startswith("preview_") and f.endswith(".wav"):
                        try:
                            A.write_mp3(os.path.join(edone, f[:-4] + ".mp3"), A.read_any(os.path.join(edone, f))[0])
                            os.remove(os.path.join(edone, f))
                        except Exception as ex:
                            self.say(f"  ! превью {f}: {ex}")
        for root, _, files in os.walk(edir):   # every engine's _cache (also of failed runs): drop the big WAVs
            if "_cache" in root.replace("\\", "/").split("/"):
                for f in files:
                    if f.endswith(".wav"):
                        try:
                            os.remove(os.path.join(root, f))
                        except OSError:
                            pass
        man["files"] = index_files(pdir, engines)
        try:   # one multitrack MIDI with every restored part (drag into FL -> all parts on their own channels)
            from . import midi_tools
            parts = []
            for slot in ("drums", "bass", "synthbass", "piano", "guitar", "other", "vocals"):
                e = engines.get(slot)
                if e and e.get("status") == "done":
                    f = midi_tools.pick_main(man["files"][slot]["midi"], e["tool"])
                    if f:
                        parts.append((slot, os.path.join(pdir, f)))
            if parts:
                midi_tools.merge(parts, float(g["bpm"]), os.path.join(pdir, "export", "full_arrangement.mid"))
                man["arrangement_midi"] = "export/full_arrangement.mid"
                self.say(f"Вся аранжировка одним MIDI ({len(parts)} дорожек): export\\full_arrangement.mid")
        except Exception as e:
            self.say(f"  ! общий MIDI не собран: {e}")
        self.set_step("finish", "done")
        n_midi = sum(len(v["midi"]) for v in man["files"].values())
        n_loops = sum(len(v["loops"]) for v in man["files"].values())
        self.say(f"\nГОТОВО: {len(out_stems)} стэмов, {n_midi} MIDI, {n_loops} лупов. Папка проекта:\n  {pdir}")
        save_manifest(self.pid, man)


def read_grid(res):
    for fn in ("grid.json", "notes.json"):
        p = os.path.join(res, "_cache", fn)
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            return dict(bpm=float(d["bpm"]), bar_t0=float(d["bar_t0"]), shift_s=float(d["shift_s"]),
                        grid_err_ms=d.get("grid_err_ms"))
    return None


def mix_loops(pdir, mix, stems, g, say):
    """4- and 8-bar loops of the whole aligned mix (and every playing stem) where the music plays.
    Identical phrases (repeats of a chorus etc.) are kept once: similarity of the chroma/energy fingerprint."""
    bar = 240.0 / g["bpm"]
    out = os.path.join(pdir, "loops_mix")
    shutil.rmtree(out, ignore_errors=True)
    nb = int(len(mix) / SR / bar)
    res = []
    for length in (4, 8):
        prints = []
        for b0 in range(0, nb - length + 1, length):
            a, b = int(round(b0 * bar * SR)), int(round((b0 + length) * bar * SR))
            seg = mix[a:b]
            if len(seg) < b - a or A.rms_db(seg) < -40:
                continue
            fp = _fingerprint(seg)
            if any(np.dot(fp, p) > 0.985 for p in prints):
                continue
            prints.append(fp)
            name = f"{length}bars_{b0 + 1:03d}-{b0 + length:03d}"
            d = os.path.join(out, name)
            A.write_wav(os.path.join(d, f"mix_{name}_{g['bpm']:g}bpm.wav"), seg)
            files = [f"loops_mix/{name}/mix_{name}_{g['bpm']:g}bpm.wav"]
            for k, y in stems.items():
                s = y[a:b]
                if len(s) == b - a and A.rms_db(s) > -45:
                    fn = f"{k}_{name}_{g['bpm']:g}bpm.wav"
                    A.write_wav(os.path.join(d, fn), s)
                    files.append(f"loops_mix/{name}/{fn}")
            res.append(dict(name=name, bars=length, bar_from=b0 + 1, bar_to=b0 + length, files=files))
    say(f"Лупы из микса: {len(res)} (4 и 8 тактов, без повторов) -> loops_mix\\")
    return res


def _fingerprint(seg):
    m = seg.mean(1)
    n = 64
    k = len(m) // n
    e = np.sqrt(np.mean(m[:k * n].reshape(n, k) ** 2, 1))
    spec = np.abs(np.fft.rfft(m[: 1 << 15] if len(m) >= 1 << 15 else m, 4096))[:400]
    v = np.concatenate([e / (e.max() + 1e-9), spec / (spec.max() + 1e-9)])
    return v / (np.linalg.norm(v) + 1e-9)


CATS = (("midi", "midi"), ("midi_raw", "midi_raw"), ("stems", "stems"), ("loops", "loops"), ("oneshots", "oneshots"),
        ("aligned", "stems"))


def index_files(pdir, engines):
    """Engine output folder -> lists for the UI (paths relative to the project)."""
    out = {}
    for slot, e in engines.items():
        if e.get("status") != "done":
            continue
        base = os.path.join(pdir, e["dir"])
        d = dict(midi=[], midi_raw=[], stems=[], loops=[], oneshots=[], docs=[], previews=[])
        for root, dirs, files in os.walk(base):
            dirs[:] = [x for x in dirs if x != "_cache"]
            rel_root = os.path.relpath(root, base).replace("\\", "/")
            top = rel_root.split("/")[0]
            for f in sorted(files):
                rel = os.path.relpath(os.path.join(root, f), pdir).replace("\\", "/")
                low = f.lower()
                if low.endswith(".peaks.json"):
                    continue
                if rel_root == ".":
                    if low.endswith(".txt"):
                        d["docs"].append(rel)
                    elif low.endswith((".wav", ".mp3")):
                        d["previews"].append(rel)
                    continue
                cat = {"midi": "midi", "midi_raw": "midi_raw", "stems": "stems", "aligned": "stems",
                       "loops": "loops", "oneshots": "oneshots"}.get(top)
                if cat == "loops":
                    if low.endswith((".wav", ".mid")):
                        d["loops"].append(rel)
                elif cat and low.endswith((".wav", ".mid")):
                    d[cat].append(rel)
                elif low.endswith(".txt"):
                    d["docs"].append(rel)
        out[slot] = d
    return out
