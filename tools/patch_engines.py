"""Applies Restore Studio patches to the vendored engines (engines/*/app). Idempotent: safe to run again
after re-copying the engines from the stand-alone tools.

  * MODELS dir from env RS_MODELS (one shared models\\ folder for the whole studio)
  * device selection: env RS_DEVICE + Apple MPS / Intel XPU besides CUDA
  * guitar-splitter: --no-separate (input is already a clean guitar stem from the 6-stem separation)
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENG = os.path.join(ROOT, "engines")

DEVICE_PATCH = '''    want = os.environ.get("RS_DEVICE") or want  # Restore Studio: device chosen in the web UI
    if want in ("auto", "mps", "gpu") and not torch.cuda.is_available() \\
            and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps"), "GPU Apple Silicon (MPS)"
    if want in ("auto", "xpu", "gpu") and not torch.cuda.is_available() and hasattr(torch, "xpu") \\
            and torch.xpu.is_available():
        return torch.device("xpu", 0), f"GPU Intel {torch.xpu.get_device_name(0)} (XPU)"
    if want == "cpu":
        return torch.device("cpu"), f"CPU ({os.cpu_count()} threads)"
'''


def patch(path, fn):
    s = open(path, encoding="utf-8").read()
    t = fn(s)
    if t != s:
        open(path, "w", encoding="utf-8", newline="\n").write(t)
        print("patched", os.path.relpath(path, ROOT))


def models_env(s):
    return s.replace('MODELS = os.path.join(ROOT, "models")',
                     'MODELS = os.environ.get("RS_MODELS") or os.path.join(ROOT, "models")')


def device(s):
    if "RS_DEVICE" in s:
        return s
    return re.sub(r'(def pick_device\(want="auto"\):\n    """[^\n]*"""\n)', lambda m: m.group(1) + DEVICE_PATCH, s)


def guitar_no_sep(s):
    if "--no-separate" not in s:
        s = s.replace('    ap.add_argument("--no-split", action="store_true")\n',
                      '    ap.add_argument("--no-split", action="store_true")\n'
                      '    ap.add_argument("--no-separate", action="store_true")\n')
    s = s.replace('    say("[1/6] Чтение и отделение гитары (нейросеть Mel-Band RoFormer Guitar)")\n',
                  '    say("[1/6] Чтение и отделение гитары" + (" - пропущено (--no-separate): вход уже чистая гитара"\n'
                  '                                            if a.no_separate else " (нейросеть Mel-Band RoFormer Guitar)"))\n')
    s = s.replace("    g_raw = separate_cached(wav, x, out, a.device)\n",
                  "    g_raw = x.copy() if a.no_separate else separate_cached(wav, x, out, a.device)\n")
    return s


def bass_cache(s):
    """bass-maker keeps its separation cache in <tool>\\output; in the studio it goes to the --out folder of the run."""
    return s.replace('separate_cached(wav, x, os.path.join(ROOT, "output"), a.device',
                     'separate_cached(wav, x, a.out, a.device')


def main():
    for tool in ("bass-maker", "drum-maker", "guitar-splitter", "piano-maker"):
        app = os.path.join(ENG, tool, "app")
        for f in os.listdir(app):
            p = os.path.join(app, f)
            if f.endswith(".py"):
                patch(p, models_env)
                if f in ("roformer.py", "mdx23c.py"):
                    patch(p, device)
    patch(os.path.join(ENG, "guitar-splitter", "app", "guitar_splitter.py"), guitar_no_sep)
    patch(os.path.join(ENG, "bass-maker", "app", "bass_maker.py"), bass_cache)
    s = open(os.path.join(ENG, "guitar-splitter", "app", "guitar_splitter.py"), encoding="utf-8").read()
    if "a.no_separate" not in s:
        sys.exit("guitar --no-separate patch did not apply")


if __name__ == "__main__":
    main()
