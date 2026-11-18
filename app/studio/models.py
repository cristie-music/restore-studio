"""Model registry + downloader. All weights live in <studio>/models (override: env RS_MODELS)."""
import os
import shutil
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.environ.get("RS_MODELS") or os.path.join(ROOT, "models")

HF = "https://huggingface.co"
SEP_RELEASE = "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs"

# separation models for the first (main) split of the full mix
SEPARATORS = {
    "sw6": dict(
        title="BS-RoFormer SW · 6 стэмов (лучшее качество)",
        short="BS-RoFormer SW 6-stem",
        about="jarredou/undef13 BS-RoFormer SW: вокал, ударные, бас, гитара, пианино, остальное. "
              "Лучшая открытая 6-стэмовая модель (средн. SDR ≈ 10.3 дБ против 7.8 у htdemucs_6s).",
        files={"BS-Rofo-SW-Fixed.ckpt": [f"{HF}/enerjazzer/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.ckpt",
                                         f"{HF}/ChanTrail/BS-RoFormer/resolve/main/BS-Rofo-SW-Fixed.ckpt"],
               "BS-Rofo-SW-Fixed.yaml": [f"{HF}/ChanTrail/BS-RoFormer/resolve/main/BS-Rofo-SW-Fixed.yaml",
                                         f"{HF}/enerjazzer/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.yaml"]},
        ckpt="BS-Rofo-SW-Fixed.ckpt", config="BS-Rofo-SW-Fixed.yaml", size_mb=700),
    "rofo4": dict(
        title="BS-RoFormer 4 stems FT · 4 стэма (быстрее)",
        short="BS-RoFormer 4-stem FT",
        about="SYH99999 BS-RoFormer 4 stems FT: вокал, ударные, бас, остальное. Быстрее и легче, "
              "гитара и клавиши остаются вместе в «остальном».",
        files={"bs_roformer_4stems_ft.pth": [f"{HF}/SYH99999/bs_roformer_4stems_ft/resolve/main/bs_roformer_4stems_ft.pth"],
               "config_bs_roformer_4stems_ft.yaml": [f"{HF}/SYH99999/bs_roformer_4stems_ft/resolve/main/config.yaml"]},
        ckpt="bs_roformer_4stems_ft.pth", config="config_bs_roformer_4stems_ft.yaml", size_mb=530),
}

# models used by the engines (second stage)
ENGINE_MODELS = {
    "drumsep": dict(
        title="MDX23C DrumSep (бочка/малый/томы/хэт/райд/крэш)",
        files={"MDX23C-DrumSep-aufr33-jarredou.ckpt": [f"{SEP_RELEASE}/MDX23C-DrumSep-aufr33-jarredou.ckpt"],
               "config_drumsep_mdx23c.yaml": [f"{SEP_RELEASE}/config_drumsep_mdx23c.yaml"]}, size_mb=420),
    "guitar": dict(
        title="Mel-Band RoFormer Guitar (becruily)",
        files={"becruily_guitar.ckpt": [f"{HF}/becruily/mel-band-roformer-guitar/resolve/main/becruily_guitar.ckpt"],
               "config_guitar_becruily.yaml": [f"{HF}/becruily/mel-band-roformer-guitar/resolve/main/config_guitar_becruily.yaml"]},
        size_mb=45),
}

ALL = {**SEPARATORS, **ENGINE_MODELS}


def path(name):
    return os.path.join(MODELS, name)


def installed(key):
    return all(os.path.exists(path(f)) for f in ALL[key]["files"])


def status():
    return {k: dict(title=v["title"], installed=installed(k), size_mb=v.get("size_mb")) for k, v in ALL.items()}


def download(key, progress=None, log=print):
    """Fetches missing files of a model (tries mirrors in order). progress(done_bytes, total_bytes, filename)."""
    os.makedirs(MODELS, exist_ok=True)
    for fn, urls in ALL[key]["files"].items():
        dst = path(fn)
        if os.path.exists(dst):
            continue
        err = None
        for url in urls:
            try:
                log(f"  загрузка {fn} <- {url.split('/')[2]}")
                req = urllib.request.Request(url, headers={"User-Agent": "restore-studio/1.0"})
                with urllib.request.urlopen(req, timeout=60) as r, open(dst + ".part", "wb") as f:
                    total = int(r.headers.get("Content-Length") or 0)
                    done, t0 = 0, time.time()
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                        done += len(b)
                        if progress:
                            progress(done, total, fn)
                shutil.move(dst + ".part", dst)
                err = None
                break
            except Exception as e:  # next mirror
                err = e
                try:
                    os.remove(dst + ".part")
                except OSError:
                    pass
        if err:
            raise RuntimeError(f"не удалось скачать {fn}: {err}")


def ensure(key, progress=None, log=print):
    if not installed(key):
        download(key, progress, log)


if __name__ == "__main__":
    import sys
    keys = sys.argv[1:] or list(ALL)
    for k in keys:
        print(f"[{k}] {ALL[k]['title']}")

        def p(d, t, fn):
            print(f"\r    {fn}: {d / 2**20:.0f}/{t / 2**20:.0f} MB", end="", flush=True)
        ensure(k, p)
        print("\n    OK")
