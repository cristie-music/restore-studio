"""Generates colab/RestoreStudio_Colab.ipynb (kept as code here so the notebook stays easy to edit)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MD_INTRO = """# 🎛️ Restore Studio — Google Colab

**Восстановление дорожек из готового трека:** стэмы (вокал, ударные, бас, гитара, клавиши, синты), MIDI каждой партии по сетке, кит ударных по частям, аккорды, табы, лупы и one-shot сэмплы, пакет для FL Studio.

Тот же веб-интерфейс, что и на компьютере — открывается в отдельной вкладке через прокси Colab или по публичной ссылке **ngrok** (шаг 4, нужен бесплатный токен с dashboard.ngrok.com).

**Перед запуском:** `Среда выполнения → Сменить среду выполнения → GPU (T4)`. Без GPU тоже работает, но в ~10 раз медленнее.

Шаги: **1** → **2** → **3** → **4** (по порядку, каждую ячейку кнопкой ▶). Шаг 5 — обработка без интерфейса."""

C_GPU = """#@title 1. Проверка GPU
import subprocess
r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                   capture_output=True, text=True)
print("GPU:", r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else
      "НЕТ — будет CPU (медленно). Среда выполнения → Сменить среду → T4 GPU")"""

C_CODE = """#@title 2. Код студии и Google Drive
#@markdown **Откуда взять код**: `git` — ваш репозиторий (вставьте ссылку в REPO_URL); `upload` / `drive` / `url` — архив `restore-studio-app.zip` (собирается `tools\\\\make_dist.py`):
CODE_SOURCE = "git"  #@param ["git", "upload", "drive", "url"]
REPO_URL = "https://github.com/USER/restore-studio"  #@param {type:"string"}
DRIVE_ZIP = "/content/drive/MyDrive/RestoreStudio/restore-studio-app.zip"  #@param {type:"string"}
CODE_URL = ""  #@param {type:"string"}
#@markdown **Хранить проекты и нейросети на Google Drive** (не пропадут после отключения Colab, модели не качаются заново):
USE_DRIVE_STORAGE = True  #@param {type:"boolean"}
import os, shutil, zipfile, urllib.request
RS = "/content/restore-studio"
if USE_DRIVE_STORAGE or CODE_SOURCE == "drive":
    from google.colab import drive
    drive.mount("/content/drive")
zip_path = "/content/restore-studio-app.zip"
if CODE_SOURCE == "git":
    import subprocess
    shutil.rmtree(RS, ignore_errors=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, RS], check=True)
elif CODE_SOURCE == "upload":
    from google.colab import files
    print("Выберите restore-studio-app.zip ...")
    up = files.upload()
    name = next(iter(up))
    if name != os.path.basename(zip_path):
        shutil.move(name, zip_path)
elif CODE_SOURCE == "drive":
    shutil.copy(DRIVE_ZIP, zip_path)
else:
    urllib.request.urlretrieve(CODE_URL, zip_path)
if CODE_SOURCE != "git":
    shutil.rmtree(RS, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RS)
STORE = "/content/drive/MyDrive/RestoreStudio" if USE_DRIVE_STORAGE else RS
os.makedirs(f"{STORE}/projects", exist_ok=True)
os.makedirs(f"{STORE}/models", exist_ok=True)
os.environ["RS_PROJECTS"] = f"{STORE}/projects"
os.environ["RS_MODELS"] = f"{STORE}/models"
print("Код:", RS, "| проекты:", os.environ["RS_PROJECTS"], "| нейросети:", os.environ["RS_MODELS"])"""

C_INSTALL = """#@title 3. Установка (≈3–5 минут, один раз за сессию)
#@markdown Отдельное окружение Python 3.11 (не трогает пакеты Colab), PyTorch под GPU Colab, нейросети.
import os, subprocess
RS = "/content/restore-studio"
VENV = "/content/rs-venv"
PY = f"{VENV}/bin/python"
def sh(cmd):
    print("$", cmd[:120]); r = subprocess.run(cmd, shell=True, cwd=RS)
    if r.returncode: raise SystemExit(f"ошибка: {cmd}")
sh("pip -q install uv")
if not os.path.exists(PY):
    sh(f"uv venv -q --python 3.11 {VENV}")
gpu = subprocess.run("nvidia-smi -L", shell=True, capture_output=True).returncode == 0
idx = "https://download.pytorch.org/whl/cu126" if gpu else "https://download.pytorch.org/whl/cpu"
sh(f"uv pip install -q --python {PY} torch==2.14.1 --index-url {idx}")
sh(f"uv pip install -q --python {PY} -r requirements.txt")
sh(f"uv pip install -q --python {PY} --no-deps basic-pitch==0.4.0")
sh(f"{PY} app/studio/models.py sw6 drumsep guitar")
sh(f"{PY} -W ignore -c \\"import sys; sys.path.insert(0,'app'); from studio import device as D; print('Устройство:', D.label(D.detect()))\\"")
print("\\n✅ Установлено")"""

C_RUN = """#@title 4. Запустить студию (веб-интерфейс)
#@markdown **Как открыть студию:** `colab` — встроенный прокси Colab (только в этом браузере); `ngrok` — постоянная ссылка `https://….ngrok-free.app`, открывается в любом браузере и на телефоне; `both` — обе ссылки.
TUNNEL = "colab"  #@param ["colab", "ngrok", "both"]
#@markdown Токен ngrok — https://dashboard.ngrok.com/get-started/your-authtoken . Можно оставить пустым и сохранить токен в секретах Colab (🔑 слева) под именем `NGROK_AUTHTOKEN`.
NGROK_TOKEN = ""  #@param {type:"string"}
#@markdown Ваш статический домен ngrok (необязательно, например `my-name.ngrok-free.app`):
NGROK_DOMAIN = ""  #@param {type:"string"}
#@markdown Пароль на ссылку ngrok (необязательно; логин `rs`). Без пароля студию откроет любой, у кого есть ссылка:
NGROK_PASSWORD = ""  #@param {type:"string"}
import os, sys, subprocess, time, urllib.request
RS = "/content/restore-studio"; PY = "/content/rs-venv/bin/python"; PORT = 8790
if "rs_server" in globals() and rs_server.poll() is None:
    rs_server.terminate(); time.sleep(1)
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1")
log = open("/content/restore-studio.log", "w")
rs_server = subprocess.Popen([PY, "-u", "app/server.py", "--no-browser", "--port", str(PORT)], cwd=RS, env=env,
                             stdout=log, stderr=subprocess.STDOUT)
for _ in range(60):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/system", timeout=2); break
    except Exception:
        time.sleep(1)
else:
    print(open("/content/restore-studio.log").read()); raise SystemExit("сервер не запустился")
print("✅ Студия запущена.")
if TUNNEL in ("ngrok", "both"):
    token = NGROK_TOKEN.strip()
    if not token:
        try:
            from google.colab import userdata
            token = (userdata.get("NGROK_AUTHTOKEN") or "").strip()
        except Exception:
            token = ""
    if not token:
        raise SystemExit("Нужен токен ngrok: вставьте его в NGROK_TOKEN или в секреты Colab (🔑) как NGROK_AUTHTOKEN")
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "pyngrok"], check=True)
    from pyngrok import ngrok
    ngrok.set_auth_token(token)
    for t in ngrok.get_tunnels():          # повторный запуск ячейки: закрыть старый туннель
        ngrok.disconnect(t.public_url)
    opts = {}
    if NGROK_DOMAIN.strip():
        opts["domain"] = NGROK_DOMAIN.strip().replace("https://", "").strip("/")
    if NGROK_PASSWORD:
        if len(NGROK_PASSWORD) < 8:
            raise SystemExit("ngrok требует пароль не короче 8 символов")
        opts["basic_auth"] = [f"rs:{NGROK_PASSWORD}"]
    rs_tunnel = ngrok.connect(PORT, "http", **opts)
    print("🌐 ngrok:", rs_tunnel.public_url, "(логин rs)" if NGROK_PASSWORD else "")
    print("   Ссылка работает, пока эта сессия Colab активна. На первой странице ngrok нажмите «Visit Site».")
if TUNNEL in ("colab", "both"):
    from google.colab import output
    print("Откройте студию по ссылке (лучше в отдельной вкладке):")
    output.serve_kernel_port_as_window(PORT, anchor_text="▶ Открыть Restore Studio")
    output.serve_kernel_port_as_iframe(PORT, height=900)"""

MD_CLI = """### 5. (необязательно) Обработка без интерфейса
Загрузите трек, получите ZIP с пакетом для FL Studio (стэмы + MIDI + лупы + сэмплы + отчёты)."""

C_CLI = """#@title 5. Восстановить трек и скачать пакет
BPM = 0  #@param {type:"number"}
MODEL = "sw6"  #@param ["sw6", "rofo4"]
QUALITY = "high"  #@param ["high", "normal", "fast"]
EXPORT = "fl"  #@param ["fl", "all", "stems", "midi", "loops"]
import os, subprocess, sys, json, glob
from google.colab import files
RS = "/content/restore-studio"; PY = "/content/rs-venv/bin/python"
up = files.upload()
src = os.path.abspath(next(iter(up)))
cmd = [PY, "-u", "app/restore.py", src, "--model", MODEL, "--quality", QUALITY] + (["--bpm", str(BPM)] if BPM else [])
env = dict(os.environ, PYTHONIOENCODING="utf-8")
p = subprocess.Popen(cmd, cwd=RS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="")
p.wait()
pid = sorted(os.listdir(os.environ["RS_PROJECTS"]))[-1]
code = f"import sys; sys.path.insert(0,'app'); import server; print(server.build_export({pid!r}, {EXPORT!r}))"
z = subprocess.run([PY, "-c", code], cwd=RS, env=env, capture_output=True, text=True).stdout.strip().splitlines()[-1]
print("Архив:", z)
files.download(z)"""


def cell_md(src):
    return dict(cell_type="markdown", metadata={}, source=src.splitlines(True))


def cell_code(src):
    return dict(cell_type="code", metadata={"cellView": "form"}, execution_count=None, outputs=[],
                source=src.splitlines(True))


nb = dict(nbformat=4, nbformat_minor=0,
          metadata=dict(colab=dict(provenance=[], gpuType="T4", name="RestoreStudio_Colab.ipynb"),
                        kernelspec=dict(name="python3", display_name="Python 3"), accelerator="GPU",
                        language_info=dict(name="python")),
          cells=[cell_md(MD_INTRO), cell_code(C_GPU), cell_code(C_CODE), cell_code(C_INSTALL), cell_code(C_RUN),
                 cell_md(MD_CLI), cell_code(C_CLI)])
os.makedirs(os.path.join(ROOT, "colab"), exist_ok=True)
dst = os.path.join(ROOT, "colab", "RestoreStudio_Colab.ipynb")
with open(dst, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(dst)
