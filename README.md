# Restore Studio

Get the parts back from a finished track. Made for musicians who lost the original project.
Drop in a mix or master (WAV, FLAC, MP3, OGG, AIFF) and the studio gives you everything you need to rebuild
the track in FL Studio, Ableton or any other DAW. Everything runs locally in your browser and the track
is never uploaded anywhere.

## What you get

- **Stems** (BS-RoFormer SW, 6 stems): vocals, drums, bass, guitar, keys, synths/other, plus an instrumental.
  The stems add up to the original, and every track is aligned to the grid: bar 1 is the start of the project.
- **Drum kit split** (MDX23C DrumSep): kick, snare, toms, hi-hat, ride, crash, plus General MIDI drum MIDI
  (FPC, GetGood Drums, SSD, Addictive Drums), a one-shot for each drum and groove loops.
- **MIDI for every part** on one shared grid: bass (with 808/sub, roots and tabs), guitar (rhythm/lead,
  chords, tabs), keys, synths and the vocal melody. The whole arrangement also comes as one multitrack MIDI file.
  You get both quantized and raw ("live") versions.
- **Loops** of 1, 4 and 8 bars at the track tempo (WAV plus MIDI of the same bars), full-mix loops without
  repeats, and one-shot samples of notes and chords.
- **Analysis**: BPM, key, Camelot / Open Key, LUFS, chords per bar, and where each instrument plays.
- **Export** as one ZIP: an FL Studio package, everything, or only stems / MIDI / loops.

## Installation

| System | Command |
|---|---|
| Windows 10/11 | `install.bat` installs everything into the program folder (~6 GB; internet is needed once) |
| Linux / macOS | `bash install.sh` |
| Google Colab | open `colab/RestoreStudio_Colab.ipynb` and paste this repository's URL in step 2 |

The installer chooses the right PyTorch build: NVIDIA → CUDA 12.6, AMD (Linux) → ROCm,
Apple Silicon → MPS, Intel Arc → XPU, anything else → CPU. Run `install.bat cpu` to force the CPU build.
It also creates a "Restore Studio" shortcut on the desktop and in the Start menu.

The neural network weights download into `models/` during installation. Any that are missing download
automatically the first time you process a track.

## Running

| Command | What it does |
|---|---|
| `RestoreStudio.bat` (Linux/macOS: `bash run.sh`) | Starts the web studio at http://127.0.0.1:8790/ |
| `restore.bat "song.flac" [--bpm 120]` | Processes a track from the command line, no browser. You can also drag a file onto it. |
| `check-gpu.bat` | Shows which device will be used, and why |

`restore.bat` options: `--bpm 120`, `--model sw6|rofo4`, `--quality high|normal|fast`, `--device auto|cuda|cpu`.
Results are saved to `projects/<date>_<track>/`.

In the studio, drop a track on the left, then choose a model, the parts to restore and a quality level,
and press **Restore**. The right side has tabs for Stems (a mixer with solo/mute/volume and A/B against the original),
MIDI (piano roll, preview, comparison with the stem), Loops & samples, Analysis & chords (reports, tabs)
and Export. Space starts and pauses playback.

If the grid is off, set the BPM (×2 / ÷2) or the time of the first downbeat and press **Recalculate**.
Separation comes from the cache, so recalculating takes seconds to a minute.

### Speed

For a 3-minute track, expect about 3 minutes on an RTX 4060 Ti and 15–25 minutes on an 8-core CPU.
You don't need a GPU. If video memory runs out, the batch size is reduced and then processing continues on the CPU.
CUDA needs NVIDIA driver 560 or newer.

## Google Colab

`colab/RestoreStudio_Colab.ipynb` runs the same web studio on a free Colab GPU.

1. Choose **Runtime → Change runtime type → T4 GPU**.
2. Run the cells in order:
   1. **Check GPU**
   2. **Code & Google Drive**: choose `git` and paste this repository's URL, or upload `restore-studio-app.zip`.
      With Drive storage on, projects and models are kept on Google Drive between sessions.
   3. **Install**: about 3–5 minutes, once per session.
   4. **Run studio**: opens the web interface.
3. Step 5 is optional. It processes a track without the interface and downloads the FL Studio package.

### Opening the studio through ngrok

In step 4, set `TUNNEL` to choose how you open the studio:

| `TUNNEL` | Link |
|---|---|
| `colab` | Colab's built-in proxy (works only in the browser where the notebook is open) |
| `ngrok` | A public `https://….ngrok-free.app` link that works in any browser or on a phone |
| `both` | Both links |

For ngrok:

1. Create a free account at https://ngrok.com and copy your authtoken from
   https://dashboard.ngrok.com/get-started/your-authtoken.
2. Paste the token into `NGROK_TOKEN`. Or, better, save it in Colab **Secrets** (the 🔑 icon on the left)
   as `NGROK_AUTHTOKEN` and leave the field empty.
3. Optional: `NGROK_DOMAIN` is your free static domain (for example `my-name.ngrok-free.app`), so the link
   stays the same between sessions.
4. Optional: `NGROK_PASSWORD` protects the link with basic auth (user `rs`, at least 8 characters).
   Without a password, anyone with the link can use the studio.

The ngrok link works only while the Colab session is running. On the first visit, ngrok shows a warning page.
Click **Visit Site** to continue. The free ngrok plan has a monthly traffic limit, and large uploads and stem
downloads count toward it. Also check Colab's usage rules for your plan before you use tunnels.

## Project layout

| Path | Contents |
|---|---|
| `app/` | Web server (`server.py`), CLI (`restore.py`), pipeline and separator (`studio/`), web UI (`web/`) |
| `engines/` | Part engines: drum-maker, bass-maker, guitar-splitter, piano-maker, bpm-getter |
| `tools/` | `patch_engines.py` (engine patches), `make_dist.py` (code ZIP), `make_colab.py` (builds the notebook) |
| `colab/` | Google Colab notebook (generated by `tools/make_colab.py`, so edit that script, not the `.ipynb`) |
| `install.bat`, `install.sh` | Installers |
| `RestoreStudio.bat`, `run.sh`, `restore.bat`, `check-gpu.bat` | Launchers |

These folders aren't stored in the repository. The installer creates them, or they're generated:
`python/` or `.venv/`, `models/`, `projects/` and `dist/`.

## Models

| Model | Author |
|---|---|
| BS-RoFormer SW (6 stems) | jarredou / undef13 (weights via huggingface.co/enerjazzer/BS-ROFO-SW-Fixed) |
| BS-RoFormer 4 stems FT | SYH99999 (Apache-2.0) |
| MDX23C DrumSep | aufr33 & jarredou |
| Mel-Band RoFormer Guitar | becruily |
| Basic Pitch (note transcription) | Spotify (Apache-2.0), via ONNX Runtime |

Network code: lucidrains BS-RoFormer, ZFTurbo Music-Source-Separation-Training (MIT).
See `THIRD_PARTY_NOTICES.txt` for details.

## Limitations

- The tempo must be constant (electronic music or tracks recorded to a click). For live playing without a click,
  use the raw MIDI (`midi_raw`).
- Transcription isn't perfect. Dense chords, distorted guitars and vocal effects will need fixing in the piano roll.
- Different synths inside the "other" stem aren't separated from each other, so they share one MIDI file.
