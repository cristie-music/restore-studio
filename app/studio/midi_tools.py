"""MIDI helpers: notes for the browser piano roll, and one multitrack MIDI with every restored part."""
import os

import mido

GM_PROGRAM = {"bass": 33, "synthbass": 38, "guitar": 27, "piano": 0, "other": 81, "vocals": 53}
TRACK_NAME = {"drums": "Drums (GM)", "bass": "Bass", "synthbass": "Synth Bass", "guitar": "Guitar", "piano": "Keys",
              "other": "Synths", "vocals": "Vocal melody"}
# which MIDI of each engine goes into the full arrangement
MAIN_MIDI = {"drum-maker": "00_ALL", "bass-maker": "00_bass", "guitar-splitter": "00_ALL", "piano-maker": "00_ALL"}


def notes(path):
    """-> dict(bpm, ticks, length_beats, tracks=[{name, channel, notes:[[start_beat, dur_beats, pitch, vel], ...]}])"""
    mf = mido.MidiFile(path)
    tpb = mf.ticks_per_beat
    tempo = 500000
    out = []
    for tr in mf.tracks:
        t, on, ns, name, ch_used = 0, {}, [], tr.name or "", None
        for msg in tr:
            t += msg.time
            if msg.type == "set_tempo" and tempo == 500000:
                tempo = msg.tempo
            elif msg.type == "track_name":
                name = msg.name
            elif msg.type == "note_on" and msg.velocity > 0:
                on[(msg.channel, msg.note)] = (t, msg.velocity)
                ch_used = msg.channel
            elif msg.type in ("note_off", "note_on"):
                k = (msg.channel, msg.note)
                if k in on:
                    t0, v = on.pop(k)
                    ns.append([round(t0 / tpb, 4), round(max(1, t - t0) / tpb, 4), msg.note, v])
        if ns:
            ns.sort()
            out.append(dict(name=name, channel=ch_used, notes=ns))
    length = max((n[0] + n[1] for tr in out for n in tr["notes"]), default=0)
    return dict(bpm=round(mido.tempo2bpm(tempo), 3), length_beats=length, tracks=out)


def pick_main(files, tool):
    pref = MAIN_MIDI.get(tool, "00_")
    for f in files:
        if os.path.basename(f).startswith(pref) and f.endswith(".mid"):
            return f
    mids = [f for f in files if f.endswith(".mid")]
    return mids[0] if mids else None


def merge(parts, bpm, dst):
    """parts: list of (slot, path). Every engine wrote its MIDI on the same grid -> just rescale ticks and stack."""
    tpb = 960
    mf = mido.MidiFile(type=1, ticks_per_beat=tpb)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="Restore Studio", time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    mf.tracks.append(meta)
    chan = 0
    for slot, path in parts:
        src = mido.MidiFile(path)
        k = tpb / src.ticks_per_beat
        ev = []
        for tr in src.tracks:
            t = 0
            for msg in tr:
                t += msg.time
                if msg.type in ("note_on", "note_off"):
                    ev.append((round(t * k), msg))
        if not ev:
            continue
        drums = slot == "drums"
        ch = 9 if drums else chan
        if not drums:
            chan += 1
            if chan == 9:
                chan = 10
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=TRACK_NAME.get(slot, slot), time=0))
        if not drums:
            tr.append(mido.Message("program_change", channel=ch, program=GM_PROGRAM.get(slot, 0), time=0))
        ev.sort(key=lambda e: (e[0], e[1].type == "note_on"))
        last = 0
        for t, msg in ev:
            tr.append(msg.copy(channel=ch, time=t - last))
            last = t
        mf.tracks.append(tr)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    mf.save(dst)
    return dst
