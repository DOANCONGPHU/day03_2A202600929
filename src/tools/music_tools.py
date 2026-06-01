import json
import math
import os
import struct
import wave
from array import array
from typing import Any, Dict, Iterable, List, Tuple


OUTPUT_DIR = "outputs"
TICKS_PER_BEAT = 480
SAMPLE_RATE = 44100

NOTE_OFFSETS = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}


SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "blues": [0, 3, 5, 6, 7, 10],
}


MOODS = {
    "happy": {"scale": "major", "tempo": 128, "velocity": 92},
    "sad": {"scale": "minor", "tempo": 76, "velocity": 74},
    "calm": {"scale": "pentatonic", "tempo": 88, "velocity": 68},
    "epic": {"scale": "minor", "tempo": 112, "velocity": 100},
    "energetic": {"scale": "minor", "tempo": 120, "velocity": 104},
    "drill": {"scale": "minor", "tempo": 120, "velocity": 108},
    "dark": {"scale": "minor", "tempo": 96, "velocity": 92},
    "lofi": {"scale": "minor", "tempo": 82, "velocity": 70},
}


def create_midi(
    title: str = "ai_music",
    mood: str = "calm",
    key: str = "C",
    tempo: int = None,
    bars: int = 4,
) -> str:
    """
    Create a MIDI file from a compact musical brief.

    Args:
        title: File-safe song title.
        mood: One of happy, sad, calm, epic, energetic, drill, dark, lofi.
        key: Musical key such as C, D, F#, Bb.
        tempo: BPM. If omitted, a mood default is used.
        bars: Number of 4/4 bars to generate.

    Returns:
        Path to the generated .mid file.
    """
    spec = _normalize_spec(title, mood, key, tempo, bars)
    midi_path = os.path.join(OUTPUT_DIR, f"{spec['title']}.mid")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    melody, chords = _compose_song(spec)
    track = bytearray()
    track.extend(_delta(0) + b"\xff\x51\x03" + _tempo_bytes(spec["tempo"]))
    track.extend(_delta(0) + b"\xc0\x00")

    current_tick = 0
    events = []
    for start, duration, note, velocity in chords + melody:
        events.append((start, b"\x90", note, velocity))
        events.append((start + duration, b"\x80", note, 0))

    for tick, status, note, velocity in sorted(events, key=lambda item: (item[0], item[1])):
        track.extend(_delta(tick - current_tick))
        track.extend(status + bytes([note, velocity]))
        current_tick = tick

    track.extend(_delta(0) + b"\xff\x2f\x00")

    with open(midi_path, "wb") as midi_file:
        midi_file.write(b"MThd")
        midi_file.write(struct.pack(">IHHH", 6, 0, 1, TICKS_PER_BEAT))
        midi_file.write(b"MTrk")
        midi_file.write(struct.pack(">I", len(track)))
        midi_file.write(track)

    return midi_path


def midi_to_wav(
    midi_path: str,
    wav_path: str = None,
    waveform: str = "sine",
) -> str:
    """
    Convert a generated MIDI file into a WAV file using a simple built-in synth.

    Args:
        midi_path: Path to a MIDI file produced by create_midi.
        wav_path: Optional target .wav path. Defaults to same name as midi_path.
        waveform: sine, square, or soft_square.

    Returns:
        Path to the generated .wav file.
    """
    if not os.path.exists(midi_path):
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    if wav_path is None:
        wav_path = os.path.splitext(midi_path)[0] + ".wav"

    notes, tempo = _read_generated_midi(midi_path)
    samples = _render_notes(notes, tempo, waveform)

    os.makedirs(os.path.dirname(wav_path) or ".", exist_ok=True)
    with wave.open(wav_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(samples.tobytes())

    return wav_path


def create_music_wav(
    title: str = "ai_music",
    mood: str = "calm",
    key: str = "C",
    tempo: int = None,
    bars: int = 4,
    waveform: str = "sine",
) -> str:
    """
    Create a complete music artifact by generating MIDI first, then WAV.

    This is useful when the agent wants to satisfy the user in one tool call.
    Returns the final .wav path.
    """
    midi_path = create_midi(title=title, mood=mood, key=key, tempo=tempo, bars=bars)
    return midi_to_wav(midi_path=midi_path, waveform=waveform)


MUSIC_TOOLS = [
    {
        "name": "create_midi",
        "description": (
            "Create a .mid file from a music brief. "
            "Input: title, mood, key, tempo, bars. "
            "Use this before converting to WAV."
        ),
        "func": create_midi,
    },
    {
        "name": "midi_to_wav",
        "description": (
            "Convert a generated .mid file to .wav. "
            "Input: midi_path and optional wav_path, waveform. "
            "Returns the final WAV file path."
        ),
        "func": midi_to_wav,
    },
    {
        "name": "create_music_wav",
        "description": (
            "Generate a .mid file and convert it to .wav in one call. "
            "Input: title, mood, key, tempo, bars, waveform. "
            "Returns the final WAV file path."
        ),
        "func": create_music_wav,
    },
]


def _normalize_spec(
    title: str,
    mood: str,
    key: str,
    tempo: int,
    bars: int,
) -> Dict[str, Any]:
    mood = str(mood or "calm").lower()
    defaults = MOODS.get(mood, MOODS["calm"])
    normalized_key = _normalize_key(key)
    if normalized_key not in NOTE_OFFSETS:
        normalized_key = "C"

    return {
        "title": _slugify(title or "ai_music"),
        "mood": mood,
        "key": normalized_key,
        "scale": defaults["scale"],
        "tempo": int(tempo or defaults["tempo"]),
        "bars": max(1, min(int(bars or 4), 16)),
        "velocity": defaults["velocity"],
    }


def _normalize_key(key: str) -> str:
    value = str(key or "C").strip()
    value = value.replace(" minor", "").replace(" major", "")
    value = value.replace(" Minor", "").replace(" Major", "")
    if len(value) == 2 and value[1] == "m":
        value = value[0]
    return value.upper()


def _compose_song(spec: Dict[str, Any]) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    root = 60 + NOTE_OFFSETS[spec["key"]]
    scale = SCALES[spec["scale"]]
    beat = TICKS_PER_BEAT
    bar = beat * 4
    melody = []
    chords = []
    progression = [0, 3, 4, 0] if spec["scale"] == "major" else [0, 5, 3, 4]

    for bar_index in range(spec["bars"]):
        bar_start = bar_index * bar
        degree = progression[bar_index % len(progression)]
        chord_root = root + scale[degree % len(scale)] - 12
        chord_notes = [chord_root, chord_root + 7, chord_root + 12]
        if spec["scale"] != "pentatonic":
            chord_notes.insert(1, chord_root + (4 if spec["scale"] == "major" else 3))

        for note in chord_notes:
            chords.append((bar_start, bar, note, max(42, spec["velocity"] - 26)))

        pattern = [0, 2, 4, 2, 5, 4, 2, 1]
        for step, degree_offset in enumerate(pattern):
            start = bar_start + step * (beat // 2)
            note = root + scale[(degree + degree_offset) % len(scale)]
            if step in (4, 5):
                note += 12
            melody.append((start, beat // 2, note, spec["velocity"]))

    return melody, chords


def _tempo_bytes(bpm: int) -> bytes:
    microseconds_per_quarter = int(60_000_000 / bpm)
    return microseconds_per_quarter.to_bytes(3, byteorder="big")


def _delta(value: int) -> bytes:
    buffer = value & 0x7F
    value >>= 7
    while value:
        buffer <<= 8
        buffer |= ((value & 0x7F) | 0x80)
        value >>= 7

    result = bytearray()
    while True:
        result.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(result)


def _read_generated_midi(midi_path: str) -> Tuple[List[Tuple[int, int, int, int]], int]:
    data = open(midi_path, "rb").read()
    track_start = data.index(b"MTrk") + 8
    track = data[track_start:]
    tick = 0
    tempo = 120
    active = {}
    notes = []
    index = 0

    while index < len(track):
        delta, index = _read_varlen(track, index)
        tick += delta
        status = track[index]
        index += 1

        if status == 0xFF:
            event_type = track[index]
            index += 1
            length, index = _read_varlen(track, index)
            payload = track[index:index + length]
            index += length
            if event_type == 0x51:
                tempo = round(60_000_000 / int.from_bytes(payload, byteorder="big"))
            elif event_type == 0x2F:
                break
            continue

        event_type = status & 0xF0
        if event_type in (0xC0, 0xD0):
            index += 1
            continue

        if event_type not in (0x80, 0x90):
            index += 2
            continue

        note = track[index]
        velocity = track[index + 1]
        index += 2

        if event_type == 0x90 and velocity > 0:
            active.setdefault(note, []).append((tick, velocity))
        elif event_type in (0x80, 0x90) and note in active and active[note]:
            start, start_velocity = active[note].pop(0)
            notes.append((start, tick - start, note, start_velocity))

    return notes, tempo


def _render_notes(
    notes: Iterable[Tuple[int, int, int, int]],
    tempo: int,
    waveform: str,
) -> array:
    seconds_per_tick = 60.0 / tempo / TICKS_PER_BEAT
    total_ticks = max((start + duration for start, duration, _, _ in notes), default=0)
    total_samples = int(total_ticks * seconds_per_tick * SAMPLE_RATE) + SAMPLE_RATE
    mix = [0.0] * total_samples

    for start, duration, midi_note, velocity in notes:
        start_sample = int(start * seconds_per_tick * SAMPLE_RATE)
        duration_samples = max(1, int(duration * seconds_per_tick * SAMPLE_RATE))
        frequency = 440.0 * (2 ** ((midi_note - 69) / 12))
        amplitude = min(0.25, velocity / 127 * 0.22)

        for offset in range(duration_samples):
            position = start_sample + offset
            if position >= len(mix):
                break
            t = offset / SAMPLE_RATE
            envelope = _envelope(offset, duration_samples)
            mix[position] += amplitude * envelope * _oscillator(frequency, t, waveform)

    pcm = array("h")
    for sample in mix:
        clipped = max(-1.0, min(1.0, sample))
        pcm.append(int(clipped * 32767))
    return pcm


def _oscillator(frequency: float, t: float, waveform: str) -> float:
    phase = 2 * math.pi * frequency * t
    if waveform == "square":
        return 1.0 if math.sin(phase) >= 0 else -1.0
    if waveform == "soft_square":
        return math.tanh(2.5 * math.sin(phase))
    return math.sin(phase)


def _envelope(offset: int, duration: int) -> float:
    attack = max(1, int(duration * 0.04))
    release = max(1, int(duration * 0.18))
    if offset < attack:
        return offset / attack
    if offset > duration - release:
        return max(0.0, (duration - offset) / release)
    return 1.0


def _read_varlen(data: bytes, index: int) -> Tuple[int, int]:
    value = 0
    while True:
        byte = data[index]
        index += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, index


def _slugify(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "_" for char in value)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "ai_music"


def tools_as_json() -> str:
    """Return tool metadata for debugging or reports."""
    return json.dumps(
        [
            {"name": tool["name"], "description": tool["description"]}
            for tool in MUSIC_TOOLS
        ],
        indent=2,
    )
