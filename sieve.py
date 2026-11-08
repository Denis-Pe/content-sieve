"""Mine long talks for the insider knowledge buried in them.

audio -> Whisper transcript (cached) -> overlapping chunks -> local LLM pulls out
nuggets through the lens of a speaker profile -> quotes are verified against the
transcript -> a per-talk ranking pass -> Markdown report.
"""

import argparse
import ctypes
import importlib
import json
import multiprocessing
import os
import queue
import re
import sys
import tomllib
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import ollama
from rapidfuzz import fuzz

PROJECT_DIR = Path(__file__).parent
AUDIO_EXTS = {".mp3", ".m4a", ".opus", ".webm", ".wav", ".flac", ".ogg", ".mp4", ".mkv"}
YOUTUBE_ID = re.compile(r"\[([A-Za-z0-9_-]{11})\]$")  # yt-dlp's default "Title [id]" naming

# The ROCm build of CTranslate2 links against ROCm 7, which comes from AMD's pip wheels
# rather than the system library path, so it must be loaded before ctranslate2 is imported.
ROCM_LIBS = [
    ("_rocm_sdk_core", ["libamdhip64.so.7"]),
    ("_rocm_sdk_libraries_gfx103X_all", ["libhipblas.so.3", "libhiprand.so.1"]),
]

# How much audio the CPU covers after a GPU crash; each consecutive stall moves one step along.
CPU_PATCH_SECONDS = [60, 5 * 60, 15 * 60]
QUOTE_MATCH_THRESHOLD = 85  # rapidfuzz partial_ratio; below this the quote is treated as invented

DIM, GREEN, YELLOW, RED, RESET = "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"


@dataclass
class Profile:
    name: str
    speaker: str
    gold: str
    chaff: str
    vocabulary: str = ""
    language: str | None = None


@dataclass
class Source:
    audio: Path
    url: str | None = None


@dataclass
class Segment:
    start: float
    text: str


@dataclass
class Nugget:
    start: float
    quote: str
    insight: str
    score: int


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def timestamp_link(source: Source, seconds: float) -> str:
    if source.url and "youtube.com/watch" in source.url:
        return f"[{hms(seconds)}]({source.url}&t={int(seconds)}s)"
    return hms(seconds)


def load_profile(arg: str) -> Profile:
    path = Path(arg)
    if not path.exists():
        path = PROJECT_DIR / "profiles" / f"{arg}.toml"
    return Profile(name=path.stem, **tomllib.loads(path.read_text()))


# ---------------------------------------------------------------- inputs


def download(url: str, audio_dir: Path) -> list[Source]:
    from yt_dlp import YoutubeDL

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(audio_dir / "%(title)s [%(id)s].%(ext)s"),
        "ignoreerrors": True,  # one private/removed video shouldn't sink a whole playlist
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [
            lambda d: d["status"] == "finished" and print(f"{DIM}downloaded {Path(d['filename']).name}{RESET}")
        ],
    }
    sources = []
    with YoutubeDL(opts) as ydl:
        def collect(entry):
            if entry is None:
                return
            if "entries" in entry:  # playlists, and channels whose tabs are playlists
                for e in entry["entries"]:
                    collect(e)
                return
            downloads = entry.get("requested_downloads") or [{}]
            path = downloads[0].get("filepath") or ydl.prepare_filename(entry)
            sources.append(Source(Path(path), entry.get("webpage_url")))

        collect(ydl.extract_info(url, download=True))
    return sources


def local_source(path: Path) -> Source:
    match = YOUTUBE_ID.search(path.stem)
    return Source(path, f"https://www.youtube.com/watch?v={match[1]}" if match else None)


def resolve_inputs(inputs: list[str], audio_dir: Path) -> list[Source]:
    sources = []
    for item in inputs:
        if re.match(r"https?://", item):
            sources += download(item, audio_dir)
            continue
        path = Path(item).expanduser()
        if path.is_dir():
            sources += [local_source(p) for p in sorted(path.rglob("*")) if p.suffix.lower() in AUDIO_EXTS]
        elif path.exists():
            sources.append(local_source(path))
        else:
            print(f"{RED}not found: {item}{RESET}")
    return sources


# ---------------------------------------------------------------- transcription


def preload_rocm():
    for package, libs in ROCM_LIBS:
        lib_dir = Path(importlib.import_module(package).__file__).parent / "lib"
        for lib in libs:
            ctypes.CDLL(str(lib_dir / lib), mode=ctypes.RTLD_GLOBAL)


def load_whisper(model_name: str, device: str):
    preload_rocm()
    import ctranslate2
    from faster_whisper import WhisperModel

    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
    if device == "cuda":
        # float16 is the only compute type that has run cleanly on the RX 6800 (gfx1030).
        return WhisperModel(model_name, device="cuda", compute_type="float16")
    return WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=max(1, (os.cpu_count() or 2) // 2))


def whisper_worker(audio: str, start: float, end: float | None, model_name: str, device: str,
                   language: str | None, vocabulary: str, out: multiprocessing.Queue):
    """Child process: transcribe audio[start:end] and stream [start, end, text] rows to `out`."""
    preload_rocm()
    from faster_whisper.audio import decode_audio

    whisper = load_whisper(model_name, device)
    samples = decode_audio(audio)
    rate = whisper.feature_extractor.sampling_rate
    out.put(("info", len(samples) / rate, whisper.model.device))
    options = {"vad_filter": True, "language": language}
    if whisper.model.device == "cuda":
        # The ROCm build aborts with "illegal memory access" whenever decoding goes beyond
        # plain greedy search with no prompt: the temperature > 0 fallback, hotwords and
        # initial_prompt all trigger it. Without the fallback, conditioning on previous text
        # loops on one phrase for minutes, so that goes too.
        options |= {"temperature": 0.0, "condition_on_previous_text": False}
    else:
        options["hotwords"] = vocabulary or None
    clip = samples[int(start * rate): int(end * rate) if end else None]
    segments, _ = whisper.transcribe(clip, **options)
    for s in segments:
        out.put((round(start + s.start, 2), round(start + s.end, 2), s.text.strip()))
    out.put(("done",))


def run_whisper(audio: Path, start: float, end: float | None, profile: Profile, args, device: str,
                on_info, on_segment) -> tuple[bool, str]:
    """Run whisper_worker in a child process. Returns (finished, device used);
    `finished` is False if the worker was killed by a native crash, which is what a GPU fault looks like."""
    ctx = multiprocessing.get_context("spawn")
    out = ctx.Queue()
    worker = ctx.Process(target=whisper_worker, args=(
        str(audio), start, end, args.whisper, device, profile.language, profile.vocabulary, out,
    ))
    worker.start()
    used_device, finished = device, False
    while True:
        try:
            msg = out.get(timeout=1)
        except queue.Empty:
            if worker.is_alive():
                continue
            try:  # it died: pick up anything it managed to send first
                msg = out.get_nowait()
            except queue.Empty:
                break
        if msg[0] == "info":
            _, duration, used_device = msg
            on_info(duration)
        elif msg[0] == "done":
            finished = True
            break
        else:
            on_segment(msg)
    worker.join()
    if not finished and worker.exitcode is not None and worker.exitcode > 0:
        raise RuntimeError(f"whisper worker failed (exit code {worker.exitcode}), see the traceback above")
    return finished, used_device


def transcribe(source: Source, profile: Profile, args) -> dict:
    """Whisper runs in child processes because a GPU fault kills the whole process rather
    than raising. When the GPU worker dies, the next stretch is transcribed on the CPU and
    the GPU resumes after it."""
    rows: list[list] = []
    duration = 0.0

    def on_info(d: float):
        nonlocal duration
        duration = d

    def on_segment(row):
        rows.append(list(row))
        print(f"\r{DIM}transcribing {hms(row[1])} / {hms(duration)}{RESET}", end="", flush=True)

    position, device, stalls = 0.0, args.device, 0
    while True:
        # A crashing worker can lose its last few messages, so `duration` may still be 0 after this.
        finished, device = run_whisper(source.audio, position, None, profile, args, device, on_info, on_segment)
        if finished:
            break
        if device == "cpu":
            raise RuntimeError("whisper worker died on the CPU")
        crashed_at = max(position, rows[-1][1] if rows else 0.0)
        # A stall is the GPU dying again right where a CPU patch handed back, with no new text.
        # Crashes are random (they even happen at startup), so a stall only means a longer patch,
        # never giving up on the GPU for the rest of the talk.
        stalls = stalls + 1 if position > 0 and crashed_at == position else 0
        patch_end = crashed_at + CPU_PATCH_SECONDS[min(stalls, len(CPU_PATCH_SECONDS) - 1)]
        print(f"\n{YELLOW}GPU crashed at {hms(crashed_at)}, using the CPU until {hms(patch_end)}{RESET}")
        finished, _ = run_whisper(source.audio, crashed_at, patch_end, profile, args, "cpu", on_info, on_segment)
        if not finished:
            raise RuntimeError("whisper worker died on the CPU")
        if patch_end >= duration:
            break
        position = patch_end
    print()
    return {"audio": str(source.audio), "url": source.url, "duration": duration,
            "segments": [[start, text] for start, _, text in rows]}


def transcript_path(workdir: Path, source: Source) -> Path:
    return workdir / "transcripts" / f"{source.audio.stem}.json"


# ---------------------------------------------------------------- extraction

EXTRACT_SYSTEM = """You are mining a long talk for hidden insider knowledge.

The speaker: {speaker}

The talk's main subject is NOT what we want. Most of it is: {chaff}
Skip all of that, even when it fills the entire segment.

What we want: {gold}

Rules:
- Extract only passages where the speaker reveals something from that insider experience: a mechanism, an anecdote, a number, a heuristic, a name, how things really work behind the scenes.
- Judge by mechanism, not by vocabulary. A passage is gold only if the cause of the outcome is worldly: markets, incentives, people, contracts, information. If the speaker credits an outcome to something in the skip list, skip it, even when it is full of dollar figures and business words.
- A boast or a big number with nothing to learn from it ("I made a fortune that year") scores 2 at most.
- Report the factual or tactical content, stripped of the speaker's moral or religious verdict on it. If the speaker says "the traders all chased bonuses and it ruined them", the insight is how bonus structures drive behaviour, not that chasing money is bad.
- `quote` must be copied verbatim from the transcript: the one or two most information-dense sentences. Never paraphrase inside `quote`.
- `insight` is a standalone, plain-language statement of the takeaway that makes sense without the transcript. Never add facts the transcript doesn't state.
- `score`: 5 = genuine insider knowledge you would not find in a book or article; 4 = specific, concrete and non-obvious; 3 = useful heuristic grounded in real experience; 2 = vague or commonly known; 1 = barely relevant.
- Most segments contain nothing worth extracting. An empty `nuggets` list is the normal, expected answer. Do not invent or stretch."""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "nuggets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string"},
                    "insight": {"type": "string"},
                    "score": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["quote", "insight", "score"],
            },
        }
    },
    "required": ["nuggets"],
}

RANK_SYSTEM = """Below are insights extracted from one talk. The speaker: {speaker}
We are looking for: {gold}

Return:
- `summary`: two or three sentences on what insider territory this talk actually covers, ignoring its main subject.
- `ranked`: the ids of the {top} best insights, best first. Best means most non-obvious, specific and actionable: prefer a mechanism someone could use over an impressive number or a boast. If two insights tell the same story, include only the better one."""

RANK_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "ranked": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["summary", "ranked"],
}


def make_chunks(segments: list[Segment], length: float, overlap: float) -> list[list[Segment]]:
    chunks: list[list[Segment]] = []
    start = 0.0
    while segments and start <= segments[-1].start:
        chunk = [s for s in segments if start <= s.start < start + length]
        # skip windows that only contain the overlap tail of the previous chunk
        if chunk and not (chunks and chunk[-1] is chunks[-1][-1]):
            chunks.append(chunk)
        start += length - overlap
    return chunks


def ask(llm: str, system: str, user: str, schema: dict, think: bool) -> dict:
    response = ollama.chat(
        model=llm,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        format=schema,
        think=think,
        options={"num_ctx": 12288, "num_predict": 6144},
    )
    return json.loads(response.message.content)


def locate(quote: str, chunk: list[Segment]) -> float | None:
    """Return the start time of the segment where `quote` appears, or None if it doesn't.
    Models like to trim quotes with "...", so each piece must be found on its own."""
    pieces = [p.strip() for p in re.split(r"\.\.\.|…", quote) if len(p.strip()) > 10] or [quote]
    starts = [locate_piece(piece, chunk) for piece in pieces]
    return None if None in starts else starts[0]


def locate_piece(quote: str, chunk: list[Segment]) -> float | None:
    quote = quote.lower()
    best_score, best_start = 0.0, None
    for i, seg in enumerate(chunk):
        window = ""
        for s in chunk[i:]:  # grow the window until it can contain the whole quote
            window += " " + s.text.lower()
            if len(window) >= len(quote):
                break
        score = fuzz.partial_ratio(quote, window)
        if score > best_score:
            best_score, best_start = score, seg.start
    return best_start if best_score >= QUOTE_MATCH_THRESHOLD else None


def dedupe(nuggets: list[Nugget]) -> list[Nugget]:
    """Drop the copies that overlapping chunks produce, keeping the higher-scored one."""
    kept: list[Nugget] = []
    for n in sorted(nuggets, key=lambda n: -n.score):
        if not any(abs(n.start - k.start) < 120 and fuzz.token_set_ratio(n.quote, k.quote) > 80 for k in kept):
            kept.append(n)
    return sorted(kept, key=lambda n: n.start)


def extract(transcript: dict, profile: Profile, args) -> tuple[list[Nugget], int]:
    segments = [Segment(start, text) for start, text in transcript["segments"]]
    chunks = make_chunks(segments, args.chunk_minutes * 60, args.overlap_seconds)
    system = EXTRACT_SYSTEM.format(speaker=profile.speaker, gold=profile.gold, chaff=profile.chaff)
    nuggets, unverified = [], 0
    for i, chunk in enumerate(chunks, 1):
        span = f"{hms(chunk[0].start)}-{hms(chunk[-1].start)}"
        print(f"{DIM}chunk {i}/{len(chunks)} ({span}){RESET}", end=" ", flush=True)
        user = f"Transcript segment {span} of a {hms(transcript['duration'])} talk:\n\n" + "\n".join(
            f"[{hms(s.start)}] {s.text}" for s in chunk
        )
        try:
            raw = ask(args.llm, system, user, EXTRACT_SCHEMA, args.think)["nuggets"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"{YELLOW}unparseable response, skipped ({e}){RESET}")
            continue
        found = []
        for n in raw:
            if n["score"] < args.min_score:
                continue
            start = locate(n["quote"], chunk)
            if start is None:
                unverified += 1
                continue
            found.append(Nugget(start, n["quote"], n["insight"], n["score"]))
        print(f"{DIM}{len(found)} kept of {len(raw)}{RESET}")
        for n in found:
            print(f"  {GREEN}[{hms(n.start)}] ({n.score}) {n.insight}{RESET}")
        nuggets += found
    return dedupe(nuggets), unverified


def rank(nuggets: list[Nugget], profile: Profile, args) -> tuple[str, list[int]]:
    system = RANK_SYSTEM.format(speaker=profile.speaker, gold=profile.gold, top=args.top)
    user = "\n\n".join(f"id {i}: {n.insight}\n  quote: \"{n.quote}\"" for i, n in enumerate(nuggets))
    try:
        result = ask(args.llm, system, user, RANK_SCHEMA, args.think)
    except (json.JSONDecodeError, KeyError):
        result = {"summary": "", "ranked": []}
    ranked = list(dict.fromkeys(i for i in result.get("ranked", []) if 0 <= i < len(nuggets)))
    if not ranked:  # ranking pass failed or returned nothing usable: fall back to the chunk scores
        ranked = sorted(range(len(nuggets)), key=lambda i: -nuggets[i].score)
    return result.get("summary", ""), ranked[: args.top]


def write_report(path: Path, source: Source, transcript: dict, profile: Profile, args,
                 nuggets: list[Nugget], summary: str, ranked: list[int], unverified: int):
    link = f" · [source]({source.url})" if source.url else ""
    lines = [
        f"# {source.audio.stem}",
        "",
        f"Profile `{profile.name}` · {hms(transcript['duration'])}{link} · {args.llm} · {date.today()}",
        "",
    ]
    if summary:
        lines += [summary, ""]
    if not nuggets:
        lines += ["Nothing found."]

    for rank_no, i in enumerate(ranked, 1):
        n = nuggets[i]
        lines += [f"## {rank_no}. {n.insight}", "", f"> {n.quote}", "",
                  f"{timestamp_link(source, n.start)} · score {n.score}/5", ""]
    rest = [n for i, n in enumerate(nuggets) if i not in ranked]  # nuggets are in chronological order
    if rest:
        lines += ["## Everything else", ""]
        lines += [f"- {timestamp_link(source, n.start)} **{n.insight}** ({n.score}/5)\n  > {n.quote}" for n in rest]
        lines += [""]
    if unverified:
        lines += [f"_{unverified} extracted quote(s) did not match the transcript and were dropped._", ""]
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profile", help="profile name in profiles/ or path to a .toml file")
    parser.add_argument("inputs", nargs="+", help="audio/video files, directories, or URLs (videos, playlists, channels)")
    parser.add_argument("--llm", default="qwen3:14b", help="ollama model (default: %(default)s)")
    parser.add_argument("--whisper", default="large-v3-turbo", help="whisper model (default: %(default)s)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="whisper device")
    parser.add_argument("--chunk-minutes", type=float, default=8)
    parser.add_argument("--overlap-seconds", type=float, default=60)
    parser.add_argument("--min-score", type=int, default=3, help="drop nuggets the LLM scored below this")
    parser.add_argument("--top", type=int, default=10, help="how many nuggets the ranking pass puts on top")
    parser.add_argument("--no-think", dest="think", action="store_false", help="faster, but worse judgment")
    parser.add_argument("--workdir", type=Path, default=PROJECT_DIR / "out")
    parser.add_argument("--force", action="store_true", help="re-extract even if a report already exists")
    parser.add_argument("--retranscribe", action="store_true", help="ignore cached transcripts")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    for sub in ("audio", "transcripts", "reports"):
        (args.workdir / sub).mkdir(parents=True, exist_ok=True)

    sources = resolve_inputs(args.inputs, args.workdir / "audio")
    print(f"{len(sources)} source(s)")

    # Transcribe everything first so whisper is off the GPU before the LLM needs it.
    pending = [s for s in sources if args.retranscribe or not transcript_path(args.workdir, s).exists()]
    if pending:
        try:
            preload_rocm()
        except (ImportError, OSError) as e:
            sys.exit(f"Could not load the ROCm runtime that ctranslate2 is linked against ({e}). Run ./setup.sh")
    for n, source in enumerate(pending, 1):
        print(f"[{n}/{len(pending)}] {source.audio.name}")
        try:
            transcript = transcribe(source, profile, args)
        except Exception as e:
            print(f"{RED}transcription failed: {e}{RESET}")
            continue
        transcript_path(args.workdir, source).write_text(json.dumps(transcript))

    try:
        ollama.show(args.llm)
    except Exception as e:
        sys.exit(f"Ollama model {args.llm!r} is unavailable ({e}). Is `ollama serve` running?")

    for n, source in enumerate(sources, 1):
        tpath = transcript_path(args.workdir, source)
        report = args.workdir / "reports" / f"{source.audio.stem}.{profile.name}.md"
        if not tpath.exists() or (report.exists() and not args.force):
            continue
        print(f"\n[{n}/{len(sources)}] extracting from {source.audio.name}")
        transcript = json.loads(tpath.read_text())
        source.url = source.url or transcript.get("url")
        try:
            nuggets, unverified = extract(transcript, profile, args)
            summary, ranked = rank(nuggets, profile, args) if nuggets else ("", [])
        except ollama.ResponseError as e:
            print(f"{RED}ollama error: {e}{RESET}")
            continue
        write_report(report, source, transcript, profile, args, nuggets, summary, ranked, unverified)
        report.with_suffix(".json").write_text(json.dumps(
            {"summary": summary, "ranked": ranked, "unverified": unverified, "nuggets": [asdict(x) for x in nuggets]},
            indent=2,
        ))
        print(f"{GREEN}{len(nuggets)} nugget(s) -> {report}{RESET}")


if __name__ == "__main__":
    main()
