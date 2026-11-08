# content-sieve

Mines long talks for the insider knowledge buried in them, e.g. the Wall Street war stories
hidden inside hours of religious lectures by an ex-hedge-fund rabbi.

Audio → Whisper transcript (cached) → overlapping chunks → local LLM extracts nuggets through
a speaker profile → quotes verified against the transcript → ranking pass → Markdown report.

## Setup

Needs Python 3.13, poetry, ollama with a model pulled (`ollama pull qwen3:14b`), and for GPU
transcription an AMD gfx103X card (RX 6800 etc.).

    ./setup.sh

This downloads the ROCm build of CTranslate2 into `vendor/` (it isn't on PyPI) and runs
`poetry sync`, which also pulls the ROCm 7 runtime from AMD's pip index (~4 GB).

The ROCm build occasionally aborts with "illegal memory access" on particular stretches of audio,
and the timing varies between runs. Whisper therefore runs in a child process: if it dies, the
next minute is transcribed on the CPU and the GPU picks up after it. If the GPU dies again
right where the CPU handed back, the CPU stretch grows to 5 and then 15 minutes, but the GPU is
always retried after it.

## Usage

    poetry run python sieve.py wallstreet-rabbi ~/Audio/lectures/
    poetry run python sieve.py wallstreet-rabbi "https://www.youtube.com/playlist?list=..."

Inputs can be files, directories, or anything yt-dlp understands (videos, playlists, channels).
Everything lands in `out/`: downloaded audio, transcripts, and one `<talk>.<profile>.md` report
per talk (plus a `.json` with the raw nuggets).

Re-running is cheap: transcripts are cached, and talks that already have a report for that
profile are skipped (`--force` redoes them). So you can rerun the same playlist command to pick up
new uploads, or tweak a profile and rerun with `--force` without re-transcribing.

Useful flags: `--llm`, `--no-think` (about 2× faster, worse judgment), `--top`, `--min-score`,
`--device cpu`, `--chunk-minutes`. See `--help`.

## Profiles

A profile in `profiles/` describes the speaker, what counts as gold, and what to skip.
Copy `profiles/_template.toml`. The `gold` description is the thing to iterate on: the LLM can
only find what you describe.
