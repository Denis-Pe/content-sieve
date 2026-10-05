import re
from datetime import timedelta
from pathlib import Path
from time import sleep
import webvtt
from faster_whisper import WhisperModel
from ollama import generate

LLM_MODEL_NAME = "qwen3:14b"
AUDIO_PATH = str(
    Path(
        "~/Audio/bingo.mp3"
    ).expanduser()
)
GENOPTS = {
    "num_ctx": 16384,
    "num_predict": 8192,
}
CHUNK_LENGTH = 60 * 5
WHISPER = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=16)

transcript_parts: list[list] = [[]]
seconds_threshold = CHUNK_LENGTH

segments, info = WHISPER.transcribe(AUDIO_PATH)

for segment in segments:
    ##print(f"{segment.start} --> {segment.end}: {segment.text}")
    if segment.start >= seconds_threshold:
        transcript_parts.append([])
        seconds_threshold += CHUNK_LENGTH
    active_part = transcript_parts[-1]
    active_part.append(segment)

responses = []
prompt_header = "You are an aggressive content filter extracting the intellectual content and arguments made in a transcript. What are the main points made in this transcript segment from this talk? Include timestamps\n\n"
for tp in transcript_parts:
    print(
        f"\033[94mPrompting for segment from {tp[0].start} to {tp[-1].end}...\033[0m"
    )
    prompt = prompt_header + "\n".join([f"{s.start} --> {s.end}: {s.text}" for s in tp])
    # print(f"\033[32m{prompt}\033[0m")
    response = generate(
        model=LLM_MODEL_NAME, prompt=prompt, stream=True, options=GENOPTS, think=False
    )
    collected = ""
    for response_part in response:
        if response_part.thinking:
            print(response_part.thinking, end="", flush=True)
            collected += response_part.thinking
        if response_part.response:
            print(response_part.response, end="", flush=True)
            collected += response_part.response
    print("")
    responses.append(collected)