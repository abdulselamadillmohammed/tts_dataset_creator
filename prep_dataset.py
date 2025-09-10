import os
import sys
import wave
import math
import pathlib
import contextlib
import subprocess
from typing import Tuple, List, Optional

# ----------------- Config (edit as needed) -----------------
MODEL_PATH  = "models/ggml-small.en.bin"   # e.g., ggml-base.en.bin / ggml-small.en.bin
CHUNK_SEC   = 10                           # seconds per chunk
OUT_DIR     = "dataset_out"                # will contain metadata.csv + wavs/
WHISPER_BIN = "whisper-cpp"                # or "./main" if you downloaded the binary manually
LANG        = "en"                         # ISO code; set to None to auto-detect
FORCE_CPU   = True                         # True = disable Metal/GPU; safest on macOS
# -----------------------------------------------------------

def ensure_dir(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def split_wav(in_wav: str, out_dir: str, chunk_sec: int = 10) -> List[Tuple[str, str]]:
    """Split a PCM WAV into fixed-length chunks WITHOUT re-encoding. Returns [(name, fullpath)]."""
    ensure_dir(out_dir)
    out: List[Tuple[str, str]] = []
    with contextlib.closing(wave.open(in_wav, "rb")) as wf:
        nchan = wf.getnchannels()
        sw    = wf.getsampwidth()
        rate  = wf.getframerate()
        nfrm  = wf.getnframes()
        frames_per_chunk = int(rate * chunk_sec)
        if frames_per_chunk <= 0:
            raise ValueError("chunk_sec must be > 0")
        n_chunks = math.ceil(nfrm / frames_per_chunk)
        if n_chunks == 0:
            raise ValueError("Input WAV appears empty.")

        for i in range(n_chunks):
            wf.setpos(i * frames_per_chunk)
            frames = wf.readframes(min(frames_per_chunk, nfrm - i * frames_per_chunk))
            name = f"{i+1:04d}.wav"
            out_path = os.path.join(out_dir, name)
            with contextlib.closing(wave.open(out_path, "wb")) as out_wav:
                out_wav.setnchannels(nchan)
                out_wav.setsampwidth(sw)
                out_wav.setframerate(rate)
                out_wav.writeframes(frames)
            out.append((name, out_path))
    return out

def _supports_flag(bin_path: str, flag: str) -> bool:
    try:
        help_out = subprocess.run([bin_path, "-h"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return flag in help_out.stdout
    except Exception:
        return False

def transcribe_chunk(
    chunk_path: str,
    model_path: str,
    whisper_bin: str,
    lang: Optional[str] = "en",
    force_cpu: bool = True
) -> str:
    """Call whisper.cpp CLI to produce a .txt transcription for the chunk and return the text."""
    base_noext = os.path.splitext(chunk_path)[0]
    out_prefix = base_noext

    env = os.environ.copy()
    args = [whisper_bin, "-m", model_path, "-f", chunk_path, "-of", out_prefix, "-otxt"]

    if lang:
        args.extend(["-l", lang])

    if force_cpu:
        # Hard-disable Metal/GPU via env
        env["WHISPER_NO_METAL"] = "1"
        # Add CLI GPU-offload flag only if supported by this binary
        if _supports_flag(whisper_bin, "-ngl"):
            args.extend(["-ngl", "0"])

    res = subprocess.run(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    txt_path = base_noext + ".txt"

    # Some builds return nonzero yet still write the txt; prefer file existence
    if not os.path.exists(txt_path):
        raise RuntimeError(
            "whisper.cpp did not produce a .txt file.\n"
            f"Command: {' '.join(args)}\n"
            f"Exit code: {res.returncode}\n"
            f"STDOUT:\n{res.stdout.decode('utf-8', 'ignore')}\n"
            f"STDERR:\n{res.stderr.decode('utf-8', 'ignore')}\n"
        )

    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def main():
    if len(sys.argv) < 2:
        print("Usage: python prep_dataset.py <input.wav> [--model PATH] [--chunk-sec N] [--lang en|None] [--whisper-bin PATH] [--cpu|--gpu]")
        sys.exit(1)

    in_wav = sys.argv[1]

    model_path  = MODEL_PATH
    chunk_sec   = CHUNK_SEC
    out_dir     = OUT_DIR
    whisper_bin = WHISPER_BIN
    lang        = LANG
    force_cpu   = FORCE_CPU

    i = 2
    while i < len(sys.argv):
        tok = sys.argv[i]
        if tok == "--model" and i+1 < len(sys.argv):
            model_path = sys.argv[i+1]; i += 2
        elif tok == "--chunk-sec" and i+1 < len(sys.argv):
            chunk_sec = int(sys.argv[i+1]); i += 2
        elif tok == "--lang" and i+1 < len(sys.argv):
            lang = None if sys.argv[i+1].lower() == "none" else sys.argv[i+1]; i += 2
        elif tok == "--whisper-bin" and i+1 < len(sys.argv):
            whisper_bin = sys.argv[i+1]; i += 2
        elif tok == "--cpu":
            force_cpu = True; i += 1
        elif tok == "--gpu":
            force_cpu = False; i += 1
        else:
            print(f"Unknown arg: {tok}")
            sys.exit(1)

    if not os.path.exists(in_wav):
        print(f"Input not found: {in_wav}")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    wavs_dir = os.path.join(out_dir, "wavs")
    ensure_dir(wavs_dir)

    print("Splitting...")
    chunks = split_wav(in_wav, wavs_dir, chunk_sec)

    print("Transcribing...")
    meta_lines: List[str] = []
    for fname, fpath in chunks:
        try:
            txt = transcribe_chunk(fpath, model_path, whisper_bin, lang=lang, force_cpu=force_cpu)
        except Exception as e:
            print(f"  ERROR on {fname}: {e}")
            txt = ""  # keep going
        meta_lines.append(f"wavs/{fname}|{txt}")
        if txt:
            preview = (txt[:60] + "...") if len(txt) > 60 else txt
            print(f"  {fname}: {preview}")
        else:
            print(f"  {fname}: (no text)")

    meta_path = os.path.join(out_dir, "metadata.csv")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines))

    print(f"\nDone.\nWrote {meta_path}\nChunks in {wavs_dir}")

if __name__ == "__main__":
    main()
