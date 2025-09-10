# prep_dataset_openai.py
import os, sys, wave, contextlib, math, pathlib
from typing import List, Tuple, Optional

# ----------------- Config (edit as needed) -----------------
CHUNK_SEC = 10                  # seconds per chunk
OUT_DIR   = "dataset_out"       # will contain metadata.csv + wavs/
LANG      = "en"                # set to None for auto
# -----------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()
# OpenAI SDK (v1+)
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # uses OPENAI_API_KEY from env

def ensure_dir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def split_wav(in_wav: str, out_dir: str, chunk_sec: int = 10) -> List[Tuple[str, str]]:
    """
    Split a PCM WAV file into fixed-length chunks WITHOUT re-encoding.
    Returns list of (filename, fullpath).
    """
    ensure_dir(out_dir)
    out: List[Tuple[str, str]] = []
    with contextlib.closing(wave.open(in_wav, "rb")) as wf:
        if wf.getcomptype() != "NONE":
            raise ValueError(f"Compressed WAV ({wf.getcomptype()}) not supported; convert to PCM (e.g., ffmpeg -i in -ac 1 -ar 16000 -c:a pcm_s16le out.wav)")

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

def sanitize_for_metadata(s: str) -> str:
    # Keep metadata.csv one row per chunk; avoid breaking the delimiter
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", " ")     # no newlines in the CSV row
    s = s.replace("|", "/")      # guard our delimiter
    return s.strip()

def transcribe_chunk_openai(chunk_path: str, lang: Optional[str] = "en") -> str:
    """
    Use OpenAI Whisper API to transcribe a single audio chunk.
    Returns plain text (no timestamps).
    """
    # Important: give the upload a filename with extension so the API infers type correctly.
    # (OpenAI’s server uses the filename/extension to detect format for some clients.)
    with open(chunk_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=("chunk.wav", f, "audio/wav"),   # (name, fileobj, mimetype)
            language=lang if lang else None,
            response_format="text"               # return pure text
        )
    # SDK returns a str when response_format="text"
    return str(transcript).strip()

def main():
    if len(sys.argv) < 2:
        print("Usage: python prep_dataset_openai.py <input.wav> [--chunk-sec N] [--lang en|None]")
        sys.exit(1)

    in_wav = sys.argv[1]
    chunk_sec = CHUNK_SEC
    lang = LANG

    # simple optional flags
    i = 2
    while i < len(sys.argv):
        tok = sys.argv[i]
        if tok == "--chunk-sec" and i+1 < len(sys.argv):
            chunk_sec = int(sys.argv[i+1]); i += 2
        elif tok == "--lang" and i+1 < len(sys.argv):
            lang = None if sys.argv[i+1].lower() == "none" else sys.argv[i+1]; i += 2
        else:
            print(f"Unknown arg: {tok}"); sys.exit(1)

    if not os.path.exists(in_wav):
        print(f"Input not found: {in_wav}")
        sys.exit(1)
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set in environment.")
        sys.exit(1)

    wavs_dir = os.path.join(OUT_DIR, "wavs")
    ensure_dir(wavs_dir)

    print("Splitting...")
    chunks = split_wav(in_wav, wavs_dir, chunk_sec)

    print("Transcribing (OpenAI Whisper)…")
    meta_lines: List[str] = []
    for fname, fpath in chunks:
        try:
            txt = transcribe_chunk_openai(fpath, lang=lang)
        except Exception as e:
            print(f"  ERROR on {fname}: {e}")
            txt = ""  # keep going; fix later if needed

        txt = sanitize_for_metadata(txt)
        meta_lines.append(f"wavs/{fname}|{txt}")
        if txt:
            preview = (txt[:60] + "...") if len(txt) > 60 else txt
            print(f"  {fname}: {preview}")
        else:
            print(f"  {fname}: (no text)")

    meta_path = os.path.join(OUT_DIR, "metadata.csv")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines))

    print(f"\nDone.\nWrote {meta_path}\nChunks in {wavs_dir}")

if __name__ == "__main__":
    main()


# command to run:
# python prep_dataset.py inputs/output.wav