import os, sys, wave, contextlib, math, subprocess, json, pathlib

MODEL_PATH = "models/ggml-small.en.bin"
CHUNK_SEC = 10
OUT_DIR = "dataset_out"
WHISPER_BIN = "whisper-cpp"

# In place path conversion
def ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def split_wav(in_wav, out_dir, chunk_sec=10):
    """
    Splits wav file into chunks (seconds). Assumes it's recieving .wav file    
    """
    with contextlib.closing(wave.open(in_wav, 'rb')) as wf:
        nchan = wf.getnchannels()
        sw = wf.getsampwidth()
        rate = wf.getframerate()
        nfrm = wf.getnframes()
        frames_per_chunk = int(rate * chunk_sec)
        n_chunks = math.ceil(nfrm / frames_per_chunk)

        for i in range(n_chunks):
            wf.setpos(i * frames_per_chunk)
            frames = wf.readframes(min(frames_per_chunk, nfrm - i*frames_per_chunk))
            name = f"{i+1:04d}.wav"
            out_path = os.path.join(out_dir, name)
            with contextlib.closing(wave.open(out_path, 'wb')) as out:
                out.setnchannels(nchan)
                out.setsampwidth(sw)
                out.setframerate(rate)
                out.writeframes(frames)
            yield name, out_path

def transcribe_chunk(chunk_path, model_path, whisper_bin):
    """
    Call whisper.cpp CLI in order to produce a .txt trancription of the chunk.
    """

    base_noext = os.path.splitext(chunk_path)[0]
    out_prefix = base_noext
    cmd = [
        whisper_bin,
        "-m", model_path, 
        "-f", chunk_path,
        "-l", "en",
        "-of", out_prefix,
        "-otxt"
        # add flags like, e.g. "-pp" for puncuation probabilites
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(f"whisper.cpp failed on {chunk_path}:\n{res.stderr.decode('utf-8', 'ignore')}")
    
    txt_path = base_noext + ".txt"
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read().strip()
    
# should return metadata.csv  
def main():
    if len(sys.argv) < 2:
        print("Usage: python prep_dataset_whispercpp.py <input.wav>")
        sys.exit(1)

    in_wav = sys.argv[1]
    if not os.path.exists(in_wav):
        print(f"Input not found: {in_wav}")
        sys.exit(1)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)

    # Prepare output structure
    wavs_dir = os.path.join(OUT_DIR, "wavs")
    ensure_dir(wavs_dir)
    meta_lines = []

    # 1) split
    print("Splitting...")
    chunks = list(split_wav(in_wav, wavs_dir, CHUNK_SEC))

    # 2) transcribe each chunk via whisper.cpp
    print("Transcribing...")
    for fname, fpath in chunks:
        txt = transcribe_chunk(fpath, MODEL_PATH, WHISPER_BIN)
        meta_lines.append(f"wavs/{fname}|{txt}")
        print(f"  {fname}: {txt[:60]}{'...' if len(txt)>60 else ''}")

    # 3) write metadata.csv
    meta_path = os.path.join(OUT_DIR, "metadata.csv")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines))
    print(f"\nDone.\nWrote {meta_path}\nChunks in {wavs_dir}")

if __name__ == "__main__":
    main()