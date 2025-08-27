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