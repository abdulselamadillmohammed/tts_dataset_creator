import os
import argparse
import shutil
from pathlib import Path

from datasets import Dataset, DatasetDict, Audio


def read_metadata(csv_path: str, data_root: str):
    """
    Read lines like:  wavs/0001.wav|transcript text...
    Returns list of dicts with RELATIVE audio paths (posix style) and text.
    """
    rows = []
    data_root = os.path.abspath(data_root)

    with open(csv_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or "|" not in ln:
                continue
            rel, text = ln.split("|", 1)
            rel = rel.strip()
            text = text.strip()

            # Normalize to a relative path rooted at data_root if an absolute was provided
            abs_path = rel if os.path.isabs(rel) else os.path.join(data_root, rel)
            abs_path = os.path.abspath(abs_path)

            # Compute path relative to data_root to keep the dataset portable
            try:
                rel_path = os.path.relpath(abs_path, data_root)
            except ValueError:
                # On weird path mismatches, just store the basename to avoid absolute paths
                rel_path = os.path.basename(abs_path)

            # Use forward slashes for portability
            rel_path = Path(rel_path).as_posix()

            rows.append({"audio": rel_path, "text": text})

    if not rows:
        raise SystemExit(
            "No rows found. Check your metadata.csv formatting: 'wavs/0001.wav|transcript'"
        )
    return rows


def copy_wavs_tree(src_wavs_dir: str, outdir: str):
    """
    Copy the entire wavs/ tree into <outdir>/wavs so the dataset is self-contained.
    Uses dirs_exist_ok for idempotency.
    """
    src_wavs_dir = os.path.abspath(src_wavs_dir)
    dst_wavs_dir = os.path.join(outdir, "wavs")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_wavs_dir, dst_wavs_dir, dirs_exist_ok=True)


def main():
    p = argparse.ArgumentParser(description="Build a local HF DatasetDict for TTS/STT, no Hub push.")
    p.add_argument("--data_dir", required=True, help="Folder containing metadata.csv and wavs/")
    p.add_argument("--csv", default="metadata.csv", help="Filename of metadata (default: metadata.csv)")
    p.add_argument("--sr", type=int, default=24000, help="Target sample rate for Audio feature (default: 24000)")
    p.add_argument("--outdir", default="my_voice_dataset", help="Where to save the DatasetDict (default: my_voice_dataset)")
    p.add_argument(
        "--no-copy-audio",
        action="store_true",
        help="Do NOT copy wavs/ into the saved dataset folder (you'll need to ship wavs/ separately).",
    )
    args = p.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    csv_path = os.path.join(data_dir, args.csv)
    wavs_dir = os.path.join(data_dir, "wavs")

    assert os.path.isfile(csv_path), f"Missing {csv_path}"
    assert os.path.isdir(wavs_dir), f"Missing {wavs_dir} (expected your audio under data_dir/wavs/)"

    # Read metadata as RELATIVE paths under data_dir
    rows = read_metadata(csv_path, data_dir)

    # === Path fix block ===
    # Ensure paths resolve correctly during save_to_disk():
    # - If copying audio, point to <outdir>/wavs/...
    # - If not copying, convert to ABSOLUTE paths rooted at data_dir.
    if not args.no_copy_audio:
        # Copy audio into the output folder so the dataset is self-contained
        copy_wavs_tree(wavs_dir, args.outdir)
        # Repoint audio paths to the copied tree inside <outdir>/wavs
        rows = [
            {"audio": (Path(args.outdir) / r["audio"]).as_posix(), "text": r["text"]}
            for r in rows
        ]
    else:
        # Keep audio in place; use absolute paths to avoid save_to_disk errors
        rows = [
            {"audio": (Path(data_dir) / r["audio"]).resolve().as_posix(), "text": r["text"]}
            for r in rows
        ]

    # Build dataset
    ds = Dataset.from_dict(
        {
            "audio": [r["audio"] for r in rows],
            "text": [r["text"] for r in rows],
        }
    )

    # Cast to Audio feature; decoding will happen on the fly from the provided paths
    ds = ds.cast_column("audio", Audio(sampling_rate=args.sr))

    dd = DatasetDict({"train": ds})

    # Save locally
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(args.outdir)

    print("✅ Saved local DatasetDict.")
    print(f"   Folder: {os.path.abspath(args.outdir)}")
    if args.no_copy_audio:
        print("⚠️  You used --no-copy-audio.")
        print("   Be sure to also upload/ship your 'wavs/' folder next to the dataset folder and keep relative paths intact.")
    else:
        print("🎁 Audio copied into dataset folder. You can upload just this one folder to Colab.")


if __name__ == "__main__":
    main()

# Examples:
# python make_local_dataset.py --data_dir /path/to/data --outdir my_voice_dataset
# python make_local_dataset.py --data_dir dataset_out --outdir my_voice_dataset
