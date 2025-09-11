import os, argparse
from datasets import Dataset, DatasetDict, Audio
from huggingface_hub import HfApi, HfFolder

def read_metadata(csv_path, data_dir):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or "|" not in ln:
                continue
            rel, text = ln.split("|", 1)
            abs_path = os.path.join(data_dir, rel) if not os.path.isabs(rel) else rel
            rows.append({"audio": abs_path, "text": text})
    if not rows:
        raise SystemExit("No rows found. Check your metadata.csv formatting: 'wavs/0001.wav|transcript'")
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="Folder containing metadata.csv and wavs/")
    p.add_argument("--csv", default="metadata.csv", help="Filename of metadata (default: metadata.csv)")
    p.add_argument("--repo", required=True, help="HF dataset repo name, e.g. 'your-user/my-voice-tts'")
    p.add_argument("--sr", type=int, default=24000, help="Target sample rate for Audio feature (default: 24000)")
    p.add_argument("--private", action="store_true", help="Create repo as private")
    args = p.parse_args()

    csv_path = os.path.join(args.data_dir, args.csv)
    assert os.path.isfile(csv_path), f"Missing {csv_path}"
    wavs_dir = os.path.join(args.data_dir, "wavs")
    assert os.path.isdir(wavs_dir), f"Missing {wavs_dir}"

    rows = read_metadata(csv_path, args.data_dir)

    # Build HF dataset with proper Audio feature
    ds = Dataset.from_dict({"audio": [r["audio"] for r in rows],
                            "text":  [r["text"]  for r in rows]})
    ds = ds.cast_column("audio", Audio(sampling_rate=args.sr))

    dd = DatasetDict({"train": ds})

    # Ensure you’re logged in
    if not HfFolder.get_token():
        raise SystemExit("No HF token found. Run: `huggingface-cli login`")

    # Create repo if it doesn’t exist
    HfApi().create_repo(args.repo, repo_type="dataset", exist_ok=True, private=args.private)

    # Push
    dd.push_to_hub(args.repo)
    print(f"✅ Pushed to: https://huggingface.co/datasets/{args.repo}")

if __name__ == "__main__":
    main()
