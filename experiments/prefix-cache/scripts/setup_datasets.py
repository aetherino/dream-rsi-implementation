#!/usr/bin/env python3
"""Download and verify raw datasets. Does not tokenize or call an LLM."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import zipfile

import gdown
import requests

ROOT = Path(__file__).resolve().parents[1]
SHARE_NAME = "ShareGPT_V3_unfiltered_cleaned_split.json"
SHARE_URL = (
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/"
    "resolve/192ab2185289094fc556ec8ce5ce1e8e587154ca/" + SHARE_NAME
)
SHARE_SHA = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
MASH_ID = "1RY_gWB4gaUPkW3w9WhIZAwxg5dzNFliK"
MASH_URL = f"https://drive.google.com/file/d/{MASH_ID}/view"
# Pinned from the original authors' archive downloaded on 2026-09-27.
MASH_SHA = "66c5b829d961a697fad4e2a6ecbb413cf1e2640fbf8b0962ec3ae9f5a55dea72"
MASH_NAMES = [f"{split}_webmd_squad_v2_full.json" for split in ("train", "val", "test")]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(path: Path, source: str, expected: str | None = None,
             drive: bool = False, verify_only: bool = False) -> None:
    if path.exists():
        if expected and digest(path) != expected:
            raise ValueError(f"Checksum mismatch: {path}; move/remove it before retrying")
        print(f"Reusing {path.name}", flush=True)
        return
    if verify_only:
        raise FileNotFoundError(f"Missing {path}; run setup without --verify-only")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {path.name}", flush=True)
    if drive:
        if not gdown.download(id=MASH_ID, output=str(partial), quiet=False):
            raise RuntimeError("MASH-QA download failed")
    else:
        with requests.get(source, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    output.write(chunk)
    if expected and digest(partial) != expected:
        raise ValueError(f"Downloaded checksum mismatch: {partial}")
    partial.replace(path)


def describe(path: Path, root: Path, source: str, kind: str) -> dict:
    result = {"path": str(path.relative_to(root)), "source": source,
              "bytes": path.stat().st_size, "sha256": digest(path)}
    if kind == "archive":
        return result
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if kind == "sharegpt":
        if not isinstance(data, list) or not data:
            raise ValueError("ShareGPT must contain a nonempty list")
        if any(not isinstance(row.get("conversations"), list) for row in data):
            raise ValueError("Invalid ShareGPT conversation record")
        result["conversations"] = len(data)
    else:
        if not isinstance(data, dict) or not isinstance(data.get("data"), list) or not data["data"]:
            raise ValueError(f"Invalid MASH-QA file: {path}")
        result["articles"] = len(data["data"])
        result["questions"] = sum(len(p["qas"]) for row in data["data"] for p in row["paragraphs"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "sharegpt", "mashqa"], default="all")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.data_dir.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}

    def expected(path: Path) -> str | None:
        return manifest["files"].get(str(path.relative_to(root)), {}).get("sha256")

    def record(path: Path, source: str, kind: str) -> None:
        entry = describe(path, root, source, kind)
        previous = expected(path)
        if previous and previous != entry["sha256"]:
            raise ValueError(f"Checksum differs from recorded manifest: {path}")
        manifest["files"][entry["path"]] = entry
        print(f"Verified {entry['path']}: {entry['bytes']:,} bytes", flush=True)
        if not args.verify_only:
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            temp = manifest_path.with_suffix(".json.part")
            temp.write_text(json.dumps(manifest, indent=2) + "\n")
            temp.replace(manifest_path)

    if args.dataset in ("all", "sharegpt"):
        path = root / "sharegpt" / SHARE_NAME
        download(path, SHARE_URL, SHARE_SHA, verify_only=args.verify_only)
        record(path, SHARE_URL, "sharegpt")

    if args.dataset in ("all", "mashqa"):
        archive = root / "mashqa" / "mashqa_data.zip"
        download(archive, MASH_URL, MASH_SHA, drive=True, verify_only=args.verify_only)
        with zipfile.ZipFile(archive) as zf:
            for name in MASH_NAMES:
                path = archive.parent / name
                if not path.exists():
                    if args.verify_only:
                        raise FileNotFoundError(path)
                    matches = [member for member in zf.infolist()
                               if Path(member.filename).name == name and not member.is_dir()
                               and not member.filename.startswith("__MACOSX/")]
                    if len(matches) != 1:
                        raise ValueError(f"Expected exactly one {name}, got {len(matches)}")
                    # Only selected files; never trust an archive's destination paths.
                    partial = path.with_suffix(".json.part")
                    with zf.open(matches[0]) as source, partial.open("wb") as dest:
                        while chunk := source.read(1024 * 1024):
                            dest.write(chunk)
                    partial.replace(path)
                record(path, MASH_URL, "mashqa")
        record(archive, MASH_URL, "archive")
    print("Raw datasets ready. Tokenization is a separate next step.")


if __name__ == "__main__":
    main()
