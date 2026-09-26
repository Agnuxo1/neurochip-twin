"""Download the pinned Brewer MAT files from Zenodo record 10257483.

The pinned SHA256 manifest is bundled beside this script. Existing and
downloaded files are verified against it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

RECORD_API = "https://zenodo.org/api/records/10257483"
DEFAULT_RAW = Path("data/raw/brewer_hfs")
MANIFEST = Path(__file__).with_name("brewer_sha256sums.txt")


def expected_hashes(path: Path) -> dict[str, str]:
    hashes = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("* ")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Malformed SHA256 for {name}")
        if Path(name).name != name:
            raise ValueError(f"Unsafe filename in manifest: {name}")
        hashes[name] = digest
    if not hashes:
        raise ValueError("Empty SHA256SUMS manifest")
    return hashes


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def download(raw_dir: Path, verify_only: bool = False) -> dict[str, str]:
    hashes = expected_hashes(MANIFEST)
    mat_files = {name: digest for name, digest in hashes.items() if name.endswith(".mat")}
    if len(mat_files) != 12:
        raise ValueError(f"Expected 12 pinned MAT files, found {len(mat_files)}")
    record = None if verify_only else fetch_json(RECORD_API)
    if record is not None:
        if record.get("id") != 10257483 or record.get("metadata", {}).get("license", {}).get("id") != "cc-zero":
            raise ValueError("Unexpected Zenodo record or license")
        files = {item["key"]: item for item in record["files"]}
        if not set(mat_files) <= set(files):
            raise ValueError("Pinned MAT files missing from Zenodo record")
        raw_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, expected in mat_files.items():
        target = raw_dir / name
        if target.exists() and sha256(target) == expected:
            result[name] = "verified"
            continue
        if verify_only:
            raise ValueError(f"Missing or SHA256 mismatch: {target}")
        part = raw_dir / f"{name}.part"
        try:
            with urllib.request.urlopen(files[name]["links"]["self"], timeout=120) as response, part.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            if sha256(part) != expected:
                raise ValueError(f"Downloaded SHA256 mismatch: {name}")
            os.replace(part, target)
            result[name] = "downloaded and verified"
        finally:
            part.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(download(args.raw_dir, args.verify_only), indent=2))


if __name__ == "__main__":
    main()
