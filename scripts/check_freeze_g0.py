"""Verify every G0 freeze digest (or only repository artifacts for a light CI job)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _crlf_sha256(path: Path) -> str:
    """SHA-256 of the text with CRLF line endings (the freeze was taken on Windows checkouts)."""
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return hashlib.sha256(data).hexdigest()


def verify(root: Path, repo_only: bool = False) -> dict:
    """``root`` is either the author's workspace (``root/repo/...`` plus ``root/data/...``) or, as in the
    published repository and CI, the repository root itself. In the published layout the raw data that is not
    shipped cannot be checked, so only the ``repo/`` entries are verified (``repo_only`` is implied), and
    a file may match either its LF or its CRLF form (git normalises line endings)."""
    root = root.resolve()
    published = not (root / "repo" / "results" / "freeze_g0.json").is_file()
    base = root if published else root / "repo"
    repo_only = repo_only or published
    manifest_path = base / "results" / "freeze_g0.json"
    sidecar = base / "results" / "freeze_g0.sha256"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_file, name = sidecar.read_text(encoding="ascii").strip().split(maxsplit=1)
    if name != "freeze_g0.json" or (sha256(manifest_path) != expected_file
                                    and not (published and _crlf_sha256(manifest_path) == expected_file)):
        raise ValueError("G0 manifest file SHA256 differs from sidecar")
    payload = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != manifest["payload_sha256"]:
        raise ValueError("G0 canonical payload SHA256 differs")
    # Documented amendments: the original freeze stays immutable; an amendment may only replace the
    # expected hash of files it lists (with the old hash it supersedes) and must state its reason.
    amended, amendments = {}, []
    for amend_path in sorted((base / "results").glob("freeze_g0_amendment_*.json")):
        amend = json.loads(amend_path.read_text(encoding="utf-8"))
        for relative, change in amend["files"].items():
            if manifest["files"].get(relative) != change["old_sha256"] and amended.get(relative) != change["old_sha256"]:
                raise ValueError(f"{amend_path.name}: old hash of {relative} does not match the freeze")
            amended[relative] = change["new_sha256"]
        amendments.append(amend_path.name)
    checked = 0
    for relative, expected in manifest["files"].items():
        if repo_only and not relative.startswith("repo/"):
            continue
        target = (base / relative[len("repo/"):] if published else root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Path escapes workspace: {relative}")
        want = amended.get(relative, expected)
        if not target.is_file() or (sha256(target) != want and not (published and _crlf_sha256(target) == want)):
            raise ValueError(f"G0 file missing or SHA256 differs: {relative}")
        checked += 1
    return dict(status="verified", repo_only=repo_only, checked_files=checked,
                manifest_sha256=expected_file, amendments=amendments,
                amended_files=sorted(amended))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="repository root (published layout) or the author workspace root")
    parser.add_argument("--repo-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.repo_only), indent=2))


if __name__ == "__main__":
    main()
