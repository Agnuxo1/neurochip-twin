"""Verify data_bundle/ against its SHA-256 manifest (exit 1 on any mismatch or missing file)."""
import hashlib
import json
import sys
from pathlib import Path

B = Path(__file__).resolve().parents[1] / "data_bundle"
man = json.loads((B / "manifest.json").read_text(encoding="utf-8"))
bad = [rel for rel, meta in man["files"].items()
       if not (B / rel).exists() or hashlib.sha256((B / rel).read_bytes()).hexdigest() != meta["sha256"]]
print(f"bundle: {len(man['files']) - len(bad)}/{len(man['files'])} files verified")
sys.exit(1 if bad else 0)
