"""Fail if a rendered document contains a number that is not traceable to results/*.json.

Checked documents: README.md, docs/*.md (except NUMBERS.md) and writeup/*.md.
A number is accepted if (a) it was produced by render_docs.py for that document (ledger),
(b) it matches a pattern in docs/number_allowlist.txt (each line: regex<TAB>justification),
or (c) it is inside a fenced code block or an inline code span (commands, paths, seeds).
Run: python scripts/check_numbers.py   (exit code 1 on any untraced number)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NUM = re.compile(r"(?<![\w.])[-+−]?\d+(?:[.,]\d+)?(?:e[-+]?\d+)?%?(?![\w])")


def strip_code(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", " ", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)          # link targets (URLs, DOIs, file paths)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)                # HTML tags / comments
    return text


def main():
    ledger_p = REPO / "results/numbers_ledger.json"
    ledger = json.loads(ledger_p.read_text(encoding="utf-8")) if ledger_p.exists() else []
    traced = {}
    for e in ledger:
        traced.setdefault(e["doc"], set()).add(e["value"].replace("−", "-"))
    allow = []
    ap = REPO / "docs/number_allowlist.txt"
    if ap.exists():
        for line in ap.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                allow.append(re.compile(line.split("\t")[0]))
    docs = [REPO / "README.md"] + sorted((REPO / "docs").glob("*.md")) + sorted((REPO / "writeup").glob("*.md"))
    bad = []
    for d in docs:
        if not d.exists() or d.name == "NUMBERS.md":
            continue
        rel = d.relative_to(REPO).as_posix()
        for ln, line in enumerate(strip_code(d.read_text(encoding="utf-8")).splitlines(), 1):
            for m in NUM.finditer(line):
                tok = m.group(0).replace("−", "-")
                core = tok.rstrip("%").lstrip("+")
                if tok in traced.get(rel, ()) or core in traced.get(rel, ()) or core.lstrip("-") in traced.get(rel, ()):
                    continue
                ctx = line[max(0, m.start() - 30): m.end() + 30]
                if any(p.search(ctx) or p.fullmatch(tok) for p in allow):
                    continue
                bad.append(f"{rel}:{ln}: '{tok}' in …{ctx.strip()}…")
    if bad:
        print(f"{len(bad)} untraced number(s):")
        print("\n".join(bad[:200]))
        sys.exit(1)
    print(f"check_numbers: OK ({len(docs)} documents, {len(ledger)} traced numbers)")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
