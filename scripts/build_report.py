"""Build the technical report PDF: render numbers, compile with XeLaTeX + Biber, check the log.

    python scripts/build_report.py            # render_docs -> xelatex, biber, xelatex, xelatex
    python scripts/build_report.py --no-render

Steps: scripts/render_docs.py renders report/neurochip_twin_v2.tex.tmpl (numbers from results/*.json);
the .tex is compiled in report/ with auxiliary files in report/_build/ (ignored by git); the PDF is
copied to report/neurochip_twin_v2.pdf. The script fails on LaTeX errors, undefined references or
citations, missing figure files and overfull boxes wider than 1 pt, and prints the page budget
(main text = pages up to the label 'end-of-main-text'). Needs a TeX distribution with xelatex,
biber and the Libertinus fonts (MiKTeX or TeX Live).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "report"
BUILD = REPORT / "_build"
NAME = "neurochip_twin_v2"
OVERFULL_TOL_PT = 1.0


def run(cmd):
    r = subprocess.run(cmd, cwd=REPORT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(f"  {' '.join(cmd[:1] + cmd[-1:])} -> exit {r.returncode}")
    return r.returncode, r.stdout + r.stderr


def main(argv):
    if "--no-render" not in argv:
        r = subprocess.run([sys.executable, str(REPO / "scripts/render_docs.py")], cwd=REPO)
        if r.returncode:
            sys.exit("render_docs.py failed")
    BUILD.mkdir(exist_ok=True)
    tex = ["xelatex", "-interaction=nonstopmode", "-file-line-error", f"-output-directory={BUILD.name}", f"{NAME}.tex"]
    bib = ["biber", f"--input-directory={BUILD.name}", f"--output-directory={BUILD.name}", NAME]
    for cmd in (tex, bib, tex, tex):
        code, out = run(cmd)
        if code and cmd[0] == "biber":
            print(out[-3000:])
            sys.exit("biber failed")
    log = (BUILD / f"{NAME}.log").read_text(encoding="utf-8", errors="replace")
    blg = (BUILD / f"{NAME}.blg").read_text(encoding="utf-8", errors="replace")

    problems = []
    errors = [ln for ln in log.splitlines() if ln.startswith("!") or re.match(r"^\S+\.tex:\d+: ", ln)]
    problems += [f"LaTeX error: {e}" for e in errors]
    problems += [f"LaTeX: {ln.strip()}" for ln in log.splitlines()
                 if re.search(r"(Reference|Citation) .* undefined|There were undefined references"
                              r"|Label\(s\) may have changed|figure file not found", ln)]
    problems += [f"biber: {ln.strip()}" for ln in blg.splitlines() if re.search(r"\b(ERROR|WARN)\b", ln)]
    over = [(float(w), rest.strip()) for w, rest in
            re.findall(r"Overfull \\[hv]box \(([\d.]+)pt too (?:wide|high)\)([^\n]*)", log)]
    problems += [f"overfull box {w:.1f}pt {where}" for w, where in over if w > OVERFULL_TOL_PT]
    small_over = [w for w, _ in over if w <= OVERFULL_TOL_PT]
    underfull = len(re.findall(r"Underfull \\[hv]box", log))

    aux = (BUILD / f"{NAME}.aux").read_text(encoding="utf-8", errors="replace")
    end = re.search(r"\\newlabel\{end-of-main-text\}\{\{[^}]*\}\{(\d+)\}", aux)
    last = re.search(r"\\newlabel\{LastPage\}\{\{[^}]*\}\{(\d+)\}", aux)
    main_pages = int(end.group(1)) if end else None
    total = int(last.group(1)) if last else None
    shutil.copy2(BUILD / f"{NAME}.pdf", REPORT / f"{NAME}.pdf")

    print(f"pages: main text 1-{main_pages} ({main_pages} pages), total {total}")
    print(f"overfull boxes <= {OVERFULL_TOL_PT} pt (invisible): {len(small_over)}; underfull boxes: {underfull}")
    if main_pages is not None and not 15 <= main_pages <= 20:
        problems.append(f"main text has {main_pages} pages (target 15-20)")
    if problems:
        print(f"{len(problems)} problem(s):")
        print("\n".join("  " + p for p in problems))
        sys.exit(1)
    print(f"OK: report/{NAME}.pdf")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main(sys.argv[1:])
