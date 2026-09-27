"""Fail if a rendered document contains a number that is not traceable to results/*.json.

Checked documents: README.md, docs/*.md (except NUMBERS.md), writeup/*.md, report/*.tex and
report/figures/captions_rendered.md. A number is accepted if
  (a) render_docs.py produced it for that document (results/numbers_ledger.json), or it is a
      number inside a string value that render_docs.py produced for that document;
  (b) it matches a rule in docs/number_allowlist.txt, one rule per line:
          regex<TAB>justification[<TAB>scope]
      A rule accepts a number only if one of its matches on that line covers the number (so a
      rule such as 'DIV\\s*\\d+' accepts the 12 of 'DIV 12' but no other number nearby), or if the
      regex matches the number itself in full. For LaTeX the rule sees the source line, markup
      included. Scope, when given, limits the rule to documents whose path matches that glob
      (e.g. report/*.tex). Rules are for structural constants only (dates, identifiers, protocol
      constants fixed a priori, rubric weights, notation), never for results;
  (c) it is not prose: in Markdown, fenced code blocks, inline code spans, link targets, URLs and
      HTML tags; in LaTeX, see strip_tex() below.
For a document rendered from a template (scripts/render_docs.py targets), the template is checked
too, more strictly: with the {{ }} tokens removed, every remaining literal number must be covered by
an allowlist rule. A hand-typed number is therefore caught even when it happens to equal a value
that some token printed elsewhere in the same document.

LaTeX (report/*.tex) is reduced to its printed prose before checking:
  * comments (unescaped %) are removed, which also removes the %% RESERVED blocks;
  * the preamble (before \\begin{document}) is layout configuration and is removed, except the
    bodies of argument-free value macros (\\newcommand{\\X}{...}), which are checked like prose;
  * arguments that are keys, paths, URLs, colours or layout are removed: \\label, \\ref, \\cref,
    \\cite and relatives, \\includegraphics, \\fig, \\file, \\url, \\href (target only),
    \\definecolor, \\color, \\textcolor, \\colorbox, \\vspace, \\hspace, \\setlength, \\setcounter,
    \\fontsize, \\rule, \\resizebox, \\scalebox, \\parbox and \\raisebox widths, \\multicolumn and
    \\multirow spans, \\cmidrule and \\cline ranges, table column specs and float placements;
  * TeX dimensions (e.g. 6pt, 3.2cm, .6ex, 0.40\\textheight) are removed.
Numbers in mathematics (inline or display) are checked like prose: the structural constants of
the equations (indices, exponents, 1 - alpha, ...) are listed with justification in the allowlist.

Run:  python scripts/check_numbers.py            (all documents; exit code 1 on any untraced number)
      python scripts/check_numbers.py report/    (only documents under the given path prefixes)
"""
from __future__ import annotations

import fnmatch
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NUM = re.compile(r"(?<![\w.])[-+−]?\d+(?:[.,]\d+)?(?:e[-+]?\d+)?%?(?![\w])")
MINUS = "−"
PAT_TOKEN = re.compile(r"\{\{[^{}]*?\.json\s*:[^{}]*?\}\}")     # a render_docs.py token (see its PAT)


def strip_code(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", " ", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)          # link targets (URLs, DOIs, file paths)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)                # HTML tags / comments
    return text


# ----------------------------------------------------------------------------------------- LaTeX
# command -> (optional args to blank, mandatory args to blank); the command name is blanked too.
# Optional [..] groups and (..) groups directly after the command are always blanked.
TEX_ARGS = {
    # cross-references and citations (keys)
    "label": 1, "ref": 1, "cref": 1, "Cref": 1, "autoref": 1, "pageref": 1, "pageref*": 1,
    "eqref": 1, "nameref": 1, "appref": 1, "hyperref": 0, "hypertarget": 1, "hyperlink": 1,
    "cite": 1, "cites": 1, "parencite": 1, "textcite": 1, "autocite": 1, "footcite": 1,
    "citeauthor": 1, "citeyear": 1, "citetitle": 1, "nocite": 1, "citep": 1, "citet": 1,
    "footfullcite": 1, "fullcite": 1,
    # paths and URLs
    "includegraphics": 1, "fig": 1, "input": 1, "include": 1, "file": 1, "url": 1,
    "nolinkurl": 1, "href": 1, "addbibresource": 1, "IfFileExists": 1,
    # colours
    "definecolor": 3, "color": 1, "textcolor": 1, "colorbox": 1, "fcolorbox": 2, "pagecolor": 1,
    "cellcolor": 1, "rowcolor": 1, "columncolor": 1,
    # layout
    "vspace": 1, "vspace*": 1, "hspace": 1, "hspace*": 1, "setlength": 2, "addtolength": 2,
    "setcounter": 2, "addtocounter": 2, "fontsize": 2, "linespread": 1, "rule": 2,
    "resizebox": 2, "resizebox*": 2, "scalebox": 1, "parbox": 1, "raisebox": 1, "makebox": 0,
    "enlargethispage": 1, "enlargethispage*": 1, "captionsetup": 1, "multicolumn": 2,
    "multirow": 2, "cmidrule": 1, "cline": 1, "addlinespace": 0, "newcolumntype": 2,
    "thispagestyle": 1, "pagestyle": 1, "setlist": 1, "titlespacing": 4, "titlespacing*": 4,
    "arrayrulecolor": 1, "specialrule": 3, "hyphenation": 1,
}
# environments -> number of mandatory arguments that are layout (column specs, widths)
TEX_ENV_ARGS = {"tabular": 1, "tabular*": 2, "tabularx": 2, "tabulary": 2, "longtable": 1,
                "minipage": 1, "table": 0, "table*": 0, "figure": 0, "figure*": 0, "wrapfigure": 2,
                "threeparttable": 0, "tablenotes": 0, "adjustbox": 1, "tcolorbox": 0,
                "claimbox": 0, "reservedbox": 0, "summarybox": 0, "multicols": 1, "itemize": 0,
                "enumerate": 0, "description": 0}
# (no space allowed between number and unit, so prose such as "243 in total" is never taken for a length)
DIM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:pt|mm|cm|in|em|ex|bp|sp|pc|dd|cc|mu|fill?l?)\b"
                 r"|[-+]?(?:\d+\.?\d*|\.\d+)\s*\\(?:textheight|textwidth|linewidth|columnwidth|hsize|vsize|"
                 r"baselineskip|paperwidth|paperheight|parindent|parskip|fboxsep|tabcolsep|headwidth)\b")
PENALTY = re.compile(r"\\[a-z]*penalty\s*=?\s*-?\d+|\\(?:arraystretch|stretch)\b")


def _blank(chars: list, a: int, b: int) -> None:
    for i in range(a, b):
        if chars[i] != "\n":
            chars[i] = " "


def _group_end(text: str, i: int, open_: str, close: str) -> int:
    """Index just after the balanced group that starts at text[i] == open_ (or i if none)."""
    if i >= len(text) or text[i] != open_:
        return i
    depth = 0
    j = i
    while j < len(text):
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return len(text)


def _skip_ws(text: str, i: int) -> int:
    while i < len(text) and text[i] in " \t":
        i += 1
    return i


def _blank_args(text: str, chars: list, start: int, n_mand: int, parens: bool = False) -> None:
    """Blank the optional [..] groups (and (..) groups if parens) that precede or sit between the
    first n_mand mandatory {..} groups starting at text[start], and those mandatory groups."""
    i = start
    while True:
        k = _skip_ws(text, i)
        if k < len(text) and text[k] == "[" and (n_mand > 0 or i == start):
            e = _group_end(text, k, "[", "]")
        elif k < len(text) and text[k] == "(" and parens:
            e = _group_end(text, k, "(", ")")
        elif k < len(text) and text[k] == "{" and n_mand > 0:
            e = _group_end(text, k, "{", "}")
            n_mand -= 1
        else:
            return
        _blank(chars, i, e)
        i = e


def strip_tex(text: str) -> str:
    """Return text of the same length and line structure with non-prose LaTeX blanked."""
    chars = list(text)
    # 1. comments (an unescaped %, i.e. preceded by an even number of backslashes)
    for m in re.finditer(r"(?<!\\)((?:\\\\)*)%[^\n]*", text):
        _blank(chars, m.start() + len(m.group(1)), m.end())
    text = "".join(chars)
    # 2. preamble: keep only the bodies of argument-free value macros
    b = text.find("\\begin{document}")
    if b > 0:
        keep = []
        for m in re.finditer(r"\\newcommand\*?\s*\{?\\[A-Za-z@]+\}?\s*", text[:b]):
            k = m.end()
            if k < b and text[k] == "{":
                keep.append((k + 1, _group_end(text, k, "{", "}") - 1))
        for i in range(b):
            if not any(a <= i < e for a, e in keep):
                if chars[i] != "\n":
                    chars[i] = " "
        text = "".join(chars)
    # 3. environments: \begin{env}[opt]{spec}
    for m in re.finditer(r"\\begin\{([A-Za-z*]+)\}", text):
        _blank(chars, m.start(), m.end())
        _blank_args(text, chars, m.end(), TEX_ENV_ARGS.get(m.group(1), 0))
    for m in re.finditer(r"\\end\{[A-Za-z*]+\}", text):
        _blank(chars, m.start(), m.end())
    text = "".join(chars)
    # 4. commands whose arguments are keys, paths, URLs, colours or layout
    for m in re.finditer(r"\\([A-Za-z]+\*?)", text):
        name = m.group(1) if m.group(1) in TEX_ARGS else m.group(1).rstrip("*")
        if name in TEX_ARGS:
            _blank(chars, m.start(), m.end())
            _blank_args(text, chars, m.end(), TEX_ARGS[name], parens=(name == "cmidrule"))
    text = "".join(chars)
    # 5. \\[2pt] line-break spacing, dimensions, penalties, colour mixes such as accent!40
    for pat in (re.compile(r"\\\\\s*\[[^\]]*\]"), DIM, PENALTY, re.compile(r"![0-9]+(?:![A-Za-z]+)?")):
        for m in pat.finditer(text):
            _blank(chars, m.start(), m.end())
        text = "".join(chars)
    # 6. remaining control words (their names have no digits, but keep tokens apart)
    text = re.sub(r"\\[A-Za-z]+\*?", lambda m: " " * len(m.group(0)), text)
    return text


# ----------------------------------------------------------------------------------------- checks
def load_allowlist():
    rules = []
    ap = REPO / "docs/number_allowlist.txt"
    if ap.exists():
        for ln, line in enumerate(ap.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 2 or not cols[1].strip():
                raise SystemExit(f"docs/number_allowlist.txt:{ln}: every rule needs a justification")
            scope = cols[2].strip() if len(cols) > 2 and cols[2].strip() else "*"
            try:
                rules.append((re.compile(cols[0]), scope))
            except re.error as e:
                raise SystemExit(f"docs/number_allowlist.txt:{ln}: bad regex ({e})")
    return rules


def documents():
    docs = [REPO / "README.md"] + sorted((REPO / "docs").glob("*.md")) + sorted((REPO / "writeup").glob("*.md"))
    docs += sorted((REPO / "report").glob("*.tex")) + [REPO / "report/figures/captions_rendered.md"]
    return [d for d in docs if d.exists() and d.name != "NUMBERS.md"]


def scan(raw: str, is_tex: bool, rules, traced=()):
    """Yield (line number, token, context) for every number that is neither traced nor allowlisted."""
    text = strip_tex(raw) if is_tex else strip_code(raw)
    src_lines = raw.splitlines() if is_tex else None
    for ln, line in enumerate(text.splitlines(), 1):
        for m in NUM.finditer(line):
            tok = m.group(0).replace(MINUS, "-")
            core = tok.rstrip("%").lstrip("+")
            if tok in traced or core in traced or core.lstrip("-") in traced:
                continue
            # allowlist: a rule accepts the number only if one of its matches on the line covers
            # the number itself (for LaTeX the source line, so rules can see the markup)
            ctx_line = src_lines[ln - 1] if src_lines is not None else line
            if any(p.fullmatch(tok) or any(r.start() <= m.start() and r.end() >= m.end()
                                           for r in p.finditer(ctx_line)) for p in rules):
                continue
            yield ln, tok, ctx_line[max(0, m.start() - 30): m.end() + 30].strip()


def template_sources():
    """Map rendered document (repo-relative) -> its template, as defined by scripts/render_docs.py."""
    sys.path.insert(0, str(REPO / "scripts"))
    import render_docs  # noqa: E402  (same folder)
    return {dst.relative_to(REPO).as_posix(): src for src, dst in render_docs.targets()}


def main(argv):
    ledger_p = REPO / "results/numbers_ledger.json"
    ledger = json.loads(ledger_p.read_text(encoding="utf-8")) if ledger_p.exists() else []
    traced = {}
    for e in ledger:
        v = e["value"].replace(MINUS, "-")
        s = traced.setdefault(e["doc"], set())
        s.add(v)
        s.update(t.group(0).replace(MINUS, "-") for t in NUM.finditer(v))   # numbers inside string values
    allow = load_allowlist()
    prefixes = [p.replace("\\", "/").rstrip("/") for p in argv]
    docs = [d for d in documents()
            if not prefixes or any(d.relative_to(REPO).as_posix().startswith(p) for p in prefixes)]
    templates = template_sources()
    bad = []
    n_templates = 0
    for d in docs:
        rel = d.relative_to(REPO).as_posix()
        rules = [p for p, scope in allow if fnmatch.fnmatch(rel, scope)]
        is_tex = d.suffix == ".tex"
        # (1) the rendered document: every number is traced (ledger) or allowlisted
        for ln, tok, ctx in scan(d.read_text(encoding="utf-8"), is_tex, rules, traced.get(rel, set())):
            bad.append(f"{rel}:{ln}: '{tok}' in …{ctx}…")
        # (2) its template, if any: every literal number outside the tokens must be allowlisted, so a
        #     hand-typed number is caught even when it happens to equal a traced value
        src = templates.get(rel)
        if src is not None and src.exists():
            n_templates += 1
            tmpl = PAT_TOKEN.sub(lambda m: " " * len(m.group(0)), src.read_text(encoding="utf-8"))
            for ln, tok, ctx in scan(tmpl, is_tex, rules):
                bad.append(f"{src.relative_to(REPO).as_posix()}:{ln}: literal '{tok}' in …{ctx}…")
    if bad:
        print(f"{len(bad)} untraced number(s):")
        print("\n".join(bad[:300]))
        sys.exit(1)
    print(f"check_numbers: OK ({len(docs)} documents, {n_templates} templates, {len(ledger)} traced numbers, "
          f"{len(allow)} allowlist rules)")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main(sys.argv[1:])
