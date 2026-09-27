"""scripts/check_numbers.py: LaTeX stripping, allowlist coverage and the strict template check."""
import importlib.util
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_numbers", REPO / "scripts/check_numbers.py")
cn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cn)


def numbers(text):
    return [m.group(0) for m in cn.NUM.finditer(text)]


def test_strip_tex_keeps_prose_and_math_numbers():
    src = (r"\documentclass{article}\usepackage[left=20mm]{geometry}\newcommand{\X}{0.915}" "\n"
           r"\begin{document}" "\n"
           r"Coverage 0.899 \ci{0.87}{0.93} at \(k=3\) % hidden 7.77" "\n"
           r"\end{document}")
    out = cn.strip_tex(src)
    assert len(out) == len(src) and out.count("\n") == src.count("\n")
    assert numbers(out) == ["0.915", "0.899", "0.87", "0.93", "3"]


def test_strip_tex_removes_keys_paths_layout():
    src = (r"\begin{document}" "\n"
           r"See \cref{sec:r2} and \cite{brown2016} \label{fig:3a}; \fig[0.94\linewidth]{figures/fig3.pdf}" "\n"
           r"\begin{tabularx}{\linewidth}{@{}P{3.2cm}L@{}} \multicolumn{2}{c}{x} \cmidrule(lr){2-3} \\[2pt]" "\n"
           r"\vspace{2.5mm}\textcolor{accent!40}{ok} \href{https://orcid.org/1}{link}")
    assert numbers(cn.strip_tex(src)) == []


def test_strip_tex_does_not_mistake_prose_for_lengths():
    # '243 in' is prose, not a TeX length
    assert numbers(cn.strip_tex(r"\begin{document}" "\n" "243 in total, 5 in 7 wells")) == ["243", "5", "7"]


def test_allowlist_rule_must_cover_the_number():
    rules = [re.compile(r"DIV(?:\\,|~|\s)*\d+")]
    bad = list(cn.scan(r"\begin{document}" "\n" r"at DIV\,12 the gain was 0.5", True, rules))
    assert [tok for _, tok, _ in bad] == ["0.5"]


def test_traced_values_accepted_in_rendered_document():
    bad = list(cn.scan("MAE 1.186 vs 1.249 (k = 3)", False, [], traced={"1.186", "1.249", "3"}))
    assert bad == []


def test_template_literal_is_flagged_even_if_value_is_traced():
    tmpl = r"\begin{document}" "\n" r"MAE {{ trajectory_cv.json : by_k.3 | .3f }} and a typed 1.186"
    blanked = cn.PAT_TOKEN.sub(lambda m: " " * len(m.group(0)), tmpl)
    assert [tok for _, tok, _ in cn.scan(blanked, True, [])] == ["1.186"]


def test_repository_allowlist_rules_have_justifications():
    rules = cn.load_allowlist()          # raises SystemExit on a missing justification or a bad regex
    assert rules
