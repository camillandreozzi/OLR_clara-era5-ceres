"""The numbers.tex rule: report.check_tex finds numbers typed by hand in a .tex body, and nothing else."""

import report


def hits(tmp_path, body, preamble=""):
    tex = tmp_path / "main.tex"
    tex.write_text(preamble + "\\begin{document}\n" + body + "\n\\end{document}\n")
    return [tok for _, tok, _ in report.check_tex(tex)]


def test_typed_numbers_in_text_and_maths_are_found(tmp_path):
    assert hits(tmp_path, "RMSE of 39.34 in 49 folds") == ["39.34", "49"]
    assert hits(tmp_path, "$r=0.993$ and $M=1000$") == ["0.993", "1000"]
    assert hits(tmp_path, "\\SI{30}{s} cadence") == ["30"]


def test_macros_structure_and_formula_constants_pass(tmp_path):
    body = "\n".join([
        "RMSE of \\scoreExp{} in \\nFolds{} folds, $r=\\eraCeresR$",                  # macros
        "$\\sigma_1^2/n_i$, $e^{-r^2/2}$, $(1+\\sqrt3 r)$, $\\mathbf{1}[t\\ge0]$",     # exponents, constants
        "\\includegraphics[width=0.49\\textwidth]{maps/2026_08_x.png}\\label{fig:a1}",  # paths, lengths, labels
        "\\cite{loeb2022trends} \\cref{tab:x2} ERA5 NorSat-1 Ed4.2.1 \\\\[4pt]",       # names
        "% 123 in a comment",
        "\\begin{equation} y = \\beta_0 + 2 x \\end{equation}",                         # display maths
    ])
    assert hits(tmp_path, body) == []


def test_preamble_is_not_checked(tmp_path):
    assert hits(tmp_path, "text", preamble="\\documentclass[10pt]{article}\n\\usepackage[margin=1.8cm]{geometry}\n") == []


def test_undefined_number_macro_is_found(tmp_path):
    numbers = tmp_path / "numbers.tex"
    numbers.write_text("\\newcommand{\\scoreExp}{39.34}  % source\n")
    tex = tmp_path / "main.tex"
    tex.write_text("\\begin{document}\n\\scoreExp{} and \\scoreExpp{} and $\\lVert x\\rVert$\n\\end{document}\n")
    assert report.undefined_macros(tex, numbers) == ["scoreExpp"]


def test_signed_or_grouped_macro_inside_siunitx_is_found(tmp_path):
    numbers = tmp_path / "numbers.tex"
    numbers.write_text("\\newcommand{\\bias}{\\ensuremath{+2.4}}  % s\n\\newcommand{\\rmse}{4.7}  % s\n")
    tex = tmp_path / "main.tex"
    tex.write_text("\\begin{document}\n\\flux{\\bias} and \\flux{\\rmse} and \\bias\\,\\Wm\n\\end{document}\n")
    assert report.siunitx_unsafe(tex, numbers) == ["bias"]
