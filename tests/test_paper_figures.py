"""The print rules of the paper figures (src/style.py): one-column or full width, no text below 8 pt."""

import pytest

from style import CM, COLUMN_W, MIN_FONT_PT, PAGE_W, PAPER_RC, paper_figure, plt, print_problems


@pytest.fixture(autouse=True)
def paper_rc():
    with plt.rc_context(PAPER_RC):
        yield
    plt.close("all")


@pytest.mark.parametrize("width", [COLUMN_W, PAGE_W])
def test_figure_at_print_width_with_default_text_passes(width):
    fig = paper_figure(width, 6)
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1], label="series")
    ax.set(xlabel="x (unit)", ylabel="y (unit)", title="panel")
    ax.legend()
    assert print_problems(fig) == []


def test_text_below_the_minimum_is_reported():
    fig = paper_figure(COLUMN_W, 6)
    ax = fig.subplots()
    ax.text(0.5, 0.5, "too small", fontsize=MIN_FONT_PT - 1)
    ax.tick_params(labelsize=MIN_FONT_PT - 0.5)
    problems = print_problems(fig)
    assert any("too small" in p for p in problems)
    assert any(f"{MIN_FONT_PT - 0.5:g} pt" in p for p in problems)  # tick labels count too


def test_other_widths_are_refused():
    with pytest.raises(ValueError):
        paper_figure(12 * CM, 6)
    fig = plt.figure(figsize=(12 * CM, 6 * CM))
    assert any(p.startswith("width") for p in print_problems(fig))
