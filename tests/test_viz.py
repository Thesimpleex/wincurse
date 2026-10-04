from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from wincurse.viz import THEMES, diverging_cmap, picture_html, save_figure, sequential_cmap


def test_save_figure_writes_light_and_dark_variants(tmp_path):
    import matplotlib.pyplot as plt

    def build(palette):
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot([0, 1], [0, 1], color=palette["series"][0])
        return fig

    paths = save_figure(build, tmp_path / "figs" / "demo")
    assert [p.name for p in paths] == ["demo-light.png", "demo-dark.png"]
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths)


def test_picture_html_references_both_variants():
    html = picture_html("docs/figures/demo", "alt text", width=600)
    assert 'srcset="docs/figures/demo-dark.png"' in html
    assert 'src="docs/figures/demo-light.png"' in html
    assert 'width="600"' in html


def test_themes_have_eight_categorical_colours_and_a_sequential_map():
    for theme in THEMES.values():
        assert len(theme["series"]) == 8
    cmap = sequential_cmap("dark")
    assert cmap(0.0) != cmap(1.0)
    with pytest.raises(KeyError):
        diverging_cmap("sepia")
