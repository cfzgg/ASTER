"""Published spatial-domain colour schemes.

Colours are indexed by domain ID (0 .. K-1) as they appear in the paper legend,
so ``get_palette(name)[d]`` is the published colour of domain ``d``.

Only the gastric cancer scheme is tabulated here. The colorectal Visium HD map
(Fig. 4a) was rendered with matplotlib's ``tab20`` applied directly to the label
array, so it needs no table -- pass ``--palette none`` (the default) and the
``tab20`` colormap is used.
"""

from __future__ import annotations

PAPER_PALETTES = {
    # Fig. 5 gastric cancer Xenium, K=17.
    "fig5_gastric": [
        "#3e73af",
        "#a3bde1",
        "#f46928",
        "#f8a368",
        "#43a03b",
        "#89d37d",
        "#c92832",
        "#f5798c",
        "#825ab3",
        "#b99ed1",
        "#7d4c4c",
        "#b78a8c",
        "#cb5db5",
        "#f098cd",
        "#918f8f",
        "#cfcece",
        "#a5b32e",
    ],
}


def get_palette(name: str, k: int | None = None) -> list[str]:
    """Return the hex colour list for ``name``, truncated to ``k`` entries."""
    if name not in PAPER_PALETTES:
        raise KeyError(f"unknown palette {name!r}; available: {sorted(PAPER_PALETTES)}")
    colors = PAPER_PALETTES[name]
    if k is None:
        return list(colors)
    if k > len(colors):
        raise ValueError(f"palette {name!r} has {len(colors)} colours, requested {k}")
    return list(colors[:k])
