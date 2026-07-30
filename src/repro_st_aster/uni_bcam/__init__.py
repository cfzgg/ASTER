"""UNI-2 feature extraction and BCAM/FusionNet input preparation.

Submodules are imported lazily so that running one as ``python -m`` does not
re-import it through the package first (which would warn), and so that ``--dry-run``
paths avoid importing torch / scanpy.

- ``uni_extract``              -- UNI-2 superpixel feature extraction
- ``bcam_prepare_inputs``      -- single-cell Xenium inputs (breast, gastric)
- ``prepare_inputs_visiumhd``  -- Visium HD bin inputs (colorectal)
- ``prepare_common``           -- shared coordinate / SVG helpers
"""

__all__ = [
    "build_dataset",
    "extract_features",
    "inside_standardized_image",
    "load_he_standardization",
    "map_to_superpixel",
    "prepare_inputs",
    "prepare_inputs_hd",
    "select_svg_morans_i",
    "standardize_coords",
]

_LAZY = {
    "build_dataset": "bcam_train",
    "extract_features": "uni_extract",
    "inside_standardized_image": "prepare_common",
    "load_he_standardization": "prepare_common",
    "map_to_superpixel": "prepare_common",
    "prepare_inputs": "bcam_prepare_inputs",
    "prepare_inputs_hd": "prepare_inputs_visiumhd",
    "select_svg_morans_i": "prepare_common",
    "standardize_coords": "prepare_common",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f"{__name__}.{_LAZY[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
