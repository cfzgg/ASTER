"""ASTER-SC workflows.

Submodules are imported lazily so that ``--dry-run`` paths, which only need
argparse and path handling, do not pay for importing torch.

- ``inr_model``   -- SIREN + Tucker-2 modules (shared)
- ``fusion_core`` -- FusionNet, the colorectal Visium HD fusion head
- ``reconstruct`` / ``reconstruct_hd`` -- INR stage CLIs (single-cell / Visium HD)
- ``bcam`` / ``fusion``               -- fusion stage CLIs (single-cell / Visium HD)
- ``cluster_visualize``               -- domain clustering + figures
"""

__all__ = [
    "CrossAttentionLayer",
    "CrossAttentionModel",
    "FusionNet",
    "LRT_Tucker2",
    "SineLayer",
    "XYNetSIREN",
    "omega_ramp",
]

_LAZY = {
    "CrossAttentionLayer": "fusion_core",
    "CrossAttentionModel": "fusion_core",
    "FusionNet": "fusion_core",
    "LRT_Tucker2": "inr_model",
    "SineLayer": "inr_model",
    "XYNetSIREN": "inr_model",
    "omega_ramp": "inr_model",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f"{__name__}.{_LAZY[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
