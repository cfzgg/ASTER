"""Spatial neighbourhood helpers shared by the fusion / clustering stages."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_knn(coords: np.ndarray, k: int, include_self: bool = False) -> np.ndarray:
    """Return a ``(n, k)`` neighbour index array.

    ``include_self=True`` keeps the query point as neighbour 0, which is what
    :class:`repro_st_aster.aster_sc.fusion_core.FusionNet` expects. The BCAM
    workflows use ``include_self=False`` (self dropped, k true neighbours).
    """
    if include_self:
        nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(coords)
        return nbrs.kneighbors(coords, return_distance=False)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(coords)
    return nbrs.kneighbors(coords, return_distance=False)[:, 1:]


def knn_gaussian_smooth(features: np.ndarray, coords: np.ndarray, k: int = 30) -> np.ndarray:
    """Gaussian-weighted KNN smoothing of a per-spot feature matrix.

    ``sigma`` is the mean distance to the non-self neighbours, matching the
    iSTAR-style smoothing used for the colorectal Visium HD latent before KMeans.
    """
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    sigma = float(np.mean(distances[:, 1:]))
    weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    weights /= weights.sum(axis=1, keepdims=True)

    smoothed = np.zeros_like(features)
    for i in range(len(features)):
        smoothed[i] = (features[indices[i]] * weights[i][:, None]).sum(axis=0)
    return smoothed.astype(features.dtype, copy=False)
