# ruff: noqa: PLR2004, PLC0415
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler, StandardScaler


@dataclass(frozen=True)
class GeometryResult:
    coordinates: dict[str, pd.DataFrame]
    pca_loadings: pd.DataFrame
    pca_explained_variance: pd.Series
    clusters: pd.DataFrame
    skipped: dict[str, str]


def run_geometry_suite(
    shares: pd.DataFrame,
    *,
    max_points: int = 2_500,
    random_state: int = 7,
) -> GeometryResult:
    clean = shares.replace([np.inf, -np.inf], np.nan).dropna()
    sample = _deterministic_sample(clean, max_points)
    coordinates: dict[str, pd.DataFrame] = {}
    skipped: dict[str, str] = {}

    pca_coords, pca_loadings, explained = run_pca(sample, n_components=3)
    coordinates["pca"] = pca_coords

    robust_pca_coords, _, _ = run_pca(sample, n_components=3, robust_scale=True)
    coordinates["robust_scaled_pca"] = robust_pca_coords

    if len(sample) >= 50:
        coordinates["tsne"] = run_tsne(sample, random_state=random_state)
    else:
        skipped["tsne"] = "not enough rows after cleaning"

    umap_coords, reason = run_umap_optional(sample, random_state=random_state)
    if umap_coords is not None:
        coordinates["umap"] = umap_coords
    else:
        skipped["umap"] = reason or "umap-learn not installed"

    clusters = run_clustering(sample, random_state=random_state)
    hdbscan_labels, reason = run_hdbscan_optional(sample)
    if hdbscan_labels is not None:
        clusters["hdbscan"] = hdbscan_labels
    else:
        skipped["hdbscan"] = reason or "hdbscan not installed"

    return GeometryResult(
        coordinates=coordinates,
        pca_loadings=pca_loadings,
        pca_explained_variance=explained,
        clusters=clusters,
        skipped=skipped,
    )


def run_pca(
    shares: pd.DataFrame,
    *,
    n_components: int = 3,
    robust_scale: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    scaler = RobustScaler() if robust_scale else StandardScaler()
    matrix = scaler.fit_transform(shares.to_numpy(dtype=float))
    pca = PCA(n_components=min(n_components, shares.shape[1]))
    coords = pca.fit_transform(matrix)
    coord_columns = [f"pc{i + 1}" for i in range(coords.shape[1])]
    loading_columns = coord_columns
    loadings = pd.DataFrame(
        pca.components_.T,
        index=shares.columns,
        columns=loading_columns,
    )
    explained = pd.Series(
        pca.explained_variance_ratio_,
        index=coord_columns,
        name="explained_variance_ratio",
    )
    label = "rpc" if robust_scale else "pc"
    coord_columns = [column.replace("pc", label) for column in coord_columns]
    return (
        pd.DataFrame(coords, index=shares.index, columns=coord_columns),
        loadings,
        explained,
    )


def run_tsne(shares: pd.DataFrame, *, random_state: int = 7) -> pd.DataFrame:
    matrix = StandardScaler().fit_transform(shares.to_numpy(dtype=float))
    perplexity = min(40.0, max(5.0, (len(shares) - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    coords = tsne.fit_transform(matrix)
    return pd.DataFrame(coords, index=shares.index, columns=["tsne1", "tsne2"])


def run_clustering(shares: pd.DataFrame, *, random_state: int = 7) -> pd.DataFrame:
    if shares.empty:
        return pd.DataFrame(index=shares.index)
    matrix = StandardScaler().fit_transform(shares.to_numpy(dtype=float))
    clusters = pd.DataFrame(index=shares.index)
    clusters["kmeans_3"] = KMeans(
        n_clusters=3,
        n_init="auto",
        random_state=random_state,
    ).fit_predict(matrix)
    clusters["dbscan"] = DBSCAN(eps=0.75, min_samples=25).fit_predict(matrix)
    if len(shares) >= 100:
        clusters["spectral_3"] = SpectralClustering(
            n_clusters=3,
            affinity="nearest_neighbors",
            n_neighbors=min(30, len(shares) - 1),
            assign_labels="kmeans",
            random_state=random_state,
        ).fit_predict(matrix)
    return clusters


def run_umap_optional(
    shares: pd.DataFrame,
    *,
    random_state: int = 7,
) -> tuple[pd.DataFrame | None, str | None]:
    try:
        import umap  # type: ignore[import-not-found]
    except ImportError:
        return None, "umap-learn not installed"
    if len(shares) < 50:
        return None, "not enough rows after cleaning"
    matrix = StandardScaler().fit_transform(shares.to_numpy(dtype=float))
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.05, random_state=random_state)
    coords = reducer.fit_transform(matrix)
    return pd.DataFrame(coords, index=shares.index, columns=["umap1", "umap2"]), None


def run_hdbscan_optional(shares: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    try:
        import hdbscan  # type: ignore[import-not-found]
    except ImportError:
        return None, "hdbscan not installed"
    if len(shares) < 100:
        return None, "not enough rows after cleaning"
    matrix = StandardScaler().fit_transform(shares.to_numpy(dtype=float))
    labels = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=15).fit_predict(matrix)
    return pd.Series(labels, index=shares.index, name="hdbscan"), None


def _deterministic_sample(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    positions = np.linspace(0, len(frame) - 1, max_points).round().astype(int)
    return frame.iloc[np.unique(positions)].copy()
