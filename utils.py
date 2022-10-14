import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import cut_tree, linkage
from collections import Counter
from numba import njit
from tqdm import tqdm
import hdbscan
import umap


def get_adjacency(expressions: np.ndarray, alpha: float = 6, signed: bool = False) -> np.ndarray:
    correlations = np.corrcoef(expressions, rowvar=False)
    if signed:
        return ((1 + correlations) / 2) ** alpha
    else:
        return np.abs(correlations) ** alpha


def get_tom_numerator(data: np.ndarray) -> np.ndarray:
    return data @ data + data


def get_tom_denominator(data: np.ndarray) -> np.ndarray:
    ss = data.sum(axis=0)
    mm = np.minimum(ss[:, np.newaxis], ss[np.newaxis, :])
    return mm + 1 - data


def get_tom_dissimilarity(data: np.ndarray) -> np.ndarray:
    data = data.copy()
    np.fill_diagonal(data, 0)  # authors "fix"...
    data = (1 - (get_tom_numerator(data) / get_tom_denominator(data)))
    np.fill_diagonal(data, 0)  # authors "fix"...
    return data


def get_mot_denominator(data: np.ndarray) -> np.ndarray:
    ss = data.sum(axis=0)
    mm = np.maximum(ss[:, np.newaxis], ss[np.newaxis, :])
    return mm + 1 - data


def get_mot_dissimilarity(data: np.ndarray) -> np.ndarray:
    data = data.copy()
    np.fill_diagonal(data, 0)  # authors "fix"...
    data = (1 - (get_tom_numerator(data) / get_mot_denominator(data)))
    np.fill_diagonal(data, 0)  # authors "fix"...
    return data


def get_jac_similarity(data: np.ndarray) -> np.ndarray:
    s_q = (data * data).sum(axis=0)
    data = data @ data
    data /= (s_q[:, np.newaxis] + s_q[np.newaxis, :] - data)
    return data


def symmetrize(square_sym_matrix: np.ndarray) -> np.ndarray:
    return (square_sym_matrix + square_sym_matrix.T) / 2


def remove_small_clusters(clusters: np.ndarray, min_cluster_size: int) -> np.ndarray:
    counter = Counter(clusters)
    for key, val in counter.items():
        if val < min_cluster_size:  # rewrite through set
            clusters[clusters == key] = 0
    return clusters


def get_clusters(data: np.ndarray, method='average', cut_height: float = 0.9, min_cluster_size: int = 30) -> np.ndarray:
    y = linkage(squareform(data), method)
    clusters = cut_tree(y, height=cut_height).flatten() + 1
    return remove_small_clusters(clusters, min_cluster_size)


@njit(fastmath=True, parallel=True)
def get_ruzicka(first: np.ndarray, second: np.ndarray) -> float:
    nom = 0
    den = 0
    for f, s in zip(first, second):
        if f > s:
            nom += s
            den += f
        else:
            nom += f
            den += s
    return nom / den


def get_ruzicka_similarity(data: np.ndarray) -> np.ndarray:
    res = data.copy()
    for i in tqdm(range(len(data))):
        for j in range(i):
            res[i, j] = get_ruzicka(data[i], data[j])
            res[j, i] = res[i, j]
    return res


def core_decomposition(data: np.ndarray) -> (np.ndarray, np.ndarray):
    connectivity = data.sum(axis=0)
    indexes = np.zeros(len(connectivity), dtype=int)
    degeneracy = np.zeros(len(connectivity))
    for i in range(len(data)):
        idx = np.argmin(connectivity)
        connectivity -= data[idx]
        indexes[i] = idx
        degeneracy[i] = connectivity[idx]
        connectivity[idx] = np.inf
    return indexes, degeneracy


def clear_data(data: pd.DataFrame, fillna=None) -> pd.DataFrame:
    if fillna is not None:
        data = data.fillna(fillna)
    else:
        data = data.dropna(axis='columns')
    data = data.loc[:, (data != data.iloc[0]).any()]
    return data


def gini(column: pd.Series) -> float:
    counts = column.value_counts().values
    probs = counts / len(column)
    return 1 - np.sum(np.square(probs))


def classification_err(column: pd.Series) -> float:
    counts = column.value_counts().values
    return 1 - np.max(counts) / len(column)


def bootstrap_filter(data: pd.DataFrame, threshold=1 / np.e) -> pd.DataFrame:
    c_errs = np.zeros(len(data.columns))
    for i, column in enumerate(data.columns):
        c_errs[i] = classification_err(data[column])
    return data.loc[:, c_errs > threshold]


def errode_clusters(clusters: np.ndarray, dissim: np.ndarray) -> np.ndarray:
    new_clusters = clusters.copy()
    for i in range(clusters.max() + 1):
        cluster = np.where(clusters == i)[0]
        genes = dissim[cluster][:, cluster].squeeze()
        indexes, degeneracy = core_decomposition(1 - genes)
        new_clusters[cluster] = -1
        new_clusters[cluster[indexes[np.argmax(degeneracy):]]] = i
    return new_clusters


def get_clust(data: pd.DataFrame, min_clust_size=30, n_components=None, random_state=42) -> (np.ndarray, np.ndarray):
    if n_components is None:
        n_components = np.around(np.log2(data.shape[0]), 0) + 1
    sim = get_adjacency(data.values, alpha=1)
    dissim = np.sqrt(np.round(symmetrize(1 - sim), 12))
    clusterable_embedding = umap.UMAP(densmap=True,
                                      dens_lambda=1.,
                                      n_neighbors=2 * min_clust_size,
                                      min_dist=0.0,
                                      n_components=n_components,
                                      random_state=random_state,
                                      low_memory=False,
                                      metric='precomputed'
                                      ).fit_transform(dissim)
    clusterer = hdbscan.HDBSCAN(metric='euclidean',
                                min_cluster_size=min_clust_size,
                                min_samples=min_clust_size // 2,
                                approx_min_span_tree=False,
                                core_dist_n_jobs=-1,
                                )

    clusterer.fit(clusterable_embedding)
    clusters = clusterer.labels_
    core_clusters = errode_clusters(clusters, dissim)
    return dissim, clusters, core_clusters
