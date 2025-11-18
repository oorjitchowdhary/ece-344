from numpy.typing import ArrayLike

from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection

def apply_sparse_random_projections(dataset: ArrayLike, num_projections: int) -> ArrayLike:
    projector = SparseRandomProjection(n_components=num_projections)
    return projector.fit_transform(dataset)

def apply_dense_random_projections(dataset: ArrayLike, num_projections: int) -> ArrayLike:
    projector = GaussianRandomProjection(n_components=num_projections)
    return projector.fit_transform(dataset)
