from ..distance.JTCorrelator import (
    phase_corr_similarity,
    classical_jtc,
    binary_jtc,
)


def make_distance_fn(
    name: str = "classical_jtc", *, squared: bool = False, shape=(28, 28)
):
    """Return a callable f(X, Y) compatible with sklearn.pairwise_distances."""
    if name == "phase":

        def _d(X, Y):
            d, _, _, _ = phase_corr_similarity(X, Y, shape=shape)
            return d

    elif name == "classical_jtc":

        def _d(X, Y):
            d, _, _, _ = classical_jtc(
                X,
                Y,
                shape=shape,
            )
            return d

    elif name == "binary_jtc":

        def _d(X, Y):
            d, _, _, _ = binary_jtc(
                X,
                Y,
                shape=shape,
            )
            return d

    elif name == "euclidean":
        return "euclidean" if not squared else "sqeuclidean"
    else:
        raise ValueError(f"Unknown distance name '{name}'")

    return (lambda X, Y, *, _f=_d: _f(X, Y) ** 2) if squared else _d
