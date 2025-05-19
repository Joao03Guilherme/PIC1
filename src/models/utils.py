from ..distance.JTCorrelator import (
    phase_corr_similarity,
    classical_jtc,
    binary_jtc,
)
from ..distance.OpticalJTCorrelator import OpticalJTCorrelator


def make_distance_fn(
    name: str = "classical_jtc", *, squared: bool = False, shape=(28, 28), optical_correlator=None
):
    """Return a callable f(X, Y) compatible with sklearn.pairwise_distances.
    
    Parameters
    ----------
    name : str
        The name of the distance metric. Options:
        - 'phase': Use phase correlation.
        - 'classical_jtc': Use classical JTC algorithm.
        - 'binary_jtc': Use binary JTC algorithm.
        - 'optical_classical_jtc': Use hardware-based optical JTC.
        - 'euclidean': Use Euclidean distance.
    squared : bool, default=False
        Whether to square the distance value.
    shape : tuple, default=(28, 28)
        Shape of the input images (H, W).
    optical_correlator : OpticalJTCorrelator, optional
        Instance of OpticalJTCorrelator, required if name is 'optical_classical_jtc'.
    """
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
            
    elif name == "optical_classical_jtc":
        if optical_correlator is None:
            raise ValueError("optical_correlator must be provided for 'optical_classical_jtc' distance")
        
        def _d(X, Y):
            d, _, _, _ = optical_correlator.correlate(
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
