"""jdca: GaussDCA for fast GPU inference -- a re-implementation of GaussDCA (Baldassi et al. 2014) in JAX and BLAS."""
from .alignment import read_alignment
from .gauss import DCA, apc, gaussdca
from .ridge import humphreys_dca, ridge_dca
from .weights import identity_counts, mean_identity, sequence_weights

__version__ = '0.1.1'


def ranking(scores, min_separation=5):
    """Position pairs (i < j, j - i >= min_separation; 0-based) sorted by decreasing score: (i, j, score) arrays."""
    import numpy as np
    i, j = np.triu_indices(scores.shape[0], min_separation)
    s = scores[i, j]
    keep = np.isfinite(s)                                   # a windowed map leaves pairs beyond the window unscored (NaN)
    i, j, s = i[keep], j[keep], s[keep]
    o = np.argsort(-s, kind='stable')
    return i[o], j[o], s[o]


__all__ = ['DCA', 'apc', 'gaussdca', 'humphreys_dca', 'identity_counts', 'mean_identity', 'ranking', 'read_alignment',
           'ridge_dca', 'sequence_weights']
