""" Fast DTW routines. """
import numba
import numpy as np


@numba.jit(nopython=True)
def dtw_core(dist_matrix, add_pen, traceback):
    """Core dynamic programming routine for DTW.

    `dist_matrix` and `traceback` will be modified in-place.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Distance matrix to update with lowest-cost path to each entry.
    add_pen : int or float
        Additive penalty for non-diagonal moves.
    traceback : np.ndarray
        Matrix to populate with the lowest-cost traceback from each entry.
    """
    # At each loop iteration, we are computing lowest cost to D[i + 1, j + 1]
    # TOOD: Would probably be faster if xrange(1, dist_matrix.shape[0])
    for i in xrange(dist_matrix.shape[0] - 1):
        for j in xrange(dist_matrix.shape[1] - 1):
            # Diagonal move (which has no penalty) is lowest
            if dist_matrix[i, j] <= dist_matrix[i, j + 1] + add_pen and \
               dist_matrix[i, j] <= dist_matrix[i + 1, j] + add_pen:
                traceback[i + 1, j + 1] = 0
                dist_matrix[i + 1, j + 1] += dist_matrix[i, j]
            # Horizontal move (has penalty)
            elif (dist_matrix[i, j + 1] <= dist_matrix[i + 1, j] and
                  dist_matrix[i, j + 1] + add_pen <= dist_matrix[i, j]):
                traceback[i + 1, j + 1] = 1
                dist_matrix[i + 1, j + 1] += dist_matrix[i, j + 1] + add_pen
            # Vertical move (has penalty)
            elif (dist_matrix[i + 1, j] <= dist_matrix[i, j + 1] and
                  dist_matrix[i + 1, j] + add_pen <= dist_matrix[i, j]):
                traceback[i + 1, j + 1] = 2
                dist_matrix[i + 1, j + 1] += dist_matrix[i + 1, j] + add_pen


@numba.jit(nopython=True)
def path_from_traceback(traceback, x_indices, y_indices):
    """Extracts the index path from the traceback matrix.

    Parameters
    ----------
    traceback : np.ndarray
        Traceback matrix.  Each entry should be 0, 1, or 2, where 1 corresponds
        to a diagonal move, 1 corresponds to a horizontal move, and 2
        corresponds to a vertical move.
    x_indices : np.ndarray
        First dimension path indices; x_indices[0] should be provided with the
        end of the path.  Will be modified by this function.
    y_indices : np.ndarray
        Same as x_indices for second dimension.
    """
    i = x_indices[0]
    j = y_indices[0]
    n = 1

    # Until we reach an edge
    while i > 0 and j > 0:
        # If the tracback matrix indicates a diagonal move...
        if traceback[i, j] == 0:
            i = i - 1
            j = j - 1
        # Horizontal move...
        elif traceback[i, j] == 1:
            i = i - 1
        # Vertical move...
        elif traceback[i, j] == 2:
            j = j - 1
        # Add these indices into the path arrays
        x_indices[n] = i
        y_indices[n] = j
        n += 1
    return x_indices[:n], y_indices[:n]


def dtw(distance_matrix, gully, penalty):
    """ Compute the dynamic time warping distance between two sequences given a
    distance matrix.  The score is normalized by the path length.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Distances between two sequences.
    gully : float
        Sequences must match up to this porportion of shorter sequence.
    penalty : int
        Non-diagonal move penalty.

    Returns
    -------
    x_indices : np.ndarray
        Indices of the lowest-cost path in the first dimension of the distance
        matrix.
    y_indices : np.ndarray
        Indices of the lowest-cost path in the second dimension of the distance
        matrix.
    score : float
        DTW score of lowest cost path through the distance matrix, including
        penalties.
    """
    # Pre-allocate path length matrix
    traceback = np.empty(distance_matrix.shape, distance_matrix.dtype)
    # Populate distance matrix with lowest cost path
    dtw_core(distance_matrix, penalty, traceback)
    # Traceback from lowest-cost point on bottom or right edge
    gully = int(gully*min(distance_matrix.shape[0], distance_matrix.shape[1]))
    i = np.argmin(distance_matrix[gully:, -1]) + gully
    j = np.argmin(distance_matrix[-1, gully:]) + gully

    if distance_matrix[-1, j] > distance_matrix[i, -1]:
        j = distance_matrix.shape[1] - 1
    else:
        i = distance_matrix.shape[0] - 1

    # Score is the final score of the best path
    score = float(distance_matrix[i, j])

    x_indices = np.zeros(distance_matrix.shape[0], dtype=np.int)
    x_indices[0] = i
    y_indices = np.zeros(distance_matrix.shape[1], dtype=np.int)
    y_indices[0] = j
    x_indices, y_indices = path_from_traceback(traceback, x_indices, y_indices)

    return x_indices[::-1], y_indices[::-1], score
