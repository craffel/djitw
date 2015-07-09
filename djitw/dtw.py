""" Fast DTW routines. """
import numba
import numpy as np
import parakeet


@numba.jit(nopython=True)
def band_mask(radius, mask):
    """Construct band-around-diagonal mask (Sakoe-Chiba band).  When
    ``mask.shape[0] != mask.shape[1]``, the radius will be expanded so that
    ``mask[-1, -1] = 1`` always.

    `mask` will be modified in place.

    Parameters
    ----------
    radius : float
        The band radius (1/2 of the width) will be
        ``int(radius*min(mask.shape))``.
    mask : np.ndarray
        Pre-allocated boolean matrix of zeros.

    Examples
    --------
    >>> mask = np.zeros((8, 8), dtype=np.bool)
    >>> band_mask(.25, mask)
    >>> mask.astype(int)
    array([[1, 1, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1]])
    >>> mask = np.zeros((8, 12), dtype=np.bool)
    >>> band_mask(.25, mask)
    >>> mask.astype(int)
    array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
    """
    nx, ny = mask.shape
    # The logic will be different depending on whether there are more rows
    # or columns in the mask.  Coding it this way results in some code
    # duplication but it's the most efficient way with numba
    if nx < ny:
        # Calculate the radius in indices, rather than proportion
        radius = int(round(nx*radius))
        # Force radius to be at least one
        radius = 1 if radius == 0 else radius
        for i in xrange(nx):
            for j in xrange(ny):
                # If this i, j falls within the band
                if i - j + (nx - radius) < nx and j - i + (nx - radius) < ny:
                    # Set the mask to 1 here
                    mask[i, j] = 1
    # Same exact approach with ny/ny and i/j switched.
    else:
        radius = int(round(ny*radius))
        radius = 1 if radius == 0 else radius
        for i in range(nx):
            for j in range(ny):
                if j - i + (ny - radius) < ny and i - j + (ny - radius) < nx:
                    mask[i, j] = 1


@numba.jit(nopython=True)
def dtw_core(dist_mat, add_pen, mul_pen, traceback):
    """Core dynamic programming routine for DTW.

    `dist_mat` and `traceback` will be modified in-place.

    Parameters
    ----------
    dist_mat : np.ndarray
        Distance matrix to update with lowest-cost path to each entry.
    add_pen : int or float
        Additive penalty for non-diagonal moves.
    mul_pen : int or float
        Multiplicative penalty for non-diagonal moves.
    traceback : np.ndarray
        Matrix to populate with the lowest-cost traceback from each entry.
    """
    # At each loop iteration, we are computing lowest cost to D[i + 1, j + 1]
    # TOOD: Would probably be faster if xrange(1, dist_mat.shape[0])
    for i in xrange(dist_mat.shape[0] - 1):
        for j in xrange(dist_mat.shape[1] - 1):
            # Diagonal move (which has no penalty) is lowest
            if dist_mat[i, j] <= mul_pen*dist_mat[i, j + 1] + add_pen and \
               dist_mat[i, j] <= mul_pen*dist_mat[i + 1, j] + add_pen:
                traceback[i + 1, j + 1] = 0
                dist_mat[i + 1, j + 1] += dist_mat[i, j]
            # Horizontal move (has penalty)
            elif (dist_mat[i, j + 1] <= dist_mat[i + 1, j] and
                  mul_pen*dist_mat[i, j + 1] + add_pen <= dist_mat[i, j]):
                traceback[i + 1, j + 1] = 1
                dist_mat[i + 1, j + 1] += mul_pen*dist_mat[i, j + 1] + add_pen
            # Vertical move (has penalty)
            elif (dist_mat[i + 1, j] <= dist_mat[i, j + 1] and
                  mul_pen*dist_mat[i + 1, j] + add_pen <= dist_mat[i, j]):
                traceback[i + 1, j + 1] = 2
                dist_mat[i + 1, j + 1] += mul_pen*dist_mat[i + 1, j] + add_pen


@numba.jit(nopython=True)
def dtw_core_masked(dist_mat, add_pen, mul_pen, traceback, mask):
    """Core dynamic programming routine for DTW, with an index mask, so that
    the possible paths are constrained.

    `dist_mat` and `traceback` will be modified in-place.

    Parameters
    ----------
    dist_mat : np.ndarray
        Distance matrix to update with lowest-cost path to each entry.
    add_pen : int or float
        Additive penalty for non-diagonal moves.
    mul_pen : int or float
        Multiplicative penalty for non-diagonal moves.
    traceback : np.ndarray
        Matrix to populate with the lowest-cost traceback from each entry.
    mask : np.ndarray
        A boolean matrix, such that ``mask[i, j] == 1`` when the index ``i, j``
        should be allowed in the DTW path and ``mask[i, j] == 0`` otherwise.
    """
    # At each loop iteration, we are computing lowest cost to D[i + 1, j + 1]
    # TOOD: Would probably be faster if xrange(1, dist_mat.shape[0])
    for i in xrange(dist_mat.shape[0] - 1):
        for j in xrange(dist_mat.shape[1] - 1):
            # If this point is not reachable, set the cost to infinity
            if not mask[i, j] and not mask[i, j + 1] and not mask[i + 1, j]:
                dist_mat[i + 1, j + 1] = np.inf
            else:
                # Diagonal move (which has no penalty) is lowest, or is the
                # only valid move
                if ((dist_mat[i, j] <= mul_pen*dist_mat[i, j + 1] + add_pen
                     or not mask[i, j + 1]) and
                    (dist_mat[i, j] <= mul_pen*dist_mat[i + 1, j] + add_pen
                     or not mask[i + 1, j])):
                    traceback[i + 1, j + 1] = 0
                    dist_mat[i + 1, j + 1] += dist_mat[i, j]
                # Horizontal move (has penalty)
                elif ((dist_mat[i, j + 1] <= dist_mat[i + 1, j]
                       or not mask[i + 1, j]) and
                      (mul_pen*dist_mat[i, j + 1] + add_pen <= dist_mat[i, j]
                       or not mask[i, j])):
                    traceback[i + 1, j + 1] = 1
                    dist_mat[i + 1, j + 1] += (mul_pen*dist_mat[i, j + 1] +
                                               add_pen)
                # Vertical move (has penalty)
                elif ((dist_mat[i + 1, j] <= dist_mat[i, j + 1]
                       or not mask[i, j + 1]) and
                      (mul_pen*dist_mat[i + 1, j] + add_pen <= dist_mat[i, j]
                       or not mask[i, j])):
                    traceback[i + 1, j + 1] = 2
                    dist_mat[i + 1, j + 1] += (mul_pen*dist_mat[i + 1, j] +
                                               add_pen)


def sqeuclidean(x, y):
    """Compute the squared euclidean distance between two vectors.

    Parameters
    ----------
    x, y : np.ndarray
        Vectors to compute the distance between

    Returns
    -------
    distance : float
        Squared Euclidean distance between the vectors.
    """
    return np.sum((x - y)**2)


def euclidean(x, y):
    """Compute the euclidean distance between two vectors.

    Parameters
    ----------
    x, y : np.ndarray
        Vectors to compute the distance between

    Returns
    -------
    distance : float
        Euclidean distance between the vectors.
    """
    return np.sqrt(sqeuclidean(x, y))


def cosine(x, y):
    """Compute the cosine distance between two vectors.

    Parameters
    ----------
    x, y : np.ndarray
        Vectors to compute the distance between

    Returns
    -------
    distance : float
        Cosine distance between the vectors.
    """
    return 1. - np.dot(x, y)/(np.sqrt(np.sum(x**2))*np.sqrt(np.sum(y**2)))


@parakeet.jit
def dtw_core_no_mat(x, y, dist_func, add_pen, mul_pen):
    """Core DTW routine where the distance matrix is calculated on-the-go,
    which avoids allocating it but is about 1/2 as fast.

    Parameters
    ----------
    x, y: np.ndarray
        Sequences between which the best-cost path will be found.
    dist_func : callable
        Local distance function to use.
    add_pen : int or float
        Additive penalty for non-diagonal moves.
    mul_pen : int or float
        Multiplicative penalty for non-diagonal moves.

    Returns
    -------
    last_row, last_col : np.ndarray
        The final row and column of the populated DTW cost matrix
    traceback : np.ndarray
        Path traceback matrix
    """
    # Pre-allocate row and column vectors, which will be used in lieu of a full
    # distance matrix.  These are the only rows/columns we need to keep track
    # of in the course of computing the DTW path.
    curr_row = np.empty(y.shape[0])
    prev_row = np.empty(y.shape[0])
    last_col = np.empty(x.shape[0])
    # Pre-allocate the traceback matrix.
    traceback = np.empty((x.shape[0], y.shape[0]), dtype=np.uint8)
    for j in xrange(y.shape[0]):
        # Pre-compute the distances for the first row
        curr_row[j] = dist_func(x[0], y[j])
        # Set the first row of traceback to 0, otherwise it will never be set
        # and the values from np.empty will persist
        traceback[0, j] = 0
    # Store the topmost entry of the last column
    last_col[0] = curr_row[-1]
    # For each row...
    for i in xrange(x.shape[0] - 1):
        # Set the first value of traceback for this row to 0, as above
        traceback[i + 1, 0] = 0
        # The "previous" row should be set to what used to be, on the last loop
        # iteration, the "current" row.  For now, the only value that matters
        # is the first value, the rest of the values will be populated below.
        prev_row[0] = curr_row[0]
        # Calculate the local distance for the first entry in this current row.
        # As above, only the first value matters now, the rest will be
        # populated below.
        curr_row[0] = dist_func(x[i + 1], y[0])
        # For each entry in the row...
        for j in xrange(y.shape[0] - 1):
            # Store the previously current value for this row entry as the
            # previous value.  We will be computing curr_row[j + 1] here.
            prev_row[j + 1] = curr_row[j + 1]
            # prev_row[j] being smallest corresponds to a diagonal move, which
            # has no penalty
            if prev_row[j] <= mul_pen*prev_row[j + 1] + add_pen and \
               prev_row[j] <= mul_pen*curr_row[j] + add_pen:
                traceback[i + 1, j + 1] = 0
                # Accumulate cost from the diagonally-above cell, and the cost
                # of this distance calculation
                curr_row[j + 1] = prev_row[j] + dist_func(x[i + 1], y[j + 1])
            # Vertical move, as above
            elif (prev_row[j + 1] <= curr_row[j] and
                  mul_pen*prev_row[j + 1] + add_pen <= prev_row[j]):
                traceback[i + 1, j + 1] = 1
                curr_row[j + 1] = (mul_pen*prev_row[j + 1] + add_pen
                                   + dist_func(x[i + 1], y[j + 1]))
            # Horizontal move
            elif (curr_row[j] <= prev_row[j + 1] and
                  mul_pen*curr_row[j] + add_pen <= prev_row[j]):
                traceback[i + 1, j + 1] = 2
                curr_row[j + 1] = (mul_pen*curr_row[j] + add_pen
                                   + dist_func(x[i + 1], y[j + 1]))
        # Store the last entry of this row in last_col, which we'll need later
        last_col[i + 1] = curr_row[-1]
    # We can just return curr_row as last_row
    return curr_row, last_col, traceback


def dtw(distance_matrix=None, x=None, y=None, distance_function=None, gully=1.,
        additive_penalty=0., multiplicative_penalty=1., mask=None,
        inplace=True):
    """ Compute the dynamic time warping distance between two sequences given a
    distance matrix.  The score is unnormalized.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Distances between two sequences.  If None, `x`, `y`, and
        `distance_function` will be used.
    x, y: np.ndarray
        Sequences between which the best-cost path will be found.  If None,
        `distance_matrix` will be used.
    distance_function : callable
        Local distance function to use, when `distance_matrix` is not provided.
    gully : float
        Sequences must match up to this porportion of shorter sequence. Default
        1., which means the entirety of the shorter sequence must be matched
        to part of the longer sequence.
    additive_penalty : int or float
        Additive penalty for non-diagonal moves. Default 0. means no penalty.
    multiplicative_penalty : int or float
        Multiplicative penalty for non-diagonal moves. Default 1. means no
        penalty.
    mask : np.ndarray
        A boolean matrix, such that ``mask[i, j] == 1`` when the index ``i, j``
        should be allowed in the DTW path and ``mask[i, j] == 0`` otherwise.
        If None (default), don't apply a mask - this is more efficient than
        providing a mask of all 1s.
    inplace : bool
        When ``inplace == True`` (default), `distance_matrix` will be modified
        in-place when computing path costs.  When ``inplace == False``,
        `distance_matrix` will not be modified.  Ignored when `x`, `y`, and
        `distance_function` are provided.

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
    if distance_matrix is None:
        if x is None or y is None or distance_function is None:
            raise ValueError("Either distance_matrix or x, y, and "
                             "distance_function must be supplied.")
        if mask is not None:
            raise ValueError("Masking is not (currently) supported when "
                             "computing the distance matrix on-the-fly.")
        last_row, last_column, traceback = dtw_core_no_mat(
            x, y, distance_function, additive_penalty, multiplicative_penalty)
        nx = x.shape[0]
        ny = y.shape[0]
    else:
        if np.isnan(distance_matrix).any():
            raise ValueError('NaN values found in distance matrix.')
        if not inplace:
            distance_matrix = distance_matrix.copy()
        # Pre-allocate traceback matrix
        traceback = np.empty(distance_matrix.shape, np.uint8)
        # Don't use masked DTW routine if no mask was provided
        if mask is None:
            # Populate distance matrix with lowest cost path
            dtw_core(distance_matrix, additive_penalty, multiplicative_penalty,
                     traceback)
        else:
            dtw_core_masked(distance_matrix, additive_penalty,
                            multiplicative_penalty, traceback, mask)
        nx, ny = distance_matrix.shape
        last_row = distance_matrix[-1, :]
        last_column = distance_matrix[:, -1]

    if gully < 1.:
        # Allow the end of the path to start within gully percentage of the
        # smaller distance matrix dimension
        gully = int(gully*min(nx, ny))
    else:
        # When gully is 1 require matching the entirety of the smaller sequence
        gully = min(nx, ny) - 1

    # Find the indices of the smallest costs on the bottom and right edges
    i = np.argmin(last_column[gully:]) + gully
    j = np.argmin(last_row[gully:]) + gully

    # Choose the smaller cost on the two edges
    if last_row[j] > last_column[i]:
        j = ny - 1
        score = float(last_column[i])
    else:
        i = nx - 1
        score = float(last_row[j])

    # Pre-allocate the x and y path index arrays
    x_indices = np.zeros(sum(traceback.shape), dtype=np.int)
    y_indices = np.zeros(sum(traceback.shape), dtype=np.int)
    # Start the arrays from the end of the path
    x_indices[0] = i
    y_indices[0] = j
    # Keep track of path length
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
    # Reverse and crop the path index arrays
    x_indices = x_indices[:n][::-1]
    y_indices = y_indices[:n][::-1]

    return x_indices, y_indices, score
