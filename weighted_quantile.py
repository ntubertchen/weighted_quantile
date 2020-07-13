
import collections.abc
import functools
import re
import sys
import warnings

import numpy as np
import numpy.core.numeric as _nx
from numpy.core import transpose
from numpy.core.numeric import (
    ones, zeros, arange, concatenate, array, asarray, asanyarray, empty,
    ndarray, around, floor, ceil, take, dot, where, intp,
    integer, isscalar, absolute
    )
from numpy.core.umath import (
    pi, add, arctan2, frompyfunc, cos, less_equal, sqrt, sin,
    mod, exp, not_equal, subtract
    )
from numpy.core.fromnumeric import (
    ravel, nonzero, partition, mean, any, sum
    )
from numpy.core.numerictypes import typecodes
from numpy.core.overrides import set_module
from numpy.core import overrides
from numpy.core.function_base import add_newdoc
from numpy.lib.twodim_base import diag
from numpy.core.multiarray import (
    _insert, add_docstring, bincount, normalize_axis_index, _monotonicity,
    interp as compiled_interp, interp_complex as compiled_interp_complex
    )
from numpy.core.umath import _add_newdoc_ufunc as add_newdoc_ufunc

import builtins

# needed in this module for compatibility
from numpy.lib.histograms import histogram, histogramdd


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


__all__ = [
    'select', 'piecewise', 'trim_zeros', 'copy', 'iterable', 'percentile',
    'diff', 'gradient', 'angle', 'unwrap', 'sort_complex', 'disp', 'flip',
    'rot90', 'extract', 'place', 'vectorize', 'asarray_chkfinite', 'average',
    'bincount', 'digitize', 'cov', 'corrcoef',
    'msort', 'median', 'sinc', 'hamming', 'hanning', 'bartlett',
    'blackman', 'kaiser', 'trapz', 'i0', 'add_newdoc', 'add_docstring',
    'meshgrid', 'delete', 'insert', 'append', 'interp', 'add_newdoc_ufunc',
    'quantile'
    ]

def c_quantile(a, q, w=None, axis=None, out=None,
             overwrite_input=False, interpolation='linear', keepdims=False):
    """
    Compute the q-th quantile of the data along the specified axis.
    .. versionadded:: 1.15.0
    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    q : array_like of float
        Quantile or sequence of quantiles to compute, which must be between
        0 and 1 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The
        default is to compute the quantile(s) along a flattened
        version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by intermediate
        calculations, to save memory. In this case, the contents of the input
        `a` after this function completes is undefined.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to
        use when the desired quantile lies between two data points
        ``i < j``:
            * linear: ``i + (j - i) * fraction``, where ``fraction``
              is the fractional part of the index surrounded by ``i``
              and ``j``.
            * lower: ``i``.
            * higher: ``j``.
            * nearest: ``i`` or ``j``, whichever is nearest.
            * midpoint: ``(i + j) / 2``.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.
    Returns
    -------
    quantile : scalar or ndarray
        If `q` is a single quantile and `axis=None`, then the result
        is a scalar. If multiple quantiles are given, first axis of
        the result corresponds to the quantiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.
    See Also
    --------
    mean
    percentile : equivalent to quantile, but with q in the range [0, 100].
    median : equivalent to ``quantile(..., 0.5)``
    nanquantile
    Notes
    -----
    Given a vector ``V`` of length ``N``, the q-th quantile of
    ``V`` is the value ``q`` of the way from the minimum to the
    maximum in a sorted copy of ``V``. The values and distances of
    the two nearest neighbors as well as the `interpolation` parameter
    will determine the quantile if the normalized ranking does not
    match the location of ``q`` exactly. This function is the same as
    the median if ``q=0.5``, the same as the minimum if ``q=0.0`` and the
    same as the maximum if ``q=1.0``.
    Examples
    --------
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> np.quantile(a, 0.5)
    3.5
    >>> np.quantile(a, 0.5, axis=0)
    array([6.5, 4.5, 2.5])
    >>> np.quantile(a, 0.5, axis=1)
    array([7.,  2.])
    >>> np.quantile(a, 0.5, axis=1, keepdims=True)
    array([[7.],
           [2.]])
    >>> m = np.quantile(a, 0.5, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.quantile(a, 0.5, axis=0, out=out)
    array([6.5, 4.5, 2.5])
    >>> m
    array([6.5, 4.5, 2.5])
    >>> b = a.copy()
    >>> np.quantile(b, 0.5, axis=1, overwrite_input=True)
    array([7.,  2.])
    >>> assert not np.all(a == b)
    """
    q = np.asanyarray(q)
    w = np.asanyarray(w)
    if not _quantile_is_valid(q):
        raise ValueError("Quantiles must be in the range [0, 1]")
    if w is not None and not _weights_is_valid(w):
        raise ValueError(
            "Weights must be nonnegative and total sum must be larger than 0")
    if w is not None:
        return _weighted_quantile_unchecked(
            a, q, w, axis, out, overwrite_input, interpolation, keepdims)
    else:
        return _quantile_unchecked(
            a, q, w, axis, out, overwrite_input, interpolation, keepdims)

def _weighted_ureduce(a, func, **kwargs):
    a = np.asanyarray(a)
    w = np.asanyarray(kwargs['w'])
    axis = kwargs.get('axis', None)
    # Sanity checks
    if a.shape != w.shape:
        if axis is None:
            if w.ndim != 1 or a.size != w.shape[0]:
                raise ValueError(
                    "Weights must have the same shape as a or the "
                    "shape of the desired axis.")
        else:
            axis_tuple = _nx.normalize_axis_tuple(axis, a.ndim)
            if len(axis_tuple) > 1:
                raise ValueError(
                    "Axis must be scalar when shape of a and weights are"
                    " different.")
            if w.ndim != 1:
                raise TypeError(
                    "1D weights expected when shapes of a and weights differ.")
            if w.shape[0] != a.shape[axis_tuple[0]]:
                raise ValueError(
                    "Length of weights not compatible with specified axis.")

            # setup w to broadcast along axis
            w = np.broadcast_to(w, 
                a.shape[:axis_tuple[0]] + a.shape[axis_tuple[0]+1:] + w.shape)
            w = np.moveaxis(w, -1, axis_tuple[0])
            assert a.shape == w.shape

    if axis is not None:
        keepdim = list(a.shape)
        nd = a.ndim
        axis = _nx.normalize_axis_tuple(axis, nd)

        for ax in axis:
            keepdim[ax] = 1

        if len(axis) == 1:
            kwargs['axis'] = axis[0]
            kwargs['w'] = w
        else:
            keep = set(range(nd)) - set(axis)
            nkeep = len(keep)
            # swap axis that should not be reduced to front
            for i, s in enumerate(sorted(keep)):
                a = a.swapaxes(i, s)
                w = w.swapaxes(i, s)
            # merge reduced axis
            a = a.reshape(a.shape[:nkeep] + (-1,))
            w = w.reshape(w.shape[:nkeep] + (-1,))
            kwargs['w'] = w
            kwargs['axis'] = -1
        keepdim = tuple(keepdim)
    else:
        keepdim = (1,) * a.ndim

    r = func(a, **kwargs)
    return r, keepdim

def _ureduce(a, func, **kwargs):
    """
    Internal Function.
    Call `func` with `a` as first argument swapping the axes to use extended
    axis on functions that don't support it natively.
    Returns result and a.shape with axis dims set to 1.
    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    func : callable
        Reduction function capable of receiving a single axis argument.
        It is called with `a` as first argument followed by `kwargs`.
    kwargs : keyword arguments
        additional keyword arguments to pass to `func`.
    Returns
    -------
    result : tuple
        Result of func(a, **kwargs) and a.shape with axis dims set to 1
        which can be used to reshape the result to the same shape a ufunc with
        keepdims=True would produce.
    """
    a = np.asanyarray(a)
    axis = kwargs.get('axis', None)
    if axis is not None:
        keepdim = list(a.shape)
        nd = a.ndim
        axis = _nx.normalize_axis_tuple(axis, nd)

        for ax in axis:
            keepdim[ax] = 1

        if len(axis) == 1:
            kwargs['axis'] = axis[0]
        else:
            keep = set(range(nd)) - set(axis)
            nkeep = len(keep)
            # swap axis that should not be reduced to front
            for i, s in enumerate(sorted(keep)):
                a = a.swapaxes(i, s)
            # merge reduced axis
            a = a.reshape(a.shape[:nkeep] + (-1,))
            kwargs['axis'] = -1
        keepdim = tuple(keepdim)
    else:
        keepdim = (1,) * a.ndim

    r = func(a, **kwargs)
    return r, keepdim


def _quantile_unchecked(
        a, q, w=None, axis=None, out=None, 
        overwrite_input=False, interpolation='linear', keepdims=False):
    """Assumes that q is in [0, 1], and is an ndarray"""
    r, k = _ureduce(a, func=_quantile_ureduce_func, q=q, w=w, axis=axis, 
                    out=out, overwrite_input=overwrite_input,
                    interpolation=interpolation)
    if keepdims:
        return r.reshape(q.shape + k)
    else:
        return r

def _weighted_quantile_unchecked(
        a, q, w=None, axis=None, out=None, 
        overwrite_input=False, interpolation='linear', keepdims=False):
    """Assumes that q is in [0, 1], and is an ndarray"""
    r, k = _weighted_ureduce(a, func=_weighted_quantile_ureduce_func, q=q, w=w, 
                    axis=axis, out=out, overwrite_input=overwrite_input,
                    interpolation=interpolation)
    if keepdims:
        return r.reshape(q.shape + k)
    else:
        return r

def _quantile_is_valid(q):
    # avoid expensive reductions, relevant for arrays with < O(1000) elements
    if q.ndim == 1 and q.size < 10:
        for i in range(q.size):
            if q[i] < 0.0 or q[i] > 1.0:
                return False
    else:
        # faster than any()
        if np.count_nonzero(q < 0.0) or np.count_nonzero(q > 1.0):
            return False
    return True

def _weights_is_valid(w):
    # avoid expensive reductions, relevant for arrays with < O(1000) elements
    if np.count_nonzero(w < 0) or np.sum(w) == 0:
        return False
    if np.isscalar(w):
        return False
    if np.isnan(w).any():
        return False
    return True

def _lerp(a, b, t, out=None):
    """ Linearly interpolate from a to b by a factor of t """
    diff_b_a = subtract(b, a)
    # asanyarray is a stop-gap until gh-13105
    lerp_interpolation = asanyarray(add(a, diff_b_a*t, out=out))
    subtract(b, diff_b_a * (1 - t), out=lerp_interpolation, where=t>=0.5)
    if lerp_interpolation.ndim == 0 and out is None:
        lerp_interpolation = lerp_interpolation[()]  # unpack 0d arrays
    return lerp_interpolation

def _weighted_quantile_ureduce_func(
        a, q, w=None, axis=None, out=None, overwrite_input=False, 
        interpolation='linear', keepdims=False):
    a = asarray(a)

    # ufuncs cause 0d array results to decay to scalars (see gh-13105), which
    # makes them problematic for __setitem__ and attribute access. As a
    # workaround, we call this on the result of every ufunc on a possibly-0d
    # array.
    not_scalar = np.asanyarray

    # prepare a for partitioning
    if overwrite_input:
        if axis is None:
            ap = a.ravel()
            wp = w.ravel()
        else:
            ap = a
            wp = w
    else:
        if axis is None:
            ap = a.flatten()
            wp = w.flatten()
        else:
            ap = a.copy()
            wp = w.copy()


    if axis is None:
        axis = 0

    if q.ndim > 2:
        # The code below works fine for nd, but it might not have useful
        # semantics. For now, keep the supported dimensions the same as it was
        # before.
        raise ValueError("q must be a scalar or 1d")
    q = np.atleast_1d(q)
    Nx = ap.shape[axis]
    sorted_indices = np.argsort(ap, axis=axis)
    sorted_a = np.take_along_axis(ap, sorted_indices, axis=axis)
    sorted_w = np.take_along_axis(wp, sorted_indices, axis=axis)
    sorted_a = np.moveaxis(sorted_a, axis, 0)
    sorted_w = np.moveaxis(sorted_w, axis, 0)

    sk = np.asarray(
        [k*sorted_w.take(indices=k, axis=0) + (Nx-1) * 
        np.sum(sorted_w.take(indices=range(k), axis=0), axis=0,)
        for k in range(Nx)])
    sn = sk.take(indices=(-1,), axis=0)
    normalized_w = sk/sn
    normalized_w = normalized_w.reshape((normalized_w.shape[0], -1))# (Nx, -1)
    
    sorted_indice_q = np.argsort(q)
    recover_original_q = np.argsort(sorted_indice_q)
    
    target_indices = []
    for j in range(len(normalized_w[0])):
        i = 0
        target_indice = []
        for k in sorted_indice_q:
            while i < Nx:
                if q[k] == normalized_w[i][j]:
                    target_indice.append(i)
                    break
                if i > 0:
                    if normalized_w[i-1][j] < q[k] < normalized_w[i][j]:
                        if interpolation == 'lower':
                            target_indice.append(i-1)
                        elif interpolation == 'higher':
                            target_indice.append(i)
                        elif interpolation == 'midpoint':
                            target_indice.append((2*i - 1) / 2)
                        elif interpolation == 'nearest':
                            if (q[k] - normalized_w[i-1][j]) < \
                                (normalized_w[i][j] - q[k]):
                                target_indice.append(i-1)
                            elif ((i-1)%2 == 0 and q[k] - normalized_w[i-1][j] \
                                == normalized_w[i][j] - q[k]):
                                target_indice.append(i-1)
                            else:
                                target_indice.append(i)
                        elif interpolation == 'linear':
                            target_indice.append((2*i - 1) / 2)
                        else:
                            raise ValueError(
                                "interpolation can only be 'linear', 'lower' "
                                "'higher', 'midpoint', or 'nearest'")
                        break
                i += 1  
            if i == Nx:
                target_indice.append(0)
        target_indices.append(target_indice)
    target_indices = asarray(target_indices)
    target_indices = np.moveaxis(target_indices, -1, 0)
    
    # The dimensions of `q` are prepended to the output shape, so we need the
    # axis being sampled from `ap` to be first.
    # ap = np.moveaxis(ap, axis, 0)
    # del axis

    if np.issubdtype(target_indices.dtype, np.integer):
        # take the points along axis
        target_indices = target_indices.reshape((-1,) 
        + sorted_a.shape[1:])

        if np.issubdtype(a.dtype, np.inexact):
            n = np.isnan(sorted_a[-1])
        else:
            # Cannot contain nan
            n = np.array(False, dtype=bool)

        r = np.take_along_axis(sorted_a, target_indices, axis=0, )

    else:
        # weight the points above and below the indices

        target_indices_below = floor(target_indices).astype(intp)
        target_indices_above = ceil(target_indices).astype(intp)
        # Nx x -1
        # q x -1
        Sk_differ = np.take_along_axis(
            normalized_w, target_indices_above, axis=0) \
            - np.take_along_axis(normalized_w, target_indices_below, axis=0)

        # print (q.size).reshape(target_indices_below.shape)
        q_minus_sk = np.expand_dims(q[sorted_indice_q], axis=-1) \
            - np.take_along_axis(normalized_w, target_indices_below, axis=0)
        q_minus_sk_nonzero = q_minus_sk > 0
        weights_above = np.divide(q_minus_sk , Sk_differ, 
            where=q_minus_sk_nonzero)
        weights_above[~ q_minus_sk_nonzero] = 0
        if target_indices.ndim != sorted_a.ndim:
            target_indices_below = target_indices_below.reshape((-1,)
                + sorted_a.shape[1:])
            target_indices_above = target_indices_above.reshape((-1,) 
                + sorted_a.shape[1:])
            weights_above = weights_above.reshape((-1,) 
                + sorted_a.shape[1:])
            normalized_w = normalized_w.reshape((-1,) + sorted_a.shape[1:])

        x_below = np.take_along_axis(sorted_a, target_indices_below, axis=0)
        x_above = np.take_along_axis(sorted_a, target_indices_above, axis=0)
        if np.issubdtype(a.dtype, np.inexact):
            # May contain nan, which would sort to the end
            n = np.isnan(sorted_a[-1])
        else:
            # Cannot contain nan
            n = np.array(False, dtype=bool)
        if interpolation == 'midpoint':
            r = 0.5 * (x_below+x_above)
        else:
            r = _lerp(x_below, x_above, weights_above, out=out)

    # if any slice contained a nan, then all results on that slice are also nan
    if r.ndim > 0:
        r = r[recover_original_q]

    if np.any(n):
        if r.ndim == 0 and out is None:
            # can't write to a scalar
            r = a.dtype.type(np.nan)
        else:
            r[..., n] = a.dtype.type(np.nan)

    return r
