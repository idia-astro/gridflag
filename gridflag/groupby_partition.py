#!/usr/bin/python
"""Group bins and apply functions.

A set of functions for grouping data on one or two dimensional bins and
efficiently apply functions to resulting groups.  To be used with Dask
delayed for multicore and distributed processing.

Todo:
    * Rewrite as a couple of class instead of disjointed functions
    * Add error checking
    * Fix index problem at row 0 and -1
    * Implement sparse arrays for 2D bin grid
"""

import numpy as np
import numba as nb
from numba import literal_unroll

import dask.array as da
import xarray as xr

from dask.array.core import slices_from_chunks

from .annulus_stats import median_absolute_deviation

# @nb.jit(nopython=False, cache=True)
def sort_bins(uvbins, values, flags=None):
    """ Sort the input array with respect to the u and v bin indicies. We use 
    lexsort which is not an currently implemented in numba.

    Parameters
    ----------
    uvbins : int64 array of size (npoints, 3)
        A numpy array with three columns for u and v bin indicies and ms index.
    values : float32 array of size (npoints, 1)
        A one-dimensional value array.     
    flags (optional) : boolean array of size (npoints)
        An array of existing flags from MS file.

    Returns
    -------
    uvbins, values 
        Sorted input array of same name
    """
    idx = np.lexsort(np.array([uvbins[:,1], uvbins[:,0]]))  # no numba type for np.lexsort, either use nopython=False or fix lexsort implementation included below 
    uvbins = uvbins[idx]
    values = values[idx]

    if not flags is None:
        flags = flags[idx]
        # Find indicies of rows to keep
        flg_idx = np.where(flags==False)
        print("MS file flags:\t Removed {:.2f}% - {}/{} rows.".format( 
            100*(len(flags) - len(flg_idx[0]))/len(flags), (len(flags) - len(flg_idx[0])), len(values))
        )
        values = values[flg_idx]
        uvbins = uvbins[flg_idx]

    null_flags = uvbins[np.where(values==0)]

    uvbins = uvbins[np.where(values!=0)]
    print("Zero values:\t Removed {:.2f}% - {}/{} rows.".format(
        100*len(null_flags)/float(len(values)), len(null_flags), len(values))
    )

    values = values[np.where(values!=0)]

    return uvbins, values, null_flags[:,2]


# @nb.jit(nopython=False)
def sort_bins_multi(uvbins, values, flags=None):
    idx = np.lexsort(np.array([uvbins[:,1], uvbins[:,0]]))  # no numba type for np.lexsort, either use nopython=False or fix lexsort implementation included below 
    uvbins = uvbins[idx]
    values = values[idx]

    if not flags is None:
        flags = flags[idx]
        # Find indicies of rows to keep
        flg_idx = np.where(flags==False)
        print("MS file flags:\t Removed {:.2f}% - {}/{} rows.".format( 
            100*(len(flags) - len(flg_idx[0]))/len(flags), (len(flags) - len(flg_idx[0])), len(values))
        )
        values = values[flg_idx]
        uvbins = uvbins[flg_idx]

    # Filter rows with zero values (often this is done by pre-flagging)
    null_idx = np.where(np.sum(values, axis=1)==0.)
    pos_idx = np.where(np.sum(values, axis=1)!=0.)
    null_flags = uvbins[null_idx]

    uvbins = uvbins[pos_idx]
    print("Zero values:\t Removed {:.2f}% - {}/{} rows.".format(
        100*len(null_flags)/float(len(values)), len(null_flags), len(values))
    )
    values = values[pos_idx]

    return uvbins, values, null_flags[:,2]



# @nb.njit(
#     nb.types.Tuple(
#         (nb.int64[:,::1], nb.float32[:,::1], nb.int64[:,::1], nb.int64[::1])
#     )(nb.int64[:,::1], nb.float32[:,::1], nb.int64[::1]),
#     locals={
#         "ubin_prev": nb.int64,
#         "vbin_prev": nb.int64,
#         "k": nb.uint32
#     },
#     nogil=True
# )
@nb.njit(nogil=True)
def create_bin_groups_sort(uvbins, values, null_flags):
    """
    Sort U and V bins for a partition of a dataset such that uv bins are contiguous 
    in the datset. Return sorted data arrays for the bins, indicies, and values 
    and a corresponding mapping of uv bins to indexical positions in these arrays. 
    Allows for contiguous and pre-computable memory allocations for the data.
    
    Parameters
    ----------
    uvbins : dask.array
        A dask array with three columns: u and v bins and index for one partition.
    values : dask.array
        A dask array with visibility values corresponding to the uvbins rows.
    null_flags :
        List of indicies for rows with zero value, this is passed through without change.
    
    Returns
    -------
    uvbins: np.array
        A sorted array with u and v bins and indicies.
    values:
        A sorted array with values.
    grid_row_map:
        A tuple with one row for each unique UV bin pair and its starting index in the 
        uvbins and values arrays.
    null_flags:
        List of indicies for rows with zero value.
    """
    
    # Return a null set if all values in the sub-grid have been removed already
    if len(uvbins) == 0:
        grid_row_map = np.array([[0, 0, 0], [-1, -1, 1]], dtype=np.int64)
        return uvbins, values, grid_row_map, null_flags


    ubin_prev, vbin_prev = uvbins[0][0], uvbins[0][1]
    grid_row_map = [[ubin_prev, vbin_prev, 0]]
    
    k = 0
    for row in uvbins:
        if ((row[0] != ubin_prev) | (row[1] != vbin_prev)):
            ubin_prev, vbin_prev = row[0], row[1]
            grid_row_map.append([ubin_prev, vbin_prev, k])
        k = k+1

    grid_row_map.append([-1, -1, len(uvbins)]) # Add the upper index for the final row
    grid_row_map = np.array(grid_row_map, dtype=np.int64)
    
    return uvbins, values, grid_row_map, null_flags


@nb.njit(nogil=True)
def apply_grid_median(values, grid_row_map):
    """
    Apply a function broadcasting across all values in each bin within a given 
    partition. Operates on the output of the create_bin_groups_sort function.
    
    Parameters
    ----------
    values : np.array
        A sorted array with values.
    grid_row_map : np.array
        A tuple with one row for each unique UV bin pair and its starting index 
        in the uvbins and values arrays.
    
    Returns
    -------
    function_grid : array-like
        A two dimensional array of uv bins with values of the given function 
        applied to the bins.
    """
        
    function_grid = np.zeros(((np.max(grid_row_map[:,0])+1), (np.max(grid_row_map[:,1])+1)), dtype=np.float32)

    print('Applying function to grid_map.')
    for i_bin, bin_location in enumerate(grid_row_map[:-1]):
        u, v = bin_location[:2]
        istart, iend =  grid_row_map[i_bin][2], grid_row_map[i_bin+1][2]

        function_grid[u][v] = np.median(values[istart:iend])

    return function_grid


@nb.njit(nogil=True)
def apply_grid_mad(values, grid_row_map):
    """
    Apply a function broadcasting across all values in each bin within a given 
    partition. Operates on the output of the create_bin_groups_sort function.
    
    Parameters
    ----------
    values : np.array
        A sorted array with values.
    grid_row_map : np.array
        A tuple with one row for each unique UV bin pair and its starting index 
        in the uvbins and values arrays.
    
    Returns
    -------
    function_grid : array-like
        A two dimensional array of uv bins with values of the given function 
        applied to the bins.
    """
    function_grid = np.zeros(((np.max(grid_row_map[:,0])+1), (np.max(grid_row_map[:,1])+1)))

    print('Applying function to grid_map.')
    for i_bin, bin_location in enumerate(grid_row_map[:-1]):
        u, v = bin_location[:2]
        istart, iend =  grid_row_map[i_bin][2], grid_row_map[i_bin+1][2]

        function_grid[u][v] = median_absolute_deviation(values[istart:iend])

    return function_grid
    

# @nb.njit(nogil=True)
def apply_grid_function(values, grid_row_map, function):
    """
    Apply a function broadcasting across all values in each bin within a given 
    partition. Operates on the output of the create_bin_groups_sort function.
    
    Parameters
    ----------
    values : np.array
        A sorted array with values.
    grid_row_map : np.array
        A tuple with one row for each unique UV bin pair and its starting index 
        in the uvbins and values arrays.
    func : function
    
    Returns
    -------
    function_grid : array-like
        A two dimensional array of uv bins with values of the given function applied to 
        the bins.
    """
    function_grid = np.zeros(((np.max(grid_row_map[:,0])+1), (np.max(grid_row_map[:,1])+1)))

    print('Applying function to grid_map.')
    for i_bin, bin_location in enumerate(grid_row_map[:-1]):
        u, v = bin_location[:2]
        istart, iend =  grid_row_map[i_bin][2], grid_row_map[i_bin+1][2]

        function_grid[u][v] = function(values[istart:iend])

    return function_grid


# @nb.jit(nopython=True, nogil=True, cache=True)
def combine_function_partitions(median_chunks):
    """
    Combine a set of uv grid partitions in to a complete uv grid.
    
    Parameters
    ----------
    median_chunks : array-like
        A set of uv grids each containing values for mutually orthogonal 
        partitions of the full grid.

    Returns
    -------
    function_grid : array-like 
    """
    dim1 = np.max(np.array([chunk.shape[0] for chunk in median_chunks], dtype=np.int32))
    dim2 = np.max(np.array([chunk.shape[1] for chunk in median_chunks], dtype=np.int32))

    function_grid = np.zeros((dim1, dim2))
    for k, chunk in enumerate(median_chunks):
        cshape = chunk.shape
#         print(f"chunk: {k}, {cshape[0]}, {cshape[1]}")
#         print(function_grid[:cshape[0],:cshape[1]].shape)
        function_grid[:cshape[0],:cshape[1]] += chunk
    return function_grid



# ---------------------------------------------------------------------------------

def combine_annulus_results(median_grid_chunks, count_grid_chunks, flag_list_chunks, val_flag_chunks):
    '''Dask Delayed Function to concatinate chunked output data from annulus_stats functions.'''
    flag_list = np.concatenate(flag_list_chunks)
    val_flag_list = np.concatenate(val_flag_chunks)
    median_grid = combine_function_partitions(median_grid_chunks)
    count_grid = combine_function_partitions(count_grid_chunks)

    return flag_list, val_flag_list, median_grid, count_grid



# ---------------------------------------------------------------------------------



#@nb.njit(
#    nb.types.Tuple(
#        (nb.int32, nb.int32[::1], nb.int32[::1], nb.float32[:], nb.boolean[::1], nb.int64[::1])
#    )(nb.int32[::1], nb.int32[::1], nb.float32[:], nb.boolean[::1], nb.int64[::1], nb.float64),
#    locals={
#        "i": nb.uint32,
#        "j": nb.uint32,
#        "v_tmp": nb.float32[::1]
#    },
#    nogil=True
#)
@nb.njit(nogil=True, cache=True)
def partition_permutation(a, b, v, f, p, pivot):
    ''' Apply a partition to the first input array using the pivot point, p and 
    sort all the input arrays according to this partial sort.

    Parameters
    ----------
    a, b: array of int32 of shape (npoints)
        Either u or v bins 
    v: array float32 (npoints)
        Representation of visibility data such as amplitude, real, imaginary, 
        etc. - one dimensional
    p: array of int64 (npoints)
        Indicies corresponding to the order in the source measurement set for inverting.
    pivot: int32
        Pivot point to use for partial sort - all points lesser or equal with be 
        in the first part of the array.

    Returns
    -------
    i: int32
        The index position of the output arrays where the pivot exists.
    a, b, v, f, p
        The partially sorted input arrays
    '''

    i = 0
    for j in range(len(a)):
        if a[j] <= pivot:
            a[i], a[j] = a[j], a[i]
            b[i], b[j] = b[j], b[i]
            v[i], v[j] = v[j], v[i]
            f[i], f[j] = f[j], f[i]
            p[i], p[j] = p[j], p[i]
            i += 1

    return i, a, b, v, f, p
    

@nb.njit(
    nb.types.Tuple(
        (nb.int32, nb.int32[::1], nb.int32[::1], nb.float32[:,::1], nb.boolean[::1], nb.int64[::1])
    )(nb.int32[::1], nb.int32[::1], nb.float32[:,::1], nb.boolean[::1], nb.int64[::1], nb.float64),
    locals={
        "i": nb.uint32,
        "j": nb.uint32,
        "v_tmp": nb.float32[::1]
    },
    nogil=True
)
def partition_permutation_multi(a, b, v, f, p, pivot):
    ''' Apply a partition to the first input array using the pivot point, p and 
    sort all the input arrays according to this partial sort.

    Parameters
    ----------
    a, b: array of int32 of shape (npoints)
        Either u or v bins 
    v: array float32 (npoints, npol)
        Representation of visibility data such as amplitude, real, imaginary, 
        etc. - two dimensional to allow for multiple polarization states.
    p: array of int64 (npoints)
        Indicies corresponding to the order in the source measurement set for inverting.
    pivot: int32
        Pivot point to use for partial sort - all points lesser or equal with be 
        in the first part of the array.

    Returns
    -------
    i: int32
        The index position of the output arrays where the pivot exists.
    a, b, v, p
        The partially sorted input arrays
    '''

    i = 0
    for j in range(len(a)):
        if a[j] <= pivot:
            a[i], a[j] = a[j], a[i]
            b[i], b[j] = b[j], b[i]

            v_tmp = v[i].copy()
            v[i] = v[j]
            v[j] = v_tmp

            f[i], f[j] = f[j], f[i]
            p[i], p[j] = p[j], p[i]
            i += 1

    return i, a, b, v, f, p

