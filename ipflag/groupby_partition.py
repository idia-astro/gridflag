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

@nb.jit(nopython=False, nogil=True, cache=True)
def create_bin_groups_sort(uvbins, values):
    """
    Sort U and V bins for a partition of a dataset such that uv bins are contiguous in the
    datset. Return sorted data arrays for the bins, indicies, and values and a 
    corresponding mapping of uv bins to indexical positions in these arrays. Allows for 
    contiguous and pre-computable memory allocations for the data.
    
    Parameters
    ----------
    uvbins: dask.array
        A dask array with three columns: u and v bins and index for one partition.
    values: dask.array
        A dask array with visibility values corresponding to the uvbins rows.
    
    Returns
    -------
    uvbins: np.array
        A sorted array with u and v bins and indicies.
    values:
        A sorted array with values.
    grid_row_map:
        A tuple with one row for each unique UV bin pair and its starting index in the 
        uvbins and values arrays.
    """
    idx = np.lexsort(np.array([uvbins[:,1], uvbins[:,0]]))  # no numba type for np.lexsort, either use nopython=False or fix lexsort implementation included below 
    uvbins = uvbins[idx]
    values = values[idx]
    
    
    # Remove rows with zero values (and return flag indicies if requested)
#     print("Zero values removed: {:.2f}%".format(100*len(np.where(values==0)[0])/float(len(values))))
    null_flags = uvbins[np.where(values==0)]
    print("Zero values removed: {:.2f}%".format(100*len(null_flags)/float(len(values))))

    uvbins = uvbins[np.where(values!=0)]
    values = values[np.where(values!=0)]

    ubin_prev, vbin_prev = uvbins[0][0], uvbins[0][1]
    grid_row_map = [[ubin_prev, vbin_prev, 0]]

#     print(f"init: ({ubin_prev}, {vbin_prev})")

    k = 0
    for row in literal_unroll(uvbins):
        if ((row[0] != ubin_prev) | (row[1] != vbin_prev)):
            ubin_prev, vbin_prev = row[0], row[1]
            grid_row_map.append([ubin_prev, vbin_prev, k])
        k = k+1

# Generates numba error:

#     for k in range(len(uvbins)):
#         if ((uvbins[k][0] != ubin_prev) | (uvbins[k][1] != vbin_prev)):
#             ubin_prev, vbin_prev = uvbins[k][0], uvbins[k][1]
#             grid_row_map.append([ubin_prev, vbin_prev, k])

#     for k, row in enumerate(zip(uvbins, values)):
#         if ((row[0][0] != ubin_prev) | (row[0][1] != vbin_prev)):
#             ubin_prev, vbin_prev = row[0][0], row[0][1]
#             grid_row_map.append([ubin_prev, vbin_prev, k])


    grid_row_map.append([-1, -1, len(uvbins)]) # Add the upper index for the final row
    grid_row_map = np.array(grid_row_map, dtype=np.int64)
    return uvbins, values, grid_row_map, null_flags[:,2]


@nb.jit(nopython=True, nogil=True, cache=True)
def apply_grid_median(values, grid_row_map):
    """
    Apply a function broadcasting across all values in each bin within a given partition. 
    Operates on the output of the create_bin_groups_sort function.
    
    Parameters
    ----------
    values : np.array
        A sorted array with values.
    grid_row_map : np.array
        A tuple with one row for each unique UV bin pair and its starting index in the 
        uvbins and values arrays.
    func : function
    
    Returns
    -------
    function_grid : array-like
        A two dimensional array of uv bins with values of the given function applied to 
        the bins.
    """
    function_grid = np.zeros(((np.max(grid_row_map[:,0])+1), (np.max(grid_row_map[:,1])+1)))
    u_prev = 0

    print('Applying function to grid_map.')
    for i_bin, bin_location in enumerate(grid_row_map[:-1]):
        u, v = bin_location[:2]
        istart, iend =  grid_row_map[i_bin][2], grid_row_map[i_bin+1][2]

        function_grid[u][v] = np.median(values[istart:iend])

    return function_grid


def apply_grid_function(values, grid_row_map, function):
    """
    Apply a function broadcasting across all values in each bin within a given partition. 
    Operates on the output of the create_bin_groups_sort function.
    
    Parameters
    ----------
    values : np.array
        A sorted array with values.
    grid_row_map : np.array
        A tuple with one row for each unique UV bin pair and its starting index in the 
        uvbins and values arrays.
    func : function
    
    Returns
    -------
    function_grid : array-like
        A two dimensional array of uv bins with values of the given function applied to 
        the bins.
    """
    function_grid = np.zeros(((np.max(grid_row_map[:,0])+1), (np.max(grid_row_map[:,1])+1)))
    u_prev = 0

    print('Applying function to grid_map.')
    for i_bin, bin_location in enumerate(grid_row_map[:-1]):
        u, v = bin_location[:2]
        istart, iend =  grid_row_map[i_bin][2], grid_row_map[i_bin+1][2]

        function_grid[u][v] = function(values[istart:iend])

    return function_grid


@nb.jit(nopython=True, nogil=True, cache=True)
def combine_function_partitions(median_chunks):
    """
    Simply combine a set of grid partitions in to a full uv grid.
    
    Parameters
    ----------
    median_chunks : array-like
        A set of uv grids each containing values for mutually orthogonal partitions of the 
        full grid.

    Returns
    -------
    function_grid : array-like 
    """
    dim1 = np.max(np.array([chunk.shape[0] for chunk in median_chunks], dtype=np.int32))
    dim2 = np.max(np.array([chunk.shape[1] for chunk in median_chunks], dtype=np.int32))
    print(dim1, dim2)
    function_grid = np.zeros((dim1, dim2))
    for k, chunk in enumerate(median_chunks):
        cshape = chunk.shape
#         print(f"chunk: {k}, {cshape[0]}, {cshape[1]}")
#         print(function_grid[:cshape[0],:cshape[1]].shape)
        function_grid[:cshape[0],:cshape[1]] += chunk
    return function_grid


@nb.jit(nopython=True, nogil=True, cache=True)
def combine_grid_partitions(uvbin_chunks, value_chunks, grid_row_map_chunks, median_grid):
    """
    Simply combine a set of grid partitions in to a full uv grid.
    
    Parameters
    ----------
    median_chunks : array-like
        A set of uv grids each containing values for mutually orthogonal partitions of the 
        full grid.

    Returns
    -------
    function_grid : array-like 
    """
    function_grid = np.zeros(median_grid.shape)

    for k, chunk in enumerate(median_chunks):
        cshape = chunk.shape
#         print(f"chunk: {k}, {cshape[0]}, {cshape[1]}")
#         print(function_grid[:cshape[0],:cshape[1]].shape)
        function_grid[:cshape[0],:cshape[1]] += chunk
    
    return function_grid



@nb.jit(nopython=True, nogil=True, cache=True)
def combine_grid_partitions(value_chunks, grid_row_maps, function_grid):
    """
    Return two uv-grid data structures where each UV cell contains either a list of values
    or a list of indicies used to map to the original measurement set.
    """

    y_max, x_max = len(function_grid), len(function_grid[0])

    idx_list_cat = [[[]for j in range(x_max)] for i in range(y_max)]
    val_list_cat = [[[]for j in range(x_max)] for i in range(y_max)]

    for value_chunk, grid_row_map in zip(value_chunks, grid_row_maps):
        for i_bin, bin_location in enumerate(grid_row_map[:-1]):
            u, v = bin_location[:2]

            istart, iend =  grid_row_map[i_bin][2], grid_row_map[i_bin+1][2]

            idx_list_cat[u][v] = np.arange(istart, iend)
            val_list_cat[u][v] = value_chunk[istart:iend]

    idx_list_cat, val_list_cat = np.array(idx_list_cat), np.array(val_list_cat)

    return (idx_list_cat, val_list_cat)



# ---------------------------------------------------------------------------------


def compute_annulus_stats(median_grid, value_group, bin_group, grid_row_map, annulus_width, sigma=3.):

    median_grid_flg = np.zeros(median_grid.shape)
    value_groups_flg = [[[] for r_ in c_] for c_ in value_groups]
    flag_ind_list = []

    for ind, edge in enumerate(annulus_width):
        minuv=0
        if ind:
            minuv = annulus_width[ind-1]
        maxuv = annulus_width[ind]

        print(f"Annulus ({minuv}-{maxuv}): ")
        
        ann_bin_val, ann_bin_ind, ann_bin_name, bin_flag_ind = select_annulus_bins(minuv, maxuv, value_groups, index_groups, median_grid, uvbins)

        
def combine_annulus_results(median_chunks, flag_chunks):

    flag_list = np.concatenate(flag_chunks)
    median_grid = combine_function_partitions(median_chunks)

    return flag_list, median_grid
        

# ---------------------------------------------------------------------------------


@nb.jit(nopython=True, nogil=True, cache=True)
def partition_permutation(a, pivot, p):
    # workaround to get numba working for empty lists

    i = 0
    for j in range(len(a)):
        if a[j] <= pivot:
            p[i], p[j] = p[j], p[i]
            a[i], a[j] = a[j], a[i]
            i += 1
    return p, i



@nb.jit(nopython=True, nogil=True, cache=True)
def binary_partition(a, binary_chunks, partition_level, p):
    """
    Automatically partition array in to a given number of chunks

    a : array-like
        The input array, will be sorted in-place.
    binary_chunks : int
        The the desired number of chunks given as a power of two, so for 16 chunks set binary_chunks=4 or
        binary_chunks = log(chunks).
    partition_level : int
        The current partition level for recursion. Should be set to zero, but no default 
        value is set due to numba.
    p : array-like
        The permutation array to be returned and used to sort the original dataset. Should 
        be set to a contiguous integer index array with length a, e.g. p = 
        np.arange(len(a), dtype=np.int64).
    """
    split_points = []

#     a = np.nan_to_num(a)
#     med = np.median(a[a!=0])

    pivot = int(np.median(a))
#     print("median pivot and partition level:", pivot, partition_level)

    p, split_point = partition_permutation(a, pivot, p)

#     print("partition length \t split point", len(p), split_point)

    partition_level+=1

#     print(partition_level, binary_chunks, pivot, split_point)

    if partition_level < binary_chunks:

        p1, sp1 = binary_partition(a[:split_point], binary_chunks, partition_level, p[:split_point])
        p2, sp2 = binary_partition(a[split_point:], binary_chunks, partition_level, p[split_point:])
#         print(len(p), type(p), len(p1), len(p2), type(p1), type(p2))

        split_points += sp1
        split_points.append(split_point),
        split_points += [sp+split_point for sp in sp2]
        p = np.concatenate((p1, p2))
    else:
        split_points.append(split_point)

    return p, split_points
    
    

    
@nb.jit(nopython=False, nogil=True, cache=True)
def partition_permutation_2d(a, b, v, p, pivot):
    # workaround to get numba working for empty lists    
    a = np.copy(a)
    b = np.copy(b)
    v = np.copy(v)
    p = np.copy(p)
    i = 0
    for j in range(len(a)):
        if a[j] <= pivot:
            a[i], a[j] = a[j], a[i]
            b[i], b[j] = b[j], b[i]
            v[i], v[j] = v[j], v[i]
            p[i], p[j] = p[j], p[i]
            i += 1
    return i, a, b, v, p

    
@nb.jit(nopython=False, nogil=True, cache=True)
def binary_partition_2d(a, b, binary_chunks, partition_level, p, v):
    """
    Automatically partition array in to a given number of chunks

    a : array-like
        The input array, will be sorted in-place.
    binary_chunks : int
        The the desired number of chunks given as a power of two, so for 16 chunks set binary_chunks=4 or
        binary_chunks = log(chunks).
    partition_level : int
        The current partition level for recursion. Should be set to zero, but no default 
        value is set due to numba.
    p : array-like
        The permutation array to be returned and used to sort the original dataset. Should 
        be set to a contiguous integer index array with length a, e.g. p = 
        np.arange(len(a), dtype=np.int64).
    """
    split_points = []

    pivot = int(np.median(a))
    print("Partition level:", partition_level)

    if pivot == max(a):
        pivot-=1
    if pivot == min(a):
        pivot+=1
        
    a, b, v, p, split_point = partition_permutation_2d(a, b, pivot, p, v)

    print("\tpivot, partition min/max", pivot, split_point, min(a), max(a))

    partition_level+=1
    
    if partition_level < binary_chunks:
        p1, sp1 = binary_partition_2d(b[:split_point], a[:split_point], binary_chunks, partition_level, p[:split_point], v[:split_point])
        p2, sp2 = binary_partition_2d(b[split_point:], a[split_point:], binary_chunks, partition_level, p[split_point:], v[split_point:])

        split_points += sp1
        split_points.append(split_point),
        split_points += [sp+split_point for sp in sp2]
        p = np.concatenate((p1, p2))

    else:
        split_points.append(split_point)

    return p, split_points


def process_bin_partitions(ds_ind, sort_data):
    sp = sort_data[4]
    
    split_chunks = [0] + sp + [len(ds_ind.newrow)]
    split_chunks = tuple(np.diff(split_chunks))
    
    dd_bins = da.stack([sort_data[0], sort_data[1], sort_data[3]]).T
    dd_vals = da.from_array(sort_data[2])
    
    dd_bins = dd_bins.rechunk((split_chunks, (3)))
    dd_vals = dd_vals.rechunk((split_chunks, (1)))

    dd_bins = dd_bins.to_delayed()
    dd_vals = dd_vals.to_delayed()
    
    return dd_bins, dd_vals

# ---------------------------------------------------------------------------------

# @nb.jit(nopython=True, nogil=True, cache=True)
def apply_partition_chunk(p, sp, ds_ind): #p, 
    """
    Apply permutation index to XArray datset by first applying chunk-by-chunk, then shuffling the
    chunks. Theoretically, this reduces the number of dask tasks, getitem/concatination operations.
    
    
    Notes:
        Some logic from https://github.com/dask/dask/pull/3901/files
    """
    
    chunks = ds_ind.U_bins.chunks    
    slices = slices_from_chunks(chunks)
    print(chunks)
    
    offsets = np.roll(np.cumsum(chunks[0]), 1)
    offsets[0] = 0
    
    
    print("Compute chunk-pivot matrix.")
    # Compute pivot points per Chunk matrix
    pivot_chunks = []
    p_chunk = []
    for slice_, offset in zip(slices, offsets):
        chunksize = slice_[0].stop - slice_[0].start
        print(slice_, offset, chunksize)
        p_chunk.append(p[(slice_[0].start<=p)&(slice_[0].stop>p)])
        sp_chunks = []
        for sp_ in zip([0]+sp[:-1],sp):
            pc = p[sp_[0]:sp_[1]]
            sp_c = len(pc[(slice_[0].start<=pc)&(pc<slice_[0].stop)])
            sp_chunks.append(sp_c)
#             print("\t", sp_, len(pc), sp_c)
        sp_l = [0] + list(np.cumsum(sp_chunks)) + [chunksize]
        pivot_chunks.append(sp_l)
    
    pivot_chunks = np.array(pivot_chunks)

    print("Create chunked permutation index.")
    # Re-parse the permutation index to do in-chunk permutations
#     p_chunk = np.concatenate([p[(slice_[0].start<=p)&(slice_[0].stop>p)] for slice_ in slices])
    p_chunk = np.concatenate(p_chunk)

    print("Reindex the datset")

    ds_ind = ds_ind.isel(newrow=p_chunk)

    del(p_chunk)
    print("Re-order and re-chunk the dataset.")
    new_chunks = []
    for chunk_pivots in zip(pivot_chunks.T[:-1], pivot_chunks.T[1:]):
        ds_chk = xr.concat([ds_ind.isel(newrow=slice(offset+a,offset+b,None)) for offset,a,b in zip(offsets, chunk_pivots[0], chunk_pivots[1])], 'newrow')
        ds_chk = ds_chk.chunk({'newrow':len(ds_chk.newrow)})
        new_chunks.append(ds_chk)

    ds_ind = xr.concat(new_chunks, 'newrow')

    print("Set chunk sizes to partition sizes.")
    del(new_chunks)
    split_chunks = [0] + sp + [len(ds_ind.newrow)]
    split_chunks = tuple(np.diff(split_chunks))
    ds_ind = ds_ind.chunk({'newrow':split_chunks})
    
    return ds_ind

# ---------------------------------------------------------------------------------




@nb.jit
def cmp_fn(l, r, *arrays):
    for a in literal_unroll(arrays):
        if a[l] < a[r]:
            return -1  # less than
        elif a[l] > a[r]:
            return 1   # greater than

    return 0  # equal


@nb.jit
def quicksort(index, L, R, *arrays):
    l, r = L, R
    pivot = index[(l + r) // 2]

    while True:
        while l < R and cmp_fn(index[l], pivot, *arrays) == -1:
            l += 1
        while r >= L and cmp_fn(pivot, index[r], *arrays) == -1:
            r -= 1

        if l >= r:
            break

        index[l], index[r] = index[r], index[l]
        l += 1
        r -= 1

        if L < r:
            quicksort(index, L, r, *arrays)

        if l < R:
            quicksort(index, l, R, *arrays)


@nb.jit(nogil=True, cache=True, debug=True)
def lexsort(arrays):
    print("starting lexsort")

    if len(arrays) == 0:
        return np.empty((), dtype=np.intp)

    if len(arrays) == 1:
        return np.argsort(arrays[0])

    for a in literal_unroll(arrays[1:]):
        if a.shape != arrays[0].shape:
            raise ValueError("lexsort array shapes don't match")

    n = arrays[0].shape[0]
    index = np.arange(n)

    quicksort(index, 0, n - 1, *arrays)

    print("ending lexsort")
    return index


# ---------------------------------------------------------------------------------


def dict_apply_grid_function(values, grid_row_map, func=np.median):
    function_grid = {}
    u_prev = 0

    for i_bin, bin_location in enumerate(grid_row_map):
        u, v = bin_location[:2]
        istart, iend =  grid_row_map[i_bin][2], grid_row_map[i_bin+1][2]

        print(u, v, istart, iend, func(values[istart:iend]))

        if u!=u_prev:
            u_prev = u
            function_grid[u] = {v:func(values[istart:iend])}
        else:
            function_grid[u][v] = func(values[istart:iend])

    return function_grid



def create_bin_groups(ubins, vbins, values, indicies):

    ubin_prev, vbin_prev = ubins[0], vbins[0]
    grid_row_map = [[ubins[0], vbins[0], 0]]

    print(f"init: ({ubin_prev}, {vbin_prev})")

    for k, row in enumerate(zip(ubins, vbins, values, indicies)):
        if ((row[0] != ubin_prev) | (row[1] != vbin_prev)):
            ubin_prev, vbin_prev = row[0], row[1]
            grid_row_map.append([row[0], row[1], k])

    return  grid_row_map


def partition_partial(a, l, r):
    x = a[r]
    i = l - 1
    for j in range(l, r):
        if a[j] <= x:
            i = i + 1
            a[i], a[j] = a[j], a[i]
    a[i + 1], a[r] = a[r], a[i + 1]
    return i + 1

def partition_pivot(a, l, r, pivot):
    x = pivot
    i = l - 1
    for j in range(l, r):
        if a[j] <= x:
            i = i + 1
            a[i], a[j] = a[j], a[i]
    a[i + 1], a[r] = a[r], a[i + 1]
    return i + 1

def partition(a, pivot):
    i = 0
    for j in range(len(a)):
        if a[j] <= pivot:
            a[i], a[j] = a[j], a[i]
            i += 1
    return i


def binary_tree_chunk(a, p, chunks, k_chunks=1, split_points=[], init_index=0):
    # change to median of medians or empirical estimate of median (from baseline distribution, etc)
    pivot = np.median(a) - 1
    print(pivot)
    k_chunks += 1

    p, split_point = partition_permutation(a, pivot)
    split_points.append(split_point+init_index)

    print(k_chunks, chunks, split_points)
    if k_chunks < chunks:
        k_chunks += 2
        binary_tree_chunk(a[:split_point], chunks, k_chunks, split_points, init_index=init_index)
        binary_tree_chunk(a[split_point+1:], chunks, k_chunks, split_points, init_index=split_point+init_index)

    return split_points


def kthlargest(a, k):
    l = 0
    r = len(a) - 1
    print(f"Partition(a[{len(a)}], l={l}, r={r})")
    split_point = partition(a, l, r) #choosing a pivot and saving its index
    if split_point == r - k + 1: #if the chosen pivot is the correct elemnt, then return it
        result = a[split_point]
    elif split_point > r - k + 1: #if the element we are looking for is in the left part to the pivot then call 'kthlargest' on that part after modifing k
        result = kthlargest(a[:split_point], k - (r - split_point + 1))
    else: #if the element we are looking for is in the left part to the pivot then call 'kthlargest' on that part
        result = kthlargest(a[split_point + 1:r + 1], k)
    return result

