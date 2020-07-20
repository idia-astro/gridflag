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

import dask.array as da

def create_bin_groups_sort(uvbins, values):

    idx = np.lexsort([ uvbins[:,1], uvbins[:,0]])
    uvbins = uvbins[idx]
    values = values[idx]

    ubin_prev, vbin_prev = uvbins[0][0], uvbins[0][1]
    grid_row_map = [[ubin_prev, vbin_prev, 0]]

    print(f"init: ({ubin_prev}, {vbin_prev})")

    for k, row in enumerate(zip(uvbins, values)):
        if ((row[0][0] != ubin_prev) | (row[0][1] != vbin_prev)):
            ubin_prev, vbin_prev = row[0][0], row[0][1]
            grid_row_map.append([ubin_prev, vbin_prev, k])

    grid_row_map.append([-1, -1, len(uvbins)]) # Add the upper index for the final row
    grid_row_map = np.array(grid_row_map)
    return uvbins, values, grid_row_map


@nb.jit(nopython=True, nogil=True, cache=True)
def apply_grid_function(values, grid_row_map, func=np.median):
    function_grid = np.zeros((np.max(grid_row_map[:,0])+1, np.max(grid_row_map[:,1])+1))
    u_prev = 0

    print('Applying function to grid_map.')
    for i_bin, bin_location in enumerate(grid_row_map[:-1]):
        u, v = bin_location[:2]
        istart, iend =  grid_row_map[i_bin][2], grid_row_map[i_bin+1][2]

        function_grid[u][v] = func(values[istart:iend])

    return function_grid


@nb.jit(nopython=True, nogil=True, cache=True)
def combine_grid_partitions(value_chunks, grid_row_maps, median_grid):
    """
    Return two uv-grid data structures where each UV cell contains either a list of values
    or a list of indicies used to map to the original measurement set.
    """

    y_max, x_max = len(median_grid), len(median_grid[0])

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


def combine_function_partitions(median_chunks):
    dim1 = np.max([chunk.shape[0] for chunk in median_chunks])
    dim2 = np.max([chunk.shape[1] for chunk in median_chunks])
    print(dim1, dim2)
    median_grid = np.zeros((dim1, dim2))
    for k, chunk in enumerate(median_chunks):
        cshape = chunk.shape
        print(f"chunk: {k}, {cshape[0]}, {cshape[1]}")
        print(median_grid[:cshape[0],:cshape[1]].shape)
        median_grid[:cshape[0],:cshape[1]] += chunk
    return median_grid


@nb.jit(nopython=True, nogil=True, cache=True)
def binary_partition(a, binary_chunks, partition_level, p):
    """
    Automatically partition array in to a given number of chunks

    a : array-like
        The input array, will be sorted in-place.
    binary_chunks : int
        The the desired number of chunks given as a power of two, so for 16 chunks set binary_chunks=4 or
        binary_chunks = log(chunks).
    p : array-like
        The permutation array to be returned and used to sort the original dataset.
    """
    split_points = []

    a = np.nan_to_num(a)
    med = np.median(a[a!=0])
    print("median is ", med)

    pivot = int(med)
    p, split_point = partition_permutation(a, pivot, p)

    partition_level+=1

    print(partition_level, binary_chunks, pivot, split_point)

    if partition_level < binary_chunks:

        p1, sp1 = binary_partition(a[:split_point], binary_chunks, partition_level, p[:split_point])
        p2, sp2 = binary_partition(a[split_point:], binary_chunks, partition_level, p[split_point:])

        split_points += sp1
        split_points.append(split_point),
        split_points += [sp+split_point for sp in sp2]
        p = np.concatenate([p1, p2])
    else:
        split_points.append(split_point)

    return p, split_points


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

