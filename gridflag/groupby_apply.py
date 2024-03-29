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
import scipy.constants
import dask.array as da

def groupby_nd(bins, init_index:int=0) -> tuple:
    '''
    Performs a groupby operation on two bin parameters and returns the list
    of indicies for mapping the unique groups to their positions.

    Parameters
    ----------
    bins: array-like
        An array or tuple containing two lists of bins values for two the
        parameters to be binned.
    init_index: int, optional
        Starting value for the indicies to use when iterating or chunking
        large lists of bins.

    Returns
    -------
    sub-arrays : list of ndarrays
        A list of arrays with indicies of bins.

    '''
    bin_group = []
    for bin_col in bins:
        bin_group.append(np.unique(bin_col, return_inverse=True))
        
    sorted_keys, bins_as_ints = [bg[0] for bg in bin_group], [bg[1] for bg in bin_group]

    n_bins = [max(skeys) + 1 for skeys in sorted_keys]

    indicies = [[[]for j in range(n_bins[1])] for i in range(n_bins[0])]

    if init_index:
        for i, k_ in enumerate(zip(bins[0], bins[1])):
            indicies[k_[0]][k_[1]].append(i+init_index)
    else:
        for i, k_ in enumerate(zip(bins[0], bins[1])):
            indicies[k_[0]][k_[1]].append(i)

    indicies = [np.array([np.array(col) for col in row]) for row in indicies]
    return indicies


def groupby_nd_wrap(dd_bins, init_index=0):
    """
    Wrap the groupby_nd function to parse dask arrays to numpy-like.

    Parameters
    ----------
    dd_bins: array-like
        A 2d dask array of integer bin values to be grouped.

    """
    bins = [dd_bins[:,0], dd_bins[:,1]]
    res = groupby_nd(bins, init_index)
    return res


def run_function_chunked(bins, chunksize=10**6):
    """
    Compute bin groups using only numpy.

    Parameters
    ----------
    bins: array-like
        A 2d array of integer bin values to be grouped.
    chunksize: int
        The chunk size to be used to split the bin data.
    """
    nvals = bins[0].shape[0]
    echunks = np.arange(0, nvals, chunksize)

    ind_arr = []
    for l, h in zip(echunks[0:-1], echunks[1:]):
        print(f'computing rows {l}-{h}')
        inds = groupby_nd([bins[0][l:h].compute(), bins[1][l:h].compute()], init_index=l)
        ind_arr.append(inds)
    return ind_arr


def find_ind_ranges(group_idx):
    '''
    Reduce bin indicies to index ranges.  Used to determine index ranges
    for each bin to determine block size or for assessing sortability.


    Parameters
    ----------
    ind_set: array-like
        Array with shape (K, N1, N2) where K is the number of blocks, (N1,
        N2) are the number of bins in each dimension.

    Returns
    -------
    sub-arrays : list of ndarrays
        A list of arrays with index ranges for each bin.
    '''

    group_ranges = [[[0, 0] for j in range(len(group_idx[0]))] for i in range(len(group_idx))]

    for i, inds in enumerate(group_idx):
        for j, inds_ in enumerate(inds):
            if len(inds_):
                group_ranges[i][j][0] = min(inds_)
                group_ranges[i][j][1] = max(inds_)

    group_ranges = [np.array([np.array(col) for col in row]) for row in group_ranges]
    return group_ranges


def group_bin_values(bins, values) -> tuple:
    '''
    Performs a groupby operation on two bin parameters and returns the list
    of indicies for mapping the unique groups to their positions.

    Parameters
    ----------
    bins: array-like
        An array or tuple containing two lists of bins values for two the
        parameters to be binned.
    init_index: int, optional
        Starting value for the indicies to use when iterating or chunking
        large lists of bins.

    Returns
    -------
    sub-arrays : list of ndarrays
        A list of arrays with indicies of bins.

    '''

    bin_group = []
    for bin_col in bins:
        bin_group.append(np.unique(bin_col, return_inverse=True))

    sorted_keys, bins_as_ints = [bg[0] for bg in bin_group], [bg[1] for bg in bin_group]

    n_bins = [int(max(skeys) + 1) for skeys in sorted_keys]

    indicies = [[[]for j in range(n_bins[1])] for i in range(n_bins[0])]

    for i, k_ in enumerate(zip(bins[0], bins[1], values)):
        indicies[k_[0]][k_[1]].append(np.absolute(k_[2]))

    indicies = [[col for col in row] for row in indicies]
    return indicies


def group_bin_values_wrap(da_bins, da_vals):
    """
    Wrap the groupby_nd function to parse dask arrays to numpy like.

    Parameters
    ----------
    dd_data: array-like
        A dask array with two columns for integer bin values and one
        column for values that will be the argument for a function.

    """

    bins = [da_bins[:,0], da_bins[:,1]]
    res = group_bin_values(bins, da_vals)
    return res


# --- --- --- --- Group all columns in one function --- --- --- ---

def group_bin_flagval_wrap(da_bins, da_vals, da_flags, init_index=0):
    """
    Wrap the groupby_nd function to parse dask arrays to numpy like.

    Parameters
    ----------
    dd_data: array-like
        A dask array with two columns for integer bin values and one
        column for values that will be the argument for a function.

    """

    bins = [da_bins[:,0], da_bins[:,1]]
    res = groupby_flagval(bins, da_vals, da_flags, init_index=init_index)
    return res


def groupby_flagval(bins, values, flags, init_index:int=0) -> tuple:
    """
    Performs a groupby operation on two bin parameters and returns the list
    of indicies for mapping the unique groups to their positions.

    Parameters
    ----------
    bins: array-like
        An array or tuple containing two lists of bins values for two the
        parameters to be binned.
    init_index: int, optional
        Starting value for the indicies to use when iterating or chunking
        large lists of bins.

    Returns
    -------
    sub-arrays : list of ndarrays
        A list of arrays with indicies of bins.
    """

#     print("Starting groupby stage")
  
    sorted_keys = [np.unique(bin_col) for bin_col in bins]

#     print("End sorting")

    n_bins = [int(max(skeys) + 1) for skeys in sorted_keys]

#     print("Make empty array")
    
    index_grid = [[[]for j in range(n_bins[1])] for i in range(n_bins[0])]
    value_grid = [[[]for j in range(n_bins[1])] for i in range(n_bins[0])]
    flags_grid = [[[]for j in range(n_bins[1])] for i in range(n_bins[0])]

    for i, k_ in enumerate(zip(bins[0], bins[1], values, flags)):
#         if not(i%10**7):
#             print(f"Processing record {i}")
        index_grid[k_[0]][k_[1]].append(i+init_index)
        value_grid[k_[0]][k_[1]].append(np.absolute(k_[2]))
        flags_grid[k_[0]][k_[1]].append(np.absolute(k_[3]))

    return (index_grid, value_grid, flags_grid)


def combine_group_partitions_flagval():
    return True


def combine_group_flagval(val_list_chunks):
    '''
    Combine function for sets of indicies, values, and flags generated by 
    two-dimensional groupby-split function.

    Parameters:
    val_list_chunks: array-like
    A tuple of two-dimensional tuples, each containing the list
    of values corresponding to the unique pairs of two parameters (see
    groupby_nd function).

    Returns
    -------
    val_list_cat: list of ndarrays
        A list of arrays with values of bins.
    '''

#     print("Start combine stage")
    
    x_max = np.max([len(x_dim[0]) for x_dim in val_list_chunks])
    y_max = np.max([len(x_dim[0][0]) for x_dim in val_list_chunks])

#     print(f"Grid dimensions: ({x_max}, {y_max})")

    idx_list_cat = [[[]for j in range(y_max)] for i in range(x_max)]
    val_list_cat = [[[]for j in range(y_max)] for i in range(x_max)]
    flg_list_cat = [[[]for j in range(y_max)] for i in range(x_max)]

    for val_list in val_list_chunks:
        for i, vals in enumerate(val_list[0]):
            for j, vals_ in enumerate(vals):
                idx_list_cat[i][j] += list(vals_)
        for i, vals in enumerate(val_list[1]):
            for j, vals_ in enumerate(vals):
                val_list_cat[i][j] += list(vals_)
        for i, vals in enumerate(val_list[2]):
            for j, vals_ in enumerate(vals):
                flg_list_cat[i][j] += list(vals_)

    idx_list_cat = np.array([[col for col in row] for row in idx_list_cat])
    val_list_cat = np.array([[col for col in row] for row in val_list_cat])
    flg_list_cat = np.array([[col for col in row] for row in flg_list_cat])

    return (idx_list_cat, val_list_cat, flg_list_cat)


# --- --- --- --- Group indicies and values in one function --- --- --- ---


def combine_group_idx_val(val_list_chunks):
    '''
    Combine function for sets of indicies, values, and flags generated by 
    two-dimensional groupby-split function.

    Parameters:
    val_list_chunks: array-like
    A tuple of two-dimensional tuples, each containing the list
    of values corresponding to the unique pairs of two parameters (see
    groupby_nd function).

    Returns
    -------
    val_list_cat: list of ndarrays
        A list of arrays with values of bins.
    '''

    x_max = np.max([len(x_dim[0]) for x_dim in val_list_chunks])
    y_max = np.max([len(x_dim[0][0]) for x_dim in val_list_chunks])

    print(f"Grid dimensions: ({x_max}, {y_max})")

    idx_list_cat = [[[]for j in range(y_max)] for i in range(x_max)]
    val_list_cat = [[[]for j in range(y_max)] for i in range(x_max)]

    for val_list in val_list_chunks:
        for i, vals in enumerate(val_list[0]):
            for j, vals_ in enumerate(vals):
                idx_list_cat[i][j] += list(vals_)
        for i, vals in enumerate(val_list[1]):
            for j, vals_ in enumerate(vals):
                val_list_cat[i][j] += list(vals_)

    idx_list_cat = np.array([[col for col in row] for row in idx_list_cat])
    val_list_cat = np.array([[col for col in row] for row in val_list_cat])

    return (idx_list_cat, val_list_cat)


def groupby_idx_val(bins, values, init_index:int=0) -> tuple:
    """
    Performs a groupby operation on two bin parameters and returns the list
    of indicies for mapping the unique groups to their positions.

    Parameters
    ----------
    bins: array-like
        An array or tuple containing two lists of bins values for two the
        parameters to be binned.
    init_index: int, optional
        Starting value for the indicies to use when iterating or chunking
        large lists of bins.

    Returns
    -------
    sub-arrays : list of ndarrays
        A list of arrays with indicies of bins.
    """
        
    bin_group = []
    for bin_col in bins:
        bin_group.append(np.unique(bin_col, return_inverse=True))

    sorted_keys, bins_as_ints = [bg[0] for bg in bin_group], [bg[1] for bg in bin_group]    

    n_bins = [int(max(skeys) + 1) for skeys in sorted_keys]

    index_grid = [[[]for j in range(n_bins[1])] for i in range(n_bins[0])]
    value_grid = [[[]for j in range(n_bins[1])] for i in range(n_bins[0])]

    if init_index:
        for i, k_ in enumerate(zip(bins[0], bins[1])):
            index_grid[k_[0]][k_[1]].append(i+init_index)
    else:
        for i, k_ in enumerate(zip(bins[0], bins[1])):
            index_grid[k_[0]][k_[1]].append(i)

    for i, k_ in enumerate(zip(bins[0], bins[1], values)):
        value_grid[k_[0]][k_[1]].append(np.absolute(k_[2]))

    return (index_grid, value_grid)
    

def group_bin_idx_val_wrap(da_bins, da_vals):
    """
    Wrap the groupby_nd function to parse dask arrays to numpy like.

    Parameters
    ----------
    dd_data: array-like
        A dask array with two columns for integer bin values and one
        column for values that will be the argument for a function.

    """

    bins = [da_bins[:,0], da_bins[:,1]]
    res = groupby_idx_val(bins, da_vals)
    return res


# --- --- --- --- Group indicies and values in one function --- --- --- ---


def combine_ind_chunks(idx_list_chunks):
    '''
    Combine function for lists of indicies generated by two-dimensional
    groupby-split function.

    Parameters:
    ind_list: array-like
    A tuple of two-dimensional tuples, each containing the list
    of inds corresponding to the unique pairs of two parameters (see
    groupby_nd function).

    Returns
    -------
    sub-arrays: list of ndarrays
        A list of arrays with indicies of bins.
    '''

    x_max = np.max([len(x_dim) for x_dim in idx_list_chunks])
    y_max = np.max([len(x_dim[0]) for x_dim in idx_list_chunks])

    idx_list_cat = [[[]for j in range(y_max)] for i in range(x_max)]

    for ind_list in idx_list_chunks:
        for i, inds in enumerate(ind_list):
            for j, inds_ in enumerate(inds):
                idx_list_cat[i][j] += list(inds_)

    idx_list_cat = [np.array([np.array(col) for col in row]) for row in idx_list_cat]

    return idx_list_cat
    

def combine_group_values(val_list_chunks):
    '''
    Combine function for sets of values generated by two-dimensional
    groupby-split function.
    Parameters:
    val_list_chunks: array-like
    A tuple of two-dimensional tuples, each containing the list
    of values corresponding to the unique pairs of two parameters (see
    groupby_nd function).
    Returns
    -------
    val_list_cat: list of ndarrays
        A list of arrays with values of bins.
    '''

    x_max = np.max([len(x_dim) for x_dim in val_list_chunks])
    y_max = np.max([len(x_dim[0]) for x_dim in val_list_chunks])

    val_list_cat = [[[]for j in range(y_max)] for i in range(x_max)]

    for val_list in val_list_chunks:
        for i, vals in enumerate(val_list):
            for j, vals_ in enumerate(vals):
                val_list_cat[i][j] += list(vals_)

    val_list_cat = [np.array([np.array(col) for col in row]) for row in val_list_cat]

    return val_list_cat


def remove_flaged_rows(value_groups, index_groups, flag_groups):
    ''' Remove flagged rows from value and index grids.
    
    Parameters
    ----------
    value_groups : array-like
        Array with shape (K, N1, N2) where K is the number of blocks, (N1,
        N2) are the number of bins in each dimension.
    index_groups : array-like
        Same as value_groups but for indicies.
    flag_groups : array-like    
        Same as value_groups but for flags from MS file.
        
    Returns:
    value_groups : uv-grid array
    index_groups : uv-grid array
    
    Example: 
        value_groups, index_groups = remove_flagged_rows(value_groups, index_groups, flag_groups)
    '''
    
    x_len, y_len = len(value_groups), len(value_groups[1])

    vg = [[[]for j in range(y_len)] for i in range(x_len)]
    ig = [[[]for j in range(y_len)] for i in range(x_len)]
    
    for i in range(x_len):
        for j in range(y_len):
            vcell, icell, fcell = np.array(value_groups[i][j]), np.array(index_groups[i][j]), np.array(flag_groups[i][j])
            if len(vcell):
                vg[i][j] = vcell[fcell==False]
                ig[i][j] = icell[fcell==False]

    return vg, ig


def apply_to_groups(value_groups, function):
    '''
    Apply the provided function to reduce each bins values.

    Parameters
    ----------
    value_groups : array-like
        Array with shape (K, N1, N2) where K is the number of blocks, (N1,
        N2) are the number of bins in each dimension.

    Returns
    -------
    results : list of ndarrays
        A list of arrays with results for each bin.
    '''

    result = np.zeros([len(value_groups), len(value_groups[0])])

    for i, vals in enumerate(value_groups):
        for j, vals_ in enumerate(vals):
            if len(vals_):
#                 print(i, j, function(vals_))
                result[i][j] = function(vals_)

    result = [np.array(row) for row in result]
    return result
