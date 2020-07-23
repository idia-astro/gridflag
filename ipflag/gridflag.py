#!/usr/bin/python
"""
GridFlag

Use XArray, Dask, and Numpy to load CASA Measurement Set (MS) data and
create binned UV data.

Todo: 
[ ] Add options for choosing stokes parameters or amplitude/complex components
"""


import numpy as np

import dask
import dask.array as da

from . import groupby_apply, groupby_partition


def map_grid_partition(ds_ind, data_columns, stokes='I', chunk_sizes=[]):
    """
    Partition the dataset in to orthogonal chunks of uv-bins on which to perform parallel
    operations.
    
    Parameters
    ----------
    ds_ind : xarray.dataset
        An xarray datset imported from a measurement set. It must contain 
        coordinates U_bins and V_bins, and relevant data and position 
        variables.
    data_columns : list
        Components for Stokes terms to be used to compute amplitude. Depends 
        on dataset.
    chunk_size : int
        The chunk size for computing split-apply functions in dask, Default is
        empty. If empty, chunks will be computed automatically via partitioning.
        
    Returns
    -------
    value_rows : numpy.array
        Values of each visibility sorted by uv-bins. For UV bins see grid_row_map.
    index_rows : numpy.array
        Indicies of each visibility corresponding to the position in the original 
        Measurement Set, sorted by uv-bins. For UV bins see grid_row_map.
    function_groups : array-like
        A two-dimensional array with the value of a function applied to the values in each 
        uv-cell. 
    grid_row_map : array-like
        A list of each partition, each partition contains a list, whose rows represent
        a map for a UV bin to it's start row in value_rows, and index_rows.
    """

    # Set new contiguous index after unfolding
    ds_ind.coords['index'] = (('newrow'), da.arange(len(ds_ind.newrow)))
    ds_ind = ds_ind.set_index({'newrow': 'index'})    

    # Load ubins in memory to compute partition (to investigate impact on memory for large datasets)
    ubins = ds_ind.U_bins.data.compute()

    print("Compute parallel partitions and do partial sort.")
#     p = np.zeros_like(ubins)
    p = np.arange(len(ubins), dtype=np.int64)
    p, sp = groupby_partition.binary_partition(ubins, 4, 0, p)

    # Sort the dataset using the partition permutation
    ds_ind = ds_ind.isel(newrow=p)
    ds_ind = ds_ind.unify_chunks()

    # Convert from chunk start-index list to a list of chunk sizes
    split_chunks = [0] + sp + [len(ds_ind.newrow)]
    split_chunks = tuple(np.diff(split_chunks))
    
    ds_ind = ds_ind.chunk({'newrow':split_chunks})

    print("Preparing dask delayed...")
    dd_bins = da.stack([ds_ind.U_bins.data, ds_ind.V_bins.data, da.array(ds_ind.newrow)]).T
    dd_vals = (da.absolute(ds_ind.DATA[:,data_columns[0]].data + ds_ind.DATA[:,data_columns[1]].data))
    dd_flgs = (ds_ind.FLAG[:,data_columns[0]].data | ds_ind.FLAG[:,data_columns[1]].data)
    
    dd_bins = dd_bins.rechunk((split_chunks, (3)))
    dd_vals = dd_vals.rechunk((split_chunks, (1)))
    
    dd_bins = dd_bins.to_delayed()
    dd_vals = dd_vals.to_delayed()
    dd_flgs = dd_flgs.to_delayed()
    
    group_chunks = [dask.delayed(groupby_partition.create_bin_groups_sort)(part[0][0], part[1]) for part in zip(dd_bins, dd_vals)] 
    function_chunks = [dask.delayed(groupby_partition.apply_grid_function)(c[1], c[2]) for c in group_chunks]
    median_chunks = dask.delayed(groupby_partition.combine_function_partitions)(function_chunks)

    print("Compute median grid on the partitions.")    
    median_chunks = median_chunks.compute()
    return ds_ind, median_chunks
        

def map_amplitude_grid(ds_ind, data_columns, stokes='I', chunk_size:int=10**6, return_index:bool=False):
    """    
    Map functions to a concurrent dask functions to an Xarray dataset with 
    pre-computed grid indicies.

    Parameters
    ----------
    ds_ind : xarray.dataset
        An xarray datset imported from a measurement set. It must contain 
        coordinates U_bins and V_bins, and relevant data and position 
        variables.
    data_columns : list
        Components for Stokes terms to be used to compute amplitude. Depends 
        on dataset.
    chunk_size : int
        The chunk size for computing split-apply functions in dask, Default is
        '10**6'.
    return_index : bool
        Determines the return data type. If true, it returns a 2d grid of lists 
        of values and indicies for each bin.

    Returns
    -------
    value_groups: numpy.array
        A two-dimensional array representing a uv-grid. Each cell in the grid 
        contains a list of values from the dataset.
    uvbins: xarray.Dataset
        A two-dimensional array representing a uv-grid. Each cell in the grid 
        contains a list of indicies to be used to map subsequent computations 
        back to the original dataset.
    """


    # Get dask arrays of UV-bins and visibilities from XArray dataset
    dd_ubins = ds_ind.U_bins.data
    dd_vbins = ds_ind.V_bins.data
    dd_flgs = (ds_ind.FLAG[:,data_columns[0]].data | ds_ind.FLAG[:,data_columns[1]].data)

    if stokes=='I':
        dd_vals = (np.absolute(ds_ind.DATA[:,data_columns[0]].data + ds_ind.DATA[:,data_columns[1]].data))
    elif stokes=='Q':
        dd_vals = (np.absolute(ds_ind.DATA[:,data_columns[0]].data - ds_ind.DATA[:,data_columns[1]].data))

    # Combine U and V bins into one dask array
    dd_bins = da.stack([dd_ubins, dd_vbins]).T
    
    # Apply unifrom chunks to both dask arrays
    dd_bins = dd_bins.rechunk([chunk_size, 2])
    dd_vals = dd_vals.rechunk([chunk_size, 1])
    dd_flgs = dd_flgs.rechunk([chunk_size, 1])
    
    # Convert to delayed data structures
    bin_partitions = dd_bins.to_delayed()
    val_partitions = dd_vals.to_delayed()
    flg_partitions = dd_flgs.to_delayed()

    # Compute indicies for each bin in the grid for each chunk
    group_chunks = [dask.delayed(groupby_apply.group_bin_flagval_wrap)(part[0][0], part[1], part[2], init_index=(chunk_size*kth)) for kth, part in enumerate(zip(bin_partitions, val_partitions, flg_partitions))]    
    groups = dask.delayed(groupby_apply.combine_group_flagval)(group_chunks)

#     group_chunks = [dask.delayed(groupby_apply.group_bin_idx_val_wrap)(part[0][0], part[1]) for part in zip(bin_partitions, val_partitions)]    
#     groups = dask.delayed(groupby_apply.combine_group_idx_val)(group_chunks)
    
    if return_index:
        # Compute the grid from above without doing the apply step
        groups = groups.compute()
        index_groups, value_groups, flag_groups = groups[0], groups[1], groups[2]
        return index_groups, value_groups, flag_groups

    else:
        # Apply the function to the grid without explicitly computing the indicies
        median_grid = dask.delayed(groupby_apply.apply_to_groups)(value_groups_, np.median)
        std_grid = dask.delayed(groupby_apply.apply_to_groups)(value_groups_, np.std) 
#         median_grid = median_grid_.compute()
#         std_grid = std_grid_.compute()
        return median_grid, std_grid 
