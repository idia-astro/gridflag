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

from . import groupby_apply, groupby_partition, annulus_stats


def map_grid_partition(ds_ind, data_columns, uvbins, partition_level=4, stokes='I', chunk_sizes=[]):
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
#     ds_ind.coords['index'] = (('newrow'), da.arange(len(ds_ind.newrow)))
#     ds_ind = ds_ind.set_index({'newrow': 'index'})

    print("Load in-memory data for sort.")
    ubins = ds_ind.U_bins.data.compute()
    vbins = ds_ind.V_bins.data.compute()    
    vals = np.array(da.absolute(ds_ind.DATA[:,data_columns[0]].data + ds_ind.DATA[:,data_columns[1]].data))
    p = np.arange(len(ubins), dtype=np.int64)

    print("Compute parallel partitions and do partial sort.")
    p, sp = groupby_partition.binary_partition_2d(ubins, vbins, partition_level, 0, p, vals)
    
    split_chunks = [0] + sp + [len(ds_ind.newrow)]
    split_chunks = tuple(np.diff(split_chunks))
    
    print("Preparing dask delayed...")
    dd_bins = da.stack([da.from_array(ubins), da.from_array(vbins), da.array(p)]).T
    dd_vals = da.from_array(vals)
    
    dd_bins = dd_bins.rechunk((split_chunks, (3)))
    dd_vals = dd_vals.rechunk((split_chunks, (1)))
    
    dd_bins = dd_bins.to_delayed()
    dd_vals = dd_vals.to_delayed()
    
    del ubins, vbins, p, vals
    
    group_chunks = [dask.delayed(groupby_partition.create_bin_groups_sort)(part[0][0], part[1]) for part in zip(dd_bins, dd_vals)] 
    function_chunks = [dask.delayed(groupby_partition.apply_grid_function)(c[1], c[2]) for c in group_chunks]
    median_grid = dask.delayed(groupby_partition.combine_function_partitions)(function_chunks)
    
    annulus_width = dask.delayed(annulus_stats.compute_annulus_bins)(median_grid, uvbins, 10)
    annuli_data = dask.delayed(annulus_stats.process_annuli)(median_grid, annulus_width, uvbins, sigma=3.)
    
    flag_results = [dask.delayed(annulus_stats.flag_one_annulus)(c[0], c[1], c[2], annuli_data[0], annuli_data[1]) for c in group_chunks]

    results = dask.delayed(groupby_partition.combine_annulus_results)([fr[0] for fr in flag_results], [fr[1] for fr in flag_results])

    print("Compute median grid on the partitions.")    

#     return results
    flag_list, median_grid = results.compute()
    
    #Note: It may not be necessary to return ds_ind in this function
    return flag_list, median_grid


def map_grid_partition_old(ds_ind, data_columns, uvbins, stokes='I', chunk_sizes=[]):
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
    
    row_stack = ds_ind.newrow.drop(['U_bins', 'V_bins'])
    ds_ind = ds_ind.set_index({'newrow': 'index'})    

    ds_ind = ds_ind.chunk({'newrow':len(ds_ind.newrow)})

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
    
    # Chunk the second dimension together
    dd_bins = dd_bins.rechunk((split_chunks, (3)))

    # Not necessary because rechunk on dataset is done above
    #dd_vals = dd_vals.rechunk((split_chunks, (1)))
    
    dd_bins = dd_bins.to_delayed()
    dd_vals = dd_vals.to_delayed()
    dd_flgs = dd_flgs.to_delayed()
    
    group_chunks = [dask.delayed(groupby_partition.create_bin_groups_sort)(part[0][0], part[1]) for part in zip(dd_bins, dd_vals)] 
    function_chunks = [dask.delayed(groupby_partition.apply_grid_function)(c[1], c[2]) for c in group_chunks]
    median_grid = dask.delayed(groupby_partition.combine_function_partitions)(function_chunks)
    
    annulus_width = dask.delayed(annulus_stats.compute_annulus_bins)(median_grid, uvbins, 10)
    annuli_data = dask.delayed(annulus_stats.process_annuli)(median_grid, annulus_width, uvbins, sigma=3.)
    
    flag_results = [dask.delayed(annulus_stats.flag_one_annulus)(c[0], c[1], c[2], annuli_data[0], annuli_data[1]) for c in group_chunks]

    results = dask.delayed(groupby_partition.combine_annulus_results)([fr[0] for fr in flag_results], [fr[1] for fr in flag_results])

    print("Compute median grid on the partitions.")    

#     return results
    flag_list, median_grid = results.compute()
    
    #Note: It may not be necessary to return ds_ind in this function
    return ds_ind, flag_list, median_grid, row_stack
        

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
