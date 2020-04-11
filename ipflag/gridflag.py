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

import groupby_apply

def map_grid_function(ds_ind, data_columns, chunk_size:int=10**6, return_index:bool=False):
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
    dd_ubins = da.from_array(ds_ind.U_bins)
    dd_vbins = da.from_array(ds_ind.V_bins)
    dd_vals = da.from_array(np.absolute(ds_ind.DATA[:,data_columns[0]]+ds_ind.DATA[:,data_columns[1]]))

    # Combine U and V bins into one dask array
    dd_bins = da.stack([dd_ubins, dd_vbins]).T
    
    # Apply unifrom chunks to both dask arrays
    dd_bins = dd_bins.rechunk([chunk_size, 2])
    dd_vals = dd_vals.rechunk([chunk_size, 1])
    
    # Convert to delayed data structures
    bin_partitions = dd_bins.to_delayed()
    val_partitions = dd_vals.to_delayed()

    # Compute indicies for each bin in the grid for each chunk
    value_group_chunks = [dask.delayed(groupby_apply.group_bin_values_wrap)(part[0][0], part[1]) for part in zip(bin_partitions, val_partitions)]
    value_groups_ = dask.delayed(groupby_apply.combine_group_values)(value_group_chunks)

    # Compute index groups for each bin
    index_group_chunks = [dask.delayed(groupby_apply.groupby_nd_wrap)(part[0], init_index=int(chunk_size*i)) for i, part in enumerate(bin_partitions)]
    index_groups_ = dask.delayed(groupby_apply.combine_ind_chunks)(index_group_chunks)

    if return_index:
        # Compute the grid from above without doing the apply step
        value_groups = value_groups_.compute()
        index_groups = index_groups_.compute()
        return value_groups, index_groups
    else:
        # Apply the function to the grid without explicitly computing the indicies
        median_grid_ = dask.delayed(groupby_apply.apply_to_groups)(value_groups_, np.median)
        std_grid_ = dask.delayed(groupby_apply.apply_to_groups)(value_groups_, np.std) 
        median_grid = median_grid_.compute()
        std_grid = std_grid_.compute()
        return median_grid, std_grid 
