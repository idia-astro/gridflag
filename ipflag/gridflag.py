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
import numba as nb

from . import groupby_apply, groupby_partition, annulus_stats

def compute_ipflag_grid(ds_ind, data_columns, uvbins, sigma=2.5, partition_level=4, stokes='I', client=None):
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
    uvbins : list
        List of two arrays, each containing the bounds for the discrete U and V bins  
        in the ds_ind dataset.
    sigma : float
        Depth of flagging - corresponds to the significance of a particular observation to 
        be labeled RFI.
    partition_level : int
        Internal parameter used to split the dataset in to congruent chunks in the uv-bin
        space. Used for concurrency.

    Returns
    -------
    flag_list: numpy.array
        A list of flag indicies to be applied to the original measurement set.
    median_grid : numpy.array
        A two-dimensional array representing a uv-grid. Each cell in the grid 
        is the median of all values, after zero removal and flagging, in that bin.     """

    # Load data from dask-ms dastaset
    ubins = ds_ind.U_bins.data
    vbins = ds_ind.V_bins.data

    # Determine which polarization state to grid and flag
    if stokes=='I':
        vals = (da.absolute(ds_ind.DATA[:,data_columns[0]].data + ds_ind.DATA[:,data_columns[1]].data))
    elif stokes=='Q':
        vals = (da.absolute(ds_ind.DATA[:,data_columns[0]].data - ds_ind.DATA[:,data_columns[1]].data))
    else:
        raise ValueError(f"compute_ipflag_grid: the stokes argument, '{stokes}', is not currently implemented, please select another value.")

    #Comute chunks
    chunks = list(ubins.chunks[0])

    # The array 'p' is used to map flags back to the original file order
    p = da.arange(len(ds_ind.newrow), dtype=np.int64, chunks=ubins.chunks)

    # Execute partition function which does a partial sort on U and V bins
    split_points, ubins_part, vbins_part, vals_part, p_part = dask_partition_sort(ubins, vbins, vals, p, chunks, partition_level, 0, client=client)

    # Convert back to delayed one final time
    dd_bins = da.stack([ubins_part, vbins_part, p_part]).T
    dd_bins = dd_bins.rechunk((ubins_part.chunks[0], 3))

    dd_bins = dd_bins.to_delayed()
    dd_vals = vals_part.to_delayed()
    
    group_chunks = [dask.delayed(groupby_partition.create_bin_groups_sort)(part[0][0], part[1]) for part in zip(dd_bins, dd_vals)] 
    function_chunks = [dask.delayed(groupby_partition.apply_grid_median)(c[1], c[2]) for c in group_chunks]
    median_grid = dask.delayed(groupby_partition.combine_function_partitions)(function_chunks)
    
    if client:
        median_grid_unflagged = client.compute(median_grid).result()
    else:
        median_grid_unflagged = median_grid.compute()    

    # Autoamatically compute annulus widths (naive)
    annulus_width = dask.delayed(annulus_stats.compute_annulus_bins)(median_grid, uvbins, 10)

    print(f"Initiate flagging with sigma = {sigma}.")

    annuli_data = dask.delayed(annulus_stats.process_annuli)(median_grid, annulus_width, uvbins, sigma=sigma)
    flag_results = [dask.delayed(annulus_stats.flag_one_annulus)(c[0], c[1], c[2], c[3], annuli_data[0], annuli_data[1]) for c in group_chunks]
    results = dask.delayed(groupby_partition.combine_annulus_results)([fr[0] for fr in flag_results], [fr[1] for fr in flag_results])

    print("Compute median grid on the partitions.")

    if client:
        flag_list, median_grid_flagged = client.compute(results).result()
    else:
        flag_list, median_grid_flagged = results.compute()

    return flag_list, median_grid_unflagged, median_grid_flagged


@nb.jit
def median_of_medians(data, sublist_length=11):
    # Compute median of medians on each chunk
    data = np.array([np.median(k) for k in [data[j:(j + sublist_length)] for j in range(0,len(data),sublist_length)]])
    return data

@nb.jit
def apply_median_func(data, depth):
    for i in range(depth):
        data = median_of_medians(data)
    return data


def da_median(a):
    m = np.median(a)
#     m = da.mean(a)
    return np.array(m)[None,None] # add dummy dimensions
    
def combine_median(meds):
    umed_list = np.concatenate(meds)
    print(f"umeds: {len(umed_list)}")
    return np.median(umed_list)

combine_median = dask.delayed(combine_median)
partition_permutation = dask.delayed(groupby_partition.partition_permutation_2d)
da_median = dask.delayed(da_median)
apply_median_func = dask.delayed(apply_median_func)

# Latest Version
def dask_partition_sort(a, b, v, p, chunks, binary_chunks, partition_level, client=None):

    split_points = np.array([])

    # Persist to compute DAG up to this point on workers. This improves the performance of 
    # in-place sorting six-fold.
    if client:    
        a = client.persist(a)
        b = client.persist(b)
        v = client.persist(v)
        p = client.persist(p)
    else:
        a = a.persist()
        b = b.persist()
        v = v.persist()
        p = p.persist()

    a_min, a_max = da.min(a), da.max(a)

    a = a.to_delayed()
    b = b.to_delayed()
    v = v.to_delayed()
    p = p.to_delayed()
    
    # Compute the median-of-medians heuristic (not the proper definition of MoM but we only need an approximate pivot)
    #    umed = [da_median(a_)[0] for a_ in a]
    resid = 50
    sublist_length=11
    min_nrow = np.min(chunks)

    med_depth = np.int32(np.floor((np.log(min_nrow/resid)/np.log(sublist_length))))

    umed = [apply_median_func(a_, med_depth) for a_ in a]

    umeds = [len(u.compute()) for u in umed]
    if client:
        pivot = combine_median(umed)
        pivot = client.compute(pivot).result()
    else:
        pivot = combine_median(umed).compute()

    if pivot == a_max:
        pivot-=0.5
    if pivot == a_min:
        pivot+=0.5

    results = [partition_permutation(a_, b_, v_, p_, pivot) for a_, b_, v_, p_ in zip(a, b, v, p)]

    print(f"Partition Level {partition_level}, med_depth: {med_depth}, pivot: {pivot}")

    # Bring split point to local process
    sp0 = [r[0] for r in results]
    if client:
        sp0 = client.compute(sp0)
        sp0 = [r.result() for r in sp0]
    else:
        sp0 = np.array([sp.compute() for sp in sp0])


    print(f"\t Split points (ave): {np.mean(sp0):.1f}")
    
    # Gather futures of sorted data from the partition function
    a = [r[1] for r in results]
    b = [r[2] for r in results]
    v = [r[3] for r in results]
    p = [r[4] for r in results]
    
    partition_level+=1
    
    if partition_level < binary_chunks:
                
        # Split each chunk of data in to two partitions as computed above then run original function on each
        a1 = da.concatenate([da.from_delayed(x_[0:s_], ((s_),), dtype=np.int32)    for x_, s_, c_ in zip(a, sp0, chunks)])
        b1 = da.concatenate([da.from_delayed(x_[0:s_], ((s_),), dtype=np.int32)    for x_, s_, c_ in zip(b, sp0, chunks)])
        v1 = da.concatenate([da.from_delayed(x_[0:s_], ((s_),), dtype=np.float32)  for x_, s_, c_ in zip(v, sp0, chunks)])
        p1 = da.concatenate([da.from_delayed(x_[0:s_], ((s_),), dtype=np.int64)    for x_, s_, c_ in zip(p, sp0, chunks)])

        a2 = da.concatenate([da.from_delayed(x_[s_:c_], ((c_-s_),), dtype=np.int32) for x_, s_, c_ in zip(a, sp0, chunks)])        
        b2 = da.concatenate([da.from_delayed(x_[s_:c_], ((c_-s_),), dtype=np.int32) for x_, s_, c_ in zip(b, sp0, chunks)])
        v2 = da.concatenate([da.from_delayed(x_[s_:c_], ((c_-s_),), dtype=np.float32) for x_, s_, c_ in zip(v, sp0, chunks)])
        p2 = da.concatenate([da.from_delayed(x_[s_:c_], ((c_-s_),), dtype=np.int64) for x_, s_, c_ in zip(p, sp0, chunks)])
                
        # Compute chunk size here as delayed objects don't have metadata
        chunks1 = list(a1.chunks[0])
        chunks2 = list(a2.chunks[0])
        
        # Do recursion step on each partition
        sp1, b1_, a1_, v1_, p1_ = dask_partition_sort(b1, a1, v1, p1, chunks1, binary_chunks, partition_level, client=client)
        sp2, b2_, a2_, v2_, p2_ = dask_partition_sort(b2, a2, v2, p2, chunks2, binary_chunks, partition_level, client=client)

        # Combine the partially sorted partitions into the original array shape
        a = da.concatenate([a1_, a2_])
        b = da.concatenate([b1_, b2_])
        v = da.concatenate([v1_, v2_])
        p = da.concatenate([p1_, p2_])

        split_points = np.concatenate((sp1, sp2))
        
    else:
        a1 = da.concatenate([da.from_delayed(x_[:sp_], ((sp_),), dtype=np.int32) for x_, sp_, ch_ in zip(a, sp0, chunks)])
        a1 = a1.rechunk(len(a1))
        a2 = da.concatenate([da.from_delayed(x_[sp_:ch_], ((ch_ - sp_),), dtype=np.int32) for x_, sp_, ch_ in zip(a, sp0, chunks)])
        a2 = a2.rechunk(len(a2))

        if (len(a1) == 0) or (len(a2) == 0):
            raise Exception(f"Partition level {binary_chunks} resulted in one or more zero-length partitions. Please choose a smaller value for 'partition_level' or increase the number of UV bins.")

        a = da.concatenate([a1, a2])
        
        b1 = da.concatenate([da.from_delayed(x_[:sp_], ((sp_),), dtype=np.int32) for x_, sp_, ch_ in zip(b, sp0, chunks)])
        b1 = b1.rechunk(len(b1))
        b2 = da.concatenate([da.from_delayed(x_[sp_:ch_], ((ch_ - sp_),), dtype=np.int32) for x_, sp_, ch_ in zip(b, sp0, chunks)])
        b2 = b2.rechunk(len(b2))

        b = da.concatenate([b1, b2])

        p1 = da.concatenate([da.from_delayed(x_[:sp_], ((sp_),), dtype=np.int32) for x_, sp_, ch_ in zip(p, sp0, chunks)])
        p1 = p1.rechunk(len(p1))
        p2 = da.concatenate([da.from_delayed(x_[sp_:ch_], ((ch_ - sp_),), dtype=np.int32) for x_, sp_, ch_ in zip(p, sp0, chunks)])
        p2 = p2.rechunk(len(p2))

        p = da.concatenate([p1, p2])

        v1 = da.concatenate([da.from_delayed(x_[:sp_], ((sp_),), dtype=np.int32) for x_, sp_, ch_ in zip(v, sp0, chunks)])
        v1 = v1.rechunk(len(v1))
        v2 = da.concatenate([da.from_delayed(x_[sp_:ch_], ((ch_ - sp_),), dtype=np.int32) for x_, sp_, ch_ in zip(v, sp0, chunks)])
        v2 = v2.rechunk(len(v2))

        v = da.concatenate([v1, v2])
        
        split_points = np.concatenate((a1.chunks, a2.chunks), axis=None)

    return split_points, a, b, v, p


def compute_median_grid(ds_ind, data_columns, uvbins, partition_level=4, stokes='I', client=None):

    print("Load in-memory data for sort.")
    ubins = ds_ind.U_bins.data
    vbins = ds_ind.V_bins.data

    if stokes=='I':
        vals = (da.absolute(ds_ind.DATA[:,data_columns[0]].data + ds_ind.DATA[:,data_columns[1]].data))
    elif stokes=='Q':
        vals = (da.absolute(ds_ind.DATA[:,data_columns[0]].data - ds_ind.DATA[:,data_columns[1]].data))
    else:
        raise ValueError(f"compute_median_grid: the stokes argument, '{stokes}', is not currently implemented, please select another value.")

    #Comute chunks
    chunks = list(ubins.chunks[0])

    # The array 'p' is used to map flags back to the original file order
    p = da.arange(len(ds_ind.newrow), dtype=np.int64, chunks=ubins.chunks)

    print("Compute parallel partitions and do partial sort.")
    split_points, ubins_, vbins_, vals_, p_ = dask_partition_sort(ubins, vbins, vals, p, chunks, partition_level, 0, client=client)

    print("Preparing dask delayed...")
    dd_bins = da.stack([ubins_, vbins_, p_]).T
    dd_bins = dd_bins.rechunk((ubins_.chunks[0], 3))

    dd_bins = dd_bins.to_delayed()
    dd_vals = vals_.to_delayed()

    del ubins, vbins, vals, p

    print("Compute UV map and median grid.")
    group_chunks = [dask.delayed(groupby_partition.create_bin_groups_sort)(part[0][0], part[1]) for part in zip(dd_bins, dd_vals)] 
    function_chunks = [dask.delayed(groupby_partition.apply_grid_median)(c[1], c[2]) for c in group_chunks]
    median_grid = dask.delayed(groupby_partition.combine_function_partitions)(function_chunks)

    if client:
        median_grid = client.compute(median_grid).result()
    else:
        median_grid = median_grid.compute()

    return median_grid
        

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
