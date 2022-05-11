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


def process_stokes_options(ds_ind, stokes='I', use_existing_flags=True):
    """
    Determine columns to use for flagging algorithm from DATA and FLAG tables.

    Inputs:
    ds_ind              The input dask dataframe
    stokes              Which Stokes to flag on, str
    use_existing_flags  Take into account existing flags in the MS, bool

    Returns:
    vals                Visibilities, dask dataframe
    flags               Flags, dask dataframe
    """

    # Determine which polarization state to grid and flag
    if stokes=='I':
        vals = ds_ind.DATA[:,0].data + ds_ind.DATA[:,-1].data
    elif stokes=='Q':
        vals = ds_ind.DATA[:,0].data - ds_ind.DATA[:,-1].data
    elif stokes=='U':
        vals = ds_ind.DATA[:,1].data + ds_ind.DATA[:,2].data
    elif stokes=='V':
        vals = ds_ind.DATA[:,1].data - ds_ind.DATA[:,2].data
    else:
        raise ValueError(f"compute_ipflag_grid: the stokes argument, '{stokes}', \
            is not currently implemented, please select another value.")

    # Take only the real part for Gaussian stats
    vals = vals.real

    if use_existing_flags:
        if stokes=='I':
            flags = ds_ind.FLAG.data[:,0] | ds_ind.FLAG.data[:,-1]
        elif stokes=='Q':
            flags = ds_ind.FLAG.data[:,0] | ds_ind.FLAG.data[:,-1]
        elif stokes=='U':
            flags = ds_ind.FLAG.data[:,1] | ds_ind.FLAG.data[:,2]
        elif stokes=='V':
            flags = ds_ind.FLAG.data[:,1] | ds_ind.FLAG.data[:,2]
    else:
        flags = None

    return vals, flags



def compute_ipflag_grid(ds_ind, uvbins, sigma=3.0, partition_level=4, stokes='I', use_existing_flags=True, client=None):
    """    
    Map functions to a concurrent dask functions to an Xarray dataset with pre-
    computed grid indicies.

    Parameters
    ----------
    ds_ind : xarray.dataset
        An xarray datset imported from a measurement set. It must contain 
        coordinates U_bins and V_bins, and relevant data and position variables.
    uvbins : list
        List of two arrays, each containing the bounds for the discrete U and V 
        bins in the ds_ind dataset.
    sigma : float
        Depth of flagging - corresponds to the significance of a particular 
        observation to be labeled RFI.
    partition_level : int
        Internal parameter used to split the dataset in to congruent chunks in 
        the uv-bin space. Used for concurrency.
    stokes : string
        Reduce the data via one or more combinations of columns - 'I', 'Q', 'U',
        'V' for corresponding stokes parameters. For amplitude of each columns 
        use 'A'. 
    use_existing_flags : bool
        Either remove rows that are flagged in the input MS file before flagging
        (True) or ignore and overwrite existing flags (False).
    client : dask.distributed.client
        Pass dask distributed client or 'None' for local computation.

    Returns
    -------
    flag_list: numpy.array
        A list of flag indicies to be applied to the original measurement set.
    median_grid_unflagged : numpy.array
        A two-dimensional array representing a uv-grid. Each cell in the grid is 
        the median of all values, after zero removal and flagging, in that bin.
    median_grid : numpy.array
        The same as the above but after flagged visibilities are removed.
    """

    # Load data from dask-ms dastaset
    ubins = ds_ind.U_bins.data
    vbins = ds_ind.V_bins.data

    flags, vals = process_stokes_options(ds_ind, stokes, True)

    # The array 'p' is used to map flags back to the original file order
    p = da.arange(len(ds_ind.newrow), dtype=np.int64, chunks=ubins.chunks)

    chunks = list(ubins.chunks[0])

    print(f"Original chunk sizes: ", chunks)

    # Execute partition function which does a partial sort on U and V bins
    split_points, ubins_part, vbins_part, vals_part, flags_part, p_part = dask_partition_sort(
        ubins, vbins, vals, flags, p, chunks, partition_level, 0, client=client)

    print(f"Dataset split into {len(ubins_part.chunks[0])} uv-plane partitions.") 
    print(f"Flagging: {ubins_part.chunks[0]} rows.")

    # Convert back to delayed one final time
    dd_bins = da.stack([ubins_part, vbins_part, p_part]).T
    dd_bins = dd_bins.rechunk((ubins_part.chunks[0], 3))

    dd_bins = dd_bins.to_delayed()
    dd_vals = vals_part.to_delayed()

    dd_flags = flags_part.to_delayed()

    vdim = vbins_part.ndim

    if use_existing_flags:
        if vdim > 1:
            group_bins_sort = [dask.delayed(groupby_partition.sort_bins_multi)
            (
                part[0][0], 
                part[1][0], 
                part[2]
            ) for part in zip(dd_bins, dd_vals, dd_flags)]
        else:
            group_bins_sort = [dask.delayed(groupby_partition.sort_bins)
            (
                part[0][0], 
                part[1], 
                part[2]
            ) for part in zip(dd_bins, dd_vals, dd_flags)]

    else:
        if vdim > 1:
            group_bins_sort = [dask.delayed(groupby_partition.sort_bins_multi)
            (
                part[0][0], 
                part[1][0]
            ) for part in zip(dd_bins, dd_vals)]
        else:
            group_bins_sort = [dask.delayed(groupby_partition.sort_bins)
            (
                part[0][0], 
                part[1]
            ) for part in zip(dd_bins, dd_vals)]


    group_chunks = [dask.delayed(groupby_partition.create_bin_groups_sort)
                            (c[0], c[1], c[2]) for c in group_bins_sort]
    function_chunks = [dask.delayed(groupby_partition.apply_grid_median)
                            (c[1], c[2]) for c in group_chunks]
    median_grid = dask.delayed(groupby_partition.combine_function_partitions)(
                                function_chunks)

    if client:
        median_grid_unflagged = client.compute(median_grid)
    else:
        median_grid_unflagged = median_grid.compute()

    # Autoamatically compute annulus widths (naive)
    annulus_width = dask.delayed(annulus_stats.compute_annulus_bins)(
        median_grid_unflagged, 
        uvbins, 
        10
    )

    print(f"Initiate flagging with sigma = {sigma}.")

    annuli_data = dask.delayed(annulus_stats.process_annuli)(
        median_grid_unflagged,
        annulus_width,
        uvbins[0],
        uvbins[1],
        sigma=sigma
    )

    # Process only one column for multi-dim data until we finish this function
    if vdim > 1:
        flag_results = [dask.delayed(annulus_stats.flag_one_annulus)(
            c[0], 
            c[1][:,0], 
            c[2], 
            c[3], 
            annuli_data[0], 
            annuli_data[1], 
            sigma=sigma) for c in group_chunks]
    else:
        flag_results = [dask.delayed(annulus_stats.flag_one_annulus)(
            c[0],
            c[1],
            c[2],
            c[3],
            annuli_data[0],
            annuli_data[1],
            sigma=sigma) for c in group_chunks]

    results = dask.delayed(groupby_partition.combine_annulus_results)(
        [fr[0] for fr in flag_results], 
        [fr[1] for fr in flag_results], 
        [fr[2] for fr in flag_results], 
        [fr[3] for fr in flag_results])

    print("Compute median grid on the partitions.")

    if client:
        ms_flag_list, da_flag_list, median_grid_flagged, count_grid = \
            client.compute(results).result()
        median_grid_unflagged = median_grid_unflagged.result()
    else:
        ms_flag_list, da_flag_list, median_grid_flagged, count_grid = \
            results.compute()


    return ms_flag_list, median_grid_unflagged, median_grid_flagged


@nb.njit(nogil=True)
def strip_zero_values(a, b, v, p):
    
    null_flags = p[np.where(v==0)]
    a = a[np.where(v!=0)]
    newlen, oldlen = len(a), len(b)
    print("Removed ", oldlen - newlen, " zero value rows of ", oldlen, 
        " rows (", round(100*(oldlen-newlen)/oldlen, 2), "%)." )
    b = b[np.where(v!=0)]
    v = v[np.where(v!=0)]
    p = p[np.where(v!=0)]
        
    return newlen, a, b, v, p, null_flags

@nb.njit(nogil=True)
def median_of_medians(data, sublist_length=11):
    # Compute median of medians on each chunk
    data = np.array([np.median(k) for k in [data[j:(j + sublist_length)] for j in range(0,len(data),sublist_length)]])
    return data

def apply_median_func(data, depth):
    for i in np.arange(depth):
        data = median_of_medians(data, 11)
    return data


def da_median(a):
    m = np.median(a)
#     m = da.mean(a)
    return np.array(m)[None,None] # add dummy dimensions
    
def combine_median(meds):
    umed_list = np.concatenate(meds)
#     print(f"umeds:", len(umed_list))
    return np.median(umed_list)

combine_median = dask.delayed(combine_median)

partition_permutation = dask.delayed(groupby_partition.partition_permutation)
partition_permutation_multi = dask.delayed(groupby_partition.partition_permutation_multi)


da_median = dask.delayed(da_median)
apply_median_func = dask.delayed(apply_median_func)

# Latest Version
def dask_partition_sort(a, b, v, f, p, chunks, binary_chunks, partition_level, client=None):

    split_points = np.array([])

    # Persist to compute DAG up to this point on workers. This improves the performance of 
    # in-place sorting six-fold.
    if client:    
        a = client.persist(a)
        b = client.persist(b)
        v = client.persist(v)
        f = client.persist(f)
        p = client.persist(p)
    else:
        a = a.persist()
        b = b.persist()
        v = v.persist()
        f = f.persist()
        p = p.persist()

    a_min, a_max = da.min(a), da.max(a)

    v = da.squeeze(v)

    if v.ndim > 1:
        ncols = v.shape[1]
    else:
        ncols = 1

    # print("Columns: ", ncols, v.ndim, v.shape, v.dtype, nb.typeof(v[:10].compute()))

    a = a.to_delayed()
    b = b.to_delayed()
    v = v.to_delayed()
    f = f.to_delayed()
    p = p.to_delayed()
    
    # Compute the median-of-medians heuristic (not the proper definition of MoM but we only need an approximate pivot)
    #    umed = [da_median(a_)[0] for a_ in a]
    resid = 50
    sublist_length=11
    min_nrow = np.min(chunks)

    med_depth = np.int32(np.floor((np.log(min_nrow/resid)/np.log(sublist_length))))

    umed = [apply_median_func(a_, med_depth) for a_ in a]

#     umeds = [len(u.compute()) for u in umed]
    if client:
        pivot = combine_median(umed)
        pivot = client.compute(pivot).result()
    else:
        pivot = combine_median(umed).compute()

    if pivot == a_max:
        pivot-=0.5
    if pivot == a_min:
        pivot+=0.5


    if ncols > 1:
        results = [partition_permutation_multi(a_, b_, v_[0], f_, p_, pivot) for a_, b_, v_, f_, p_ in zip(a, b, v, f, p)]
    else:
        results = [partition_permutation(a_, b_, v_, f_, p_, pivot) for a_, b_, v_, f_, p_ in zip(a, b, v, f, p)]

    print(f"Partition Level {partition_level}, med_depth: {med_depth}, pivot: {pivot}")   

    if client:
        results = client.persist(results)
        sp0 = [r[0].compute() for r in results]
        a = [r[1] for r in results]
        b = [r[2] for r in results]
        v = [r[3] for r in results]
        f = [r[4] for r in results]
        p = [r[5] for r in results]

    else:
        results = dask.persist(results)
        sp0 = [r[0].compute() for r in results[0]]
        a = [r[1] for r in results[0]]
        b = [r[2] for r in results[0]]
        v = [r[3] for r in results[0]]
        f = [r[4] for r in results[0]]
        p = [r[5] for r in results[0]]
    
    partition_level+=1
    
    if partition_level < binary_chunks:
                
        # Split each chunk of data in to two partitions as computed above then recurse on each partition.
        a1 = da.concatenate([da.from_delayed(x_[0:s_], ((s_),), dtype=np.int32)    for x_, s_, c_ in zip(a, sp0, chunks)])
        b1 = da.concatenate([da.from_delayed(x_[0:s_], ((s_),), dtype=np.int32)    for x_, s_, c_ in zip(b, sp0, chunks)])
        f1 = da.concatenate([da.from_delayed(x_[0:s_], ((s_),), dtype=np.int64)    for x_, s_, c_ in zip(f, sp0, chunks)])
        p1 = da.concatenate([da.from_delayed(x_[0:s_], ((s_),), dtype=np.int64)    for x_, s_, c_ in zip(p, sp0, chunks)])

        a2 = da.concatenate([da.from_delayed(x_[s_:c_], ((c_-s_),), dtype=np.int32) for x_, s_, c_ in zip(a, sp0, chunks)])        
        b2 = da.concatenate([da.from_delayed(x_[s_:c_], ((c_-s_),), dtype=np.int32) for x_, s_, c_ in zip(b, sp0, chunks)])
        f2 = da.concatenate([da.from_delayed(x_[s_:c_], ((c_-s_),), dtype=np.int64) for x_, s_, c_ in zip(f, sp0, chunks)])
        p2 = da.concatenate([da.from_delayed(x_[s_:c_], ((c_-s_),), dtype=np.int64) for x_, s_, c_ in zip(p, sp0, chunks)])

        # Requires a different format for array shape for 1d and 2d arrays
        if ncols > 1:
            v1 = da.concatenate([da.from_delayed(x_[0:s_], ((s_,ncols)), dtype=np.float32)  for x_, s_, c_ in zip(v, sp0, chunks)])
            v2 = da.concatenate([da.from_delayed(x_[s_:c_], ((c_-s_,ncols)), dtype=np.float32) for x_, s_, c_ in zip(v, sp0, chunks)])
        else:
            v1 = da.concatenate([da.from_delayed(x_[0:s_], ((s_),), dtype=np.float32)  for x_, s_, c_ in zip(v, sp0, chunks)])
            v2 = da.concatenate([da.from_delayed(x_[s_:c_], ((c_-s_),), dtype=np.float32) for x_, s_, c_ in zip(v, sp0, chunks)])            
                
        # Compute chunk size here as delayed objects don't have metadata
        chunks1 = list(a1.chunks[0])
        chunks2 = list(a2.chunks[0])
        
        # Do recursion step on each partition
        sp1, b1_, a1_, v1_, f1_, p1_ = dask_partition_sort(b1, a1, v1, f1, p1, chunks1, binary_chunks, partition_level, client=client)
        sp2, b2_, a2_, v2_, f2_, p2_ = dask_partition_sort(b2, a2, v2, f2, p2, chunks2, binary_chunks, partition_level, client=client)

        # Combine the partially sorted partitions into the original array shape
        a = da.concatenate([a1_, a2_])
        b = da.concatenate([b1_, b2_])
        v = da.concatenate([v1_, v2_])
        f = da.concatenate([f1_, f2_])
        p = da.concatenate([p1_, p2_])

        split_points = np.concatenate((sp1, sp2))
        
    else:
        # Break out of the recursion and combine the partitions into a partially sorted array.
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

        f1 = da.concatenate([da.from_delayed(x_[:sp_], ((sp_),), dtype=np.int64) for x_, sp_, ch_ in zip(f, sp0, chunks)])
        f1 = f1.rechunk(len(f1))
        f2 = da.concatenate([da.from_delayed(x_[sp_:ch_], ((ch_ - sp_),), dtype=np.int64) for x_, sp_, ch_ in zip(f, sp0, chunks)])
        f2 = f2.rechunk(len(f2))

        f = da.concatenate([f1, f2])

        p1 = da.concatenate([da.from_delayed(x_[:sp_], ((sp_),), dtype=np.int64) for x_, sp_, ch_ in zip(p, sp0, chunks)])
        p1 = p1.rechunk(len(p1))
        p2 = da.concatenate([da.from_delayed(x_[sp_:ch_], ((ch_ - sp_),), dtype=np.int64) for x_, sp_, ch_ in zip(p, sp0, chunks)])
        p2 = p2.rechunk(len(p2))

        p = da.concatenate([p1, p2])

        if ncols > 1:
            v1 = da.concatenate([da.from_delayed(x_[:sp_], ((sp_),ncols), dtype=np.float32) for x_, sp_, ch_ in zip(v, sp0, chunks)])
            v1 = v1.rechunk(len(v1))
            v2 = da.concatenate([da.from_delayed(x_[sp_:ch_], ((ch_ - sp_),ncols), dtype=np.float32) for x_, sp_, ch_ in zip(v, sp0, chunks)])
            v2 = v2.rechunk(len(v2))
        else:
            v1 = da.concatenate([da.from_delayed(x_[:sp_], ((sp_),), dtype=np.float32) for x_, sp_, ch_ in zip(v, sp0, chunks)])
            v1 = v1.rechunk(len(v1))
            v2 = da.concatenate([da.from_delayed(x_[sp_:ch_], ((ch_ - sp_),), dtype=np.float32) for x_, sp_, ch_ in zip(v, sp0, chunks)])
            v2 = v2.rechunk(len(v2))            

        v = da.concatenate([v1, v2])
        
        # The split points are indicies for the array position of the partitions, use later to define dask chunk with unique (u,v) bin ranges 
        # for concurrent statistical calculations.
        split_points = np.concatenate((a1.chunks, a2.chunks), axis=None)

    return split_points, a, b, v, f, p


def compute_median_grid(
    ds_ind, 
    uvbins, 
    partition_level=4, 
    stokes='I', 
    client=None
):

    print("Load in-memory data for sort.")
    ubins = ds_ind.U_bins.data
    vbins = ds_ind.V_bins.data

    vals, flags = process_stokes_options(ds_ind, stokes, True)

    #Comute chunks
    chunks = list(ubins.chunks[0])

    # The array 'p' is used to map flags back to the original file order
    p = da.arange(len(ds_ind.newrow), dtype=np.int64, chunks=ubins.chunks)

    print("Compute parallel partitions and do partial sort.")
    split_points, 
    ubins_part, 
    vbins_part, 
    vals_part, 
    flags_part, 
    p_part = dask_partition_sort(ubins, 
                                 vbins, 
                                 vals, 
                                 flags, 
                                 p, 
                                 chunks, 
                                 partition_level, 
                                 0, 
                                 client=client
                                 )

    print("Preparing dask delayed...")
    dd_bins = da.stack([ubins_part, vbins_part, p_part]).T
    dd_bins = dd_bins.rechunk((ubins_part.chunks[0], 3))

    dd_bins = dd_bins.to_delayed()
    dd_vals = vals_part.to_delayed()

    dd_flags = flags_part.to_delayed()

    vdim = vbins_part.ndim

    del ubins, vbins, vals, p

    print("Compute UV map and median grid.")

    # group_bins_sort = [dask.delayed(groupby_partition.sort_bins)(part[0][0], part[1]) for part in zip(dd_bins, dd_vals)] 
    if vdim > 1:
        group_bins_sort = [dask.delayed(groupby_partition.sort_bins_multi)(
                            part[0][0], 
                            part[1][0], 
                            part[2]) for part in zip(dd_bins, dd_vals, dd_flags)]
    else:
        group_bins_sort = [dask.delayed(groupby_partition.sort_bins)(
                            part[0][0], 
                            part[1], 
                            part[2]
                            ) for part in zip(dd_bins, dd_vals, dd_flags)]

    group_chunks = [dask.delayed(groupby_partition.create_bin_groups_sort)(c[0], c[1], c[2]) for c in group_bins_sort] 
    function_chunks = [dask.delayed(groupby_partition.apply_grid_median)(c[1], c[2]) for c in group_chunks]
    median_grid = dask.delayed(groupby_partition.combine_function_partitions)(function_chunks)

    if client:
        median_grid = client.compute(median_grid).result()
    else:
        median_grid = median_grid.compute()

    return median_grid


def check_exising_flags(ds_ind, stokes='I', client=None):
    """ Check the existing flags in the input Measurement Set."""
    
    # Determine which polarization state to grid and flag
    if stokes=='I':
        flags = ds_ind.FLAG.data[:,0] | ds_ind.FLAG.data[:,-1]
    elif stokes=='Q':
        flags = ds_ind.FLAG.data[:,0] | ds_ind.FLAG.data[:,-1]
    elif stokes=='U':
        flags = ds_ind.FLAG.data[:,1] | ds_ind.FLAG.data[:,2]
    elif stokes=='V':
        flags = ds_ind.FLAG.data[:,1] | ds_ind.FLAG.data[:,2]
    elif stokes=='A':
        flags = da.sum(ds_ind.FLAG.data, axis=1, dtype=np.bool)
    else:
        raise ValueError(f"check_existing_flags: the stokes argument, \
        '{stokes}', is not currently implemented, please select another value.")
    
    flag_loc = da.where(flags==True)
    if not client is None:
        flag_loc = client.compute(flag_loc)
    else: 
        flag_loc = flag_loc[0].compute()

    nflags = len(flag_loc)
    nrows = len(flags)
    
    print( f"Rows alrady flagged: {(100*nflags/nrows):.1f}% ({nflags}/{nrows}),\
 in file \"{ds_ind.attrs['Measurement Set']}\"." ) 


def map_amplitude_grid(
    ds_ind, 
    data_columns, 
    stokes='I', 
    chunk_size:int=10**6, 
    return_index:bool=False
):
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
    dd_flgs = (ds_ind.FLAG[:,data_columns[0]].data | 
               ds_ind.FLAG[:,data_columns[1]].data)

    if stokes=='I':
        dd_vals = (np.absolute(ds_ind.DATA[:,data_columns[0]].data + 
                               ds_ind.DATA[:,data_columns[1]].data))
    elif stokes=='Q':
        dd_vals = (np.absolute(ds_ind.DATA[:,data_columns[0]].data - 
                               ds_ind.DATA[:,data_columns[1]].data))

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
    group_chunks = [dask.delayed(groupby_apply.group_bin_flagval_wrap)(
                            part[0][0], 
                            part[1], 
                            part[2], 
                            init_index=(chunk_size*kth)
                            ) for kth, part in enumerate(zip(bin_partitions, 
                                                             val_partitions, 
                                                             flg_partitions)
                                                        )
                    ]    
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


def bin_rms(a):
    mu = np.median(a)
    rms = np.sqrt(np.sum((a-mu)**2))
    return rms


def bin_rms_grid(ds_ind, flag_list):
    """ Create a two dimensional UV-grid with the RMS value for each bin, after
    removing flags in a provided list of flags. 
    
    Parameters 
    ----------
    ds_ind : xarray.dataset
        Dataset for input MS file.
    flag_list : array of ints 
        Indicies for flagged rows in the dataset.
        
    Returns
    -------
    rms_grid : array of shape (ubins, vbins)
        A two-dimensional array wiht the RMS value for the values in each  UV bin.
        
    """
    
    
    flags = np.zeros((len(ds_ind.newrow)), dtype=bool)
    flags[flag_list] = True
    
    # Use Calculated Flags Column
    ubins = ds_ind.U_bins.data
    vbins = ds_ind.V_bins.data
    vals = (da.absolute(ds_ind.DATA[:,0].data + ds_ind.DATA[:,-1].data))
    
    print("Processing RMS Grid with ", np.sum(1*flags), len(flag_list), "flags.")
    
    p = da.arange(len(ds_ind.newrow), dtype=np.int64, chunks=ubins.chunks)
    
    dd_bins = da.stack([ubins, vbins, p]).T
    
    dd_bins = dd_bins.rechunk((ubins.chunks[0], 3))
    
    bins = dd_bins.compute()
    vals = vals.compute()
    
    bins_sort, vals_sort, null_flags = groupby_partition.sort_bins(bins, vals, flags)
    
    print(len(vals_sort))
    
    uv_ind, values, grid_row_map, null_flags = groupby_partition.create_bin_groups_sort(bins_sort, 
                                                                                        vals_sort, 
                                                                                        null_flags)
    
    print(len(values))
    
    rms_grid = groupby_partition.apply_grid_function(values, grid_row_map, bin_rms)
    
    return rms_grid
