#!/usr/bin/python
"""
GridFlag 

Use XArray, Dask, and Numpy to load CASA Measurement Set (MS) data and 
create binned UV data.
"""

import numpy as np
import scipy.constants

import dask.array as da
import xarray as xr
from daskms import xds_from_table, xds_from_ms


def load_ms_file(msfile, attributes=False, chunksize:int=10**7):
    """
    Load selected data from the measurement set (MS) file and convert 
    to xarray DataArrays.

    Parameters
    ----------
    msfile: string
        Location of the MS file.
    
    Returns
    -------
    da_vis: xarray.DataArray
        The contents of the main table of the MS without metadata.
    ds_spw: xarray.DataSet
        The contents of the SPECTRAL_WINDOW table of the MS.

    Returns
    -------
    ds_ind: xarray.Dataset 
        A dataset containing flattened visabilities with a binned UV
        set for each observation point.
        
    """

    # Load Data from MS
    ds = xds_from_ms(msfile, table_keywords=True, column_keywords=True, group_cols=['DATA_DESC_ID'])

    # Split the dataset and attributes
    ds_, attr_ = ds[0][0], ds[1:]

    da_vis = ds_.DATA
    ds_spw = xds_from_table(f'{msfile}SPECTRAL_WINDOW', column_keywords=True)
    ds_spw = ds_spw[0][0]
    
    # Create UV values scaled using spectral window frequencies
    nchan = int(ds_spw.NUM_CHAN.data.compute()[0])

    chan_freq = ds_spw.CHAN_FREQ
    chan_wavelength = scipy.constants.c/chan_freq

    chan_wavelength = chan_wavelength.squeeze()
    chan_wavelength = xr.DataArray(chan_wavelength.data, dims=['chan'])

    uvw_chan = xr.concat([ds_.UVW[:,0] / chan_wavelength, ds_.UVW[:,1] / chan_wavelength, ds_.UVW[:,2] / chan_wavelength], 'uvw')
    uvw_chan = uvw_chan.transpose('row', 'chan', 'uvw')  
    
    # Compute UV bins
    ds_vis = xr.Dataset(data_vars = {'DATA': da_vis, 'UVW_scaled': uvw_chan})

    uval, vval = ds_vis.UVW_scaled[:,:,0], ds_vis.UVW_scaled[:,:,1]

    n = ds_vis.UVW_scaled.shape[0] * ds_vis.UVW_scaled.shape[1] # Total row count

    std_k = [float(uval.reduce(np.std)), 
             float(vval.reduce(np.std))]

    binwidth = [x * (3.5/n**(1./4)) for x in std_k]

    uvlimit = [[da.min(ds_vis.UVW_scaled[:,:,0]).compute(), da.max(ds_vis.UVW_scaled[:,:,0]).compute()],
               [da.min(ds_vis.UVW_scaled[:,:,1]).compute(), da.max(ds_vis.UVW_scaled[:,:,1]).compute()]]

    bincount = [int((uvlimit[0][1] - uvlimit[0][0])/binwidth[0]), 
                int((uvlimit[1][1] - uvlimit[1][0])/binwidth[1])]

    uvbins = [np.linspace( uvlimit[0][0], uvlimit[0][1], bincount[0] ),
              np.linspace( uvlimit[1][0], uvlimit[1][1], bincount[1] )]
              
    uval_dig = xr.apply_ufunc(da.digitize, ds_vis.UVW_scaled[:,:,0], uvbins[0], dask='allowed')
    vval_dig = xr.apply_ufunc(da.digitize, ds_vis.UVW_scaled[:,:,1], uvbins[1], dask='allowed')

    uv_index = xr.concat([uval_dig, vval_dig], 'uvw')
    uv_index = uv_index.transpose('row', 'chan', 'uvw')  

    ds_ind = xr.Dataset( data_vars = {'DATA': da_vis, 'UV': uvw_chan[:,:,:2]}, coords = {'U_bins': uval_dig, 'V_bins': vval_dig})

    ds_ind = ds_ind.stack(newrow=['row', 'chan']).transpose('newrow', 'uvw', 'corr')
    ds_ind = ds_ind.drop('ROWID')
    ds_ind = ds_ind.chunk({'corr': 4, 'uvw': 2, 'newrow': chunksize})
    ds_ind = ds_ind.unify_chunks()

    return ds_ind

