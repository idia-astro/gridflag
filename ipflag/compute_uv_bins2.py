#!/usr/bin/python
"""
GridFlag

Use XArray, Dask, and Numpy to load CASA Measurement Set (MS) data and
create binned UV data.

Todo:
    [ ] Add uv range parameters
    [ ] Add hermitian folding code
    [X] Redo gridding based on resolution / fov of data
"""

import os
import numpy as np
import scipy.constants

import dask.array as da
import xarray as xr
from daskms import xds_from_table, xds_from_ms

from .listobs_daskms import ListObs as listobs

def compute_angular_resolution(uvw_chan):
    '''Compute the angular resolution as set by the psf / dirty beam.'''
    uv_dist = np.sqrt(uvw_chan[:,:,0]**2 + uvw_chan[:,:,1]**2)
    max_baseline = np.max(uv_dist).data.compute()
    ang_res = (1/max_baseline) * (180/np.pi) * 3600  # convert to arcseconds
    return ang_res

def compute_fov(chan_freq, antennas):
    '''
    Calculate the field of view of the antenna, given the
    observing frequency and antenna diameter.
    '''

    chan_freq_list = chan_freq.data.compute()

    diameter = np.average([a['diameter'] for a in antennas])

    max_freq, min_freq = chan_freq_list[0][-1], chan_freq_list[0][0]
    max_lambda = scipy.constants.c/min_freq

    theta = 1.22 * max_lambda/diameter
    return theta

def list_fields(msmd):
    field_list = msmd.get_fields(verbose=False)
    scan_list = msmd.get_scan_list(verbose=False)
    field_intent = {}
    for scan in scan_list:
        field_intent[scan['field_name']] = scan['intent']
    print(f"Fields: ({len(field_list)})")
    print(f"---------------------------")
    print("ID    Field Name      Intent    ")
    for i, field in enumerate(field_list):
        name = field['Name']
        print(f"{i:<5} {name:<15} {field_intent[name]}")

def load_ms_file(msfile, fieldid=None, datacolumn='DATA', method='physical', ddid=0, chunksize:int=10**7):
    """
    Load selected data from the measurement set (MS) file and convert to xarray
    DataArrays. Transform data for analysis.

    Parameters
    ----------
    msfile : string
        Location of the MS file.
    fieldid : int
        The unique identifier for the field that is to be analyzed (FIELD_ID
        in a Measurement Set). This information can be retreived using the
        'listobs_daskms' class.
    method : string
        String representing the method for binning the UV plane. Choices are
        'physical' (default) to use the telescope resolution and field of
        view to set bin size and number, or statistical to compute the bins
        based on a statistical MSE estimate.
    ddid : int
        The unique identifier for the data description id. Default is zero.
    chunksize : int
        Size of chunks to be used with Dask.

    Returns
    -------
    ds_ind: xarray.Dataset
        The contents of the main table of the MS columns DATA and UVW flattened
        and scaled by wavelength.
    uvbins: xarray.Dataset
        A dataset containing flattened visabilities with a binned UV set for
        each observation point.

    """
    
    # Load Metadata from the Measurement Set
    msmd = listobs(msfile, datacolumn)

    if fieldid==None:
        fields = msmd.get_fields(verbose=False)
        if(len(fields))==1:
            fieldid=0
        else:
            print("Error: Please choose a field from the list below.")
            list_fields(msmd)
            raise ValueError("Parameter, \'fieldid\', not set.")

    if not( (method=='statistical') or (method=='physical') ):
        raise ValueError('The \'method\' parameter should be either \'physical\' or \'statistical\'.')

    print('Compute UV Bins')
    print('---------------')
    
    ms = xds_from_ms(msfile, columns=[datacolumn, 'UVW'], group_cols=['FIELD_ID'], table_keywords=True, column_keywords=True)

    # Split the dataset and attributes
    try:
        ds_ms, ms_attrs = ms[0][fieldid], ms[1:]
    except:
        list_fields(msmd)
        raise ValueError(f"The fieldid value of {fieldid} is not a valid field in this dataset.")


    bl_limits = [da.min(ds_ms.UVW[:,0]).compute(), da.max(ds_ms.UVW[:,0]).compute(), da.min(ds_ms.UVW[:,1]).compute(), da.max(ds_ms.UVW[:,1]).compute()]

    # Get spectral window table information
    spw_table_name = 'SPECTRAL_WINDOW'
    spw_col = 'CHAN_FREQ'
    uvw_col = 'UVW'    
    
    spw = xds_from_table(f'{msfile}::{spw_table_name}', columns=['NUM_CHAN', spw_col], column_keywords=True, group_cols="__row__")
    ds_spw, spw_attrs = spw[0][0], spw[1]
    
    
    # Compute the min and max of the unscaled UV coordinates to calculate the grid boundaries
    chan_wavelength_lim = []
    for spw_ in spw[0]:
        spw_cf = np.array(spw_.CHAN_FREQ.data.compute())
        chan_wavelength_lim.append(np.array([scipy.constants.c/np.max(spw_cf), scipy.constants.c/np.min(spw_cf)]))
    chan_wavelength_lim = np.array(chan_wavelength_lim)

    # In order to avoid comuting out-of-order datasets in the case that the MS is split by spectral window or polarization, We use a heuristic where all spectral windows  are considered to find the minimum and maximum limits of the UV grid. A user can specify UV limits instead.    
    uvlimit = [bl_limits[0]/np.min(chan_wavelength_lim), bl_limits[1]/np.min(chan_wavelength_lim)], [bl_limits[2]/np.min(chan_wavelength_lim), bl_limits[3]/np.min(chan_wavelength_lim)]


    if method=='physical':
        '''
        Calculate the field of view of the antenna, given the
        observing frequency and antenna diameter.
        '''
        antennas = msmd.get_antenna_list(verbose=False)
        
        diameter = np.average([a['diameter'] for a in antennas])
        
        max_lambda = np.max(chan_wavelength_lim)
        theta = 1.22 * max_lambda/diameter
        
        binwidth = 1./theta
        binwidth = [int(binwidth), int(binwidth)]
        
        bincount = [int((uvlimit[0][1] - uvlimit[0][0])/binwidth[0]),
                    int((uvlimit[1][1] - uvlimit[1][0])/binwidth[1])]
                    
    uvbins = np.array([np.linspace( uvlimit[0][0], uvlimit[0][1], bincount[0] ),
              np.linspace( uvlimit[1][0], uvlimit[1][1], bincount[1] )], dtype=np.float64)


    # Reload the Main table grouped by DATA_DESC_ID
    ms = xds_from_ms(msfile, columns=[datacolumn, 'UVW'], group_cols=['FIELD_ID', 'DATA_DESC_ID'], table_keywords=True, column_keywords=True)
    ds_ms, ms_attrs = ms[0], ms[1:]

    dd = xds_from_table(f"{msfile}::DATA_DESCRIPTION")
    dd = dd[0].compute()


    # Use the DATA_DESCRIPTION table to process each subset of data (different ddids can have 
    # a different number of channels). The subsets will be stacked after the channel scaling.
    ds_bindex = []

    print(f"Field, Data ID, SPW ID, Channels")
    
    for ds_ in ds_ms:
            fid = ds_.attrs['FIELD_ID']
            ddid = ds_.attrs['DATA_DESC_ID']
    
            if fid != fieldid:
                print(f"Skipping channel: {fid}.")
                continue

            spwid = int(dd.SPECTRAL_WINDOW_ID[ddid].data)
            chan_freq = spw[0][spwid].CHAN_FREQ.data[0]
            chan_wavelength = scipy.constants.c/chan_freq
    
            chan_wavelength = chan_wavelength.squeeze()
            chan_wavelength = xr.DataArray(chan_wavelength, dims=['chan'])
    
            print(f"{fieldid:>5}, {ddid:<7}, {spwid:<6}, {len(chan_freq):<5}")
        
            uvw_chan = xr.concat([ds_.UVW[:,0] / chan_wavelength, ds_.UVW[:,1] / chan_wavelength, ds_.UVW[:,2] / chan_wavelength], 'uvw')
            uvw_chan = uvw_chan.transpose('row', 'chan', 'uvw')

            uval_dig = xr.apply_ufunc(da.digitize, uvw_chan[:,:,0], uvbins[0], dask='allowed')
            vval_dig = xr.apply_ufunc(da.digitize, uvw_chan[:,:,1], uvbins[1], dask='allowed')

            ds_ind = xr.Dataset(data_vars = {'DATA': ds_[datacolumn], 'UV': uvw_chan[:,:,:2]}, coords = {'U_bins': uval_dig, 'V_bins': vval_dig})
    
            ds_ind = ds_ind.stack(newrow=['row', 'chan']).transpose('newrow', 'uvw', 'corr')
            ds_ind = ds_ind.drop('ROWID')
            ds_ind = ds_ind.chunk({'corr': 4, 'uvw': 2, 'newrow': chunksize})
            ds_ind = ds_ind.unify_chunks()

            ds_bindex.append(ds_ind)
            
    return ds_bindex, uvbins