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

    #return np.rad2deg(theta)*3600.0 # Convert into arcsec

    #max_beam_size = 1./theta * (180/np.pi)
    #d_lambda = diameter / max_lambda
    #max_beam_size = 1/d_lambda * (180/np.pi) * 3600 # Convert to arcseconds
    #return max_beam_size


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

    # Get spectral window table information
    spw_table_name = 'SPECTRAL_WINDOW'
    spw_col = 'CHAN_FREQ'

    da_vis = ds_ms[datacolumn]
    spw = xds_from_table(f'{msfile}::{spw_table_name}', columns=['NUM_CHAN', spw_col], column_keywords=True)
    ds_spw, spw_attrs = spw[0][ddid], spw[1]

    col_units = spw_attrs['CHAN_FREQ']['QuantumUnits'][0]

    print(f"Selected column {spw_col} from {spw_table_name} with units {col_units}")

    nchan = int(ds_spw.NUM_CHAN.data.compute()[0])
    nrow = len(ds_ms.ROWID)

    relfile = os.path.basename(msfile)
    if not len(relfile):
        relfile = os.path.basename(msfile.rstrip('/'))

    #fileparts = msfile.split('/')
    #if fileparts[-1]:
    #    relfile = msfile.split('/')[-1]
    #else:
    #    relfile = msfile.split('/')[-2]

    print(f'Processing dataset {relfile} with {nchan} channels and {nrow} rows.')

    # Create UV values scaled using spectral window frequencies
    chan_freq = ds_spw.CHAN_FREQ
    chan_wavelength = scipy.constants.c/chan_freq

    chan_wavelength = chan_wavelength.squeeze()
    chan_wavelength = xr.DataArray(chan_wavelength.data, dims=['chan'])

    uvw_chan = xr.concat([ds_ms.UVW[:,0] / chan_wavelength, ds_ms.UVW[:,1] / chan_wavelength, ds_ms.UVW[:,2] / chan_wavelength], 'uvw')
    uvw_chan = uvw_chan.transpose('row', 'chan', 'uvw')

    # Compute UV bins
    ds_vis = xr.Dataset(data_vars = {'DATA': da_vis, 'UVW_scaled': uvw_chan})

    uval, vval = ds_vis.UVW_scaled[:,:,0], ds_vis.UVW_scaled[:,:,1]

    n = ds_vis.UVW_scaled.shape[0] * ds_vis.UVW_scaled.shape[1] # Total row count

    uvlimit = [[da.min(ds_vis.UVW_scaled[:,:,0]).compute(), da.max(ds_vis.UVW_scaled[:,:,0]).compute()],
               [da.min(ds_vis.UVW_scaled[:,:,1]).compute(), da.max(ds_vis.UVW_scaled[:,:,1]).compute()]]

    if method=='statistical':
        std_k = [float(uval.reduce(np.std)),
                 float(vval.reduce(np.std))]
        print(f"The calculated STD of the UV distribution is {std_k[0]} by {std_k[1]} lambda.")

        binwidth = [x * (3.5/n**(1./4)) for x in std_k]

        bincount = [int((uvlimit[0][1] - uvlimit[0][0])/binwidth[0]),
                    int((uvlimit[1][1] - uvlimit[1][0])/binwidth[1])]

    if method=='physical':
        antennas = msmd.get_antenna_list(verbose=False)
        #ang_res = compute_angular_resolution(uvw_chan)

        #max_beam_size = compute_fov(chan_freq, antennas)
        #bincount = max_beam_size / ang_res

        fov = compute_fov(chan_freq, antennas)
        binwidth = 1./fov # in lambda
        binwidth = [int(binwidth), int(binwidth)]
        print(f"The calculated FoV is {np.rad2deg(fov)} deg.")

        #print(f"The calculated resolution of the instrument is {ang_res:.2f} arcseconds and the calculated field of view is {max_beam_size:.1f} arcseconds.")
        bincount = [int((uvlimit[0][1] - uvlimit[0][0])/binwidth[0]),
                    int((uvlimit[1][1] - uvlimit[1][0])/binwidth[1])]
    uvbins = [np.linspace( uvlimit[0][0], uvlimit[0][1], bincount[0] ),
              np.linspace( uvlimit[1][0], uvlimit[1][1], bincount[1] )]

    print(f"Creating a UV-grid with ({bincount[0]}, {bincount[1]}) bins with bin size {binwidth[0]} by {binwidth[1]} lambda.")

    uval_dig = xr.apply_ufunc(da.digitize, ds_vis.UVW_scaled[:,:,0], uvbins[0], dask='allowed')
    vval_dig = xr.apply_ufunc(da.digitize, ds_vis.UVW_scaled[:,:,1], uvbins[1], dask='allowed')

    uv_index = xr.concat([uval_dig, vval_dig], 'uvw')
    uv_index = uv_index.transpose('row', 'chan', 'uvw')

    ds_ind = xr.Dataset( data_vars = {'DATA': da_vis, 'UV': uvw_chan[:,:,:2]}, coords = {'U_bins': uval_dig, 'V_bins': vval_dig})

    ds_ind = ds_ind.stack(newrow=['row', 'chan']).transpose('newrow', 'uvw', 'corr')
    ds_ind = ds_ind.drop('ROWID')
    ds_ind = ds_ind.chunk({'corr': 4, 'uvw': 2, 'newrow': chunksize})
    ds_ind = ds_ind.unify_chunks()

    return ds_ind, uvbins
