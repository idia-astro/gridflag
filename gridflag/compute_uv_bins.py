#!/usr/bin/python
"""
Compute UV Bins

Use XArray, Dask, and Numpy to load CASA Measurement Set (MS) data and
create binned UV data.

Todo:
    [ ] Add uv range parameters
    [ ] Add hermitian folding code (do we need it?)
    [X] What do we do with existing flags
    [X] Redo gridding based on resolution / fov of data
    [X] Fix write function to take pol argument
    [X] Fix write function to handle split-spw files and compliance with MS 2.0
"""

import os
import numpy as np
import scipy.constants

import dask
import dask.array as da

import xarray as xr
from daskms import xds_from_table, xds_from_ms, xds_to_table
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time

import logging

# Add colours for warnings and errors
logging.addLevelName(logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)-20s %(levelname)-8s %(message)s',
    handlers=[
        logging.FileHandler("gridflag.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()


import numba as nb

import logging
# Add colours for warnings and errors
logging.addLevelName(
    logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(
    logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)-20s %(levelname)-8s %(message)s',
    handlers=[
        logging.FileHandler("plumber.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()

from .listobs_daskms import ListObs as listobs


def compute_angular_resolution(uvw_chan):
    '''Compute the angular resolution as set by the psf / dirty beam.'''
    uv_dist = np.sqrt(uvw_chan[:,:,0]**2 + uvw_chan[:,:,1]**2)
    max_baseline = np.max(uv_dist).data.compute()
    ang_res = (1/max_baseline) * (180/np.pi) * 3600  # convert to arcseconds
    return ang_res



def list_fields(msmd):
    """Print a list fields in measurement set.

    Parameters
    ----------
    msmd : listobs-object
    """
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


@nb.jit(nopython=True, nogil=True, cache=True)
def compute_longest_baseline(baseline_lengths):
    """
    Given a list of baseline lengths, compute the largest. The input baseline lengths are
    with reference to the array centre, so the distance between every pair of antennas
    needs to be computed.
    """

    maxbase = 0
    nant = baseline_lengths.shape[0]

    for mm in range(nant):
        for nn in range(nant):
            if mm == nn:
                continue

            blen = baseline_lengths[mm] - baseline_lengths[nn]

            blen2 = np.sqrt(blen[0]**2 + blen[1]**2 + blen[2]**2)
            if blen2 > maxbase:
                maxbase = blen2

    return maxbase



def compute_uv_from_ms(msfile, fieldid, spw):
    """
    Compute the maximum UV value from the antenna and source positions.
    """

    antds = xds_from_table(msfile+'::ANTENNA')
    pos = antds[0]['POSITION'].compute()
    # Get the central position of the array
    meanpos = np.mean(pos, axis=0)

    # Get baseline lengths
    baseline = pos - meanpos
    maxbaseline = compute_longest_baseline(baseline.data)

    array_loc = EarthLocation.from_geocentric(x=meanpos[0], y=meanpos[1], z=meanpos[2], unit='m')
    fds = xds_from_table(msfile + '::FIELD')
    # Direction of field
    dirs = fds[0]['DELAY_DIR'].compute()
    try:
        dirs = dirs[fieldid][0]
    except IndexError as e:
        print(f"Field id {fieldid} does not appear to exist in the MS.")
        raise

    skydir = SkyCoord(dirs[0], dirs[1], unit='rad')
    maxfreq = np.amax(spw['CHAN_FREQ'].compute().data)

    msds = xds_from_ms(msfile, columns=['TIME'], group_cols=['FIELD_ID', 'DATA_DESC_ID'])
    time = msds[0]['TIME']

    begtime = time[0].compute()/86400
    endtime = time[-1].compute()/86400
    print(begtime.data, endtime.data)

    midtime = (begtime + endtime)/2.

    times = Time([begtime, midtime, endtime], format='mjd', scale='utc')
    sidereal_time = times.sidereal_time(kind='mean', longitude = array_loc.lon)
    ha = np.deg2rad((sidereal_time - skydir.ra).deg)
    dec = np.deg2rad(skydir.dec.deg)

    uvw = []
    for hh in ha:
        transform_mat = np.asarray([[np.sin(hh), np.cos(hh), 0],
                                   [-np.sin(dec)*np.cos(hh), np.sin(dec)*np.sin(hh), np.cos(dec)],
                                   [np.cos(dec)*np.cos(hh), -np.cos(dec)*np.sin(hh), np.sin(dec)]
                                  ])
        uvw.append(np.dot(transform_mat, maxbaseline))

    uvlen = [np.hypot(uv[0], uv[1]) for uv in uvw]
    lambd = 299792458/maxfreq

    return np.amax(uvlen)/lambd



def get_data_column(msfile, datacolumn, group_cols=['FIELD_ID']):
    # FIXME
    # This cascading logic is only because of the lack of easily configurable command line arguments at the moment.
    # Ideally we would want the program to error out and die if the requested column doesn't exist, because unexpected things can happen otherwise.
    # But in order to test gridflag inclusion in self-cal without modifying the existing script this logic is implemented
    # Pre-selfcal the residual column won't exist, so it'll cascade down to DATA
    # During selfcal the residual column _will_ exist (*should* exist) and we can flag on the residuals
    # Dirty & hacky - fix ASAP.
    if datacolumn == 'RESIDUAL':
        try:
            ms = xds_from_ms(msfile, columns=['CORRECTED_DATA', 'MODEL', 'UVW', 'FLAG'], group_cols=group_cols, table_keywords=True, column_keywords=True)
            ms['RESIDUAL'] = ms['CORRECTED_DATA'] - ms['MODEL']
        except RuntimeError:
            logger.error("RESIDUAL column cannot be constructed - attempting to read CORRECTED_DATA column instead.")

    else:
        # Data column is not residual, so just read the regular column
        ms = xds_from_ms(msfile, columns=[datacolumn, 'UVW', 'FLAG'], group_cols=group_cols, table_keywords=True, column_keywords=True)

    return ms



def load_ms_file(msfile, fieldid=None, datacolumn='RESIDUAL', method='physical', ddid=0, chunksize:int=10**7, bin_count_factor=1):
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
    bin_count_factor : float
        A factor to control binning if the automatic binning doesn't work
        right. A factor of 0.5 results in half the bins in u and v.
        default: 1.

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
    msmd = listobs(msfile, 'DATA')

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


    # Print uv-grid information to console
    title_string = "Compute UV bins"
    logger.info(f"{title_string:^80}")
    logger.info('_'*80)

    ms = get_data_column(msfile, datacolumn, group_cols=['FIELD_ID'])

    # Split the dataset and attributes
    try:
        ds_ms, ms_attrs = ms[0][fieldid], ms[1:]
    except:
        list_fields(msmd)
        raise ValueError(f"The fieldid value of {fieldid} is not a valid field in this dataset.")


    # Get spectral window table information
    spw_table_name = 'SPECTRAL_WINDOW'
    spw_col = 'CHAN_FREQ'

    spw = xds_from_table(f'{msfile}::{spw_table_name}', columns=['NUM_CHAN', spw_col], column_keywords=True, group_cols="__row__")
    ds_spw, spw_attrs = spw[0][0], spw[1]

    col_units = spw_attrs['CHAN_FREQ']['QuantumUnits'][0]
    logger.info(f"Selected column {spw_col} from {spw_table_name} with units {col_units}.")

    nchan = int(ds_spw.NUM_CHAN.data.compute()[0])
    nrow = len(ds_ms.ROWID)
    ncorr = len(ds_ms.corr)

    relfile = os.path.basename(msfile)
    if not len(relfile):
        relfile = os.path.basename(msfile.rstrip('/'))

    logger.info(f'Processing dataset {relfile} with {nchan} channels and {nrow} rows.')

#     maxuv = compute_uv_from_ms(msfile, fieldid, ds_spw)
#     uvlimit = [0, maxuv],[0, maxuv]
#     print("UV limit is ", uvlimit)
    # Compute the min and max of the unscaled UV coordinates to calculate the grid boundaries
    bl_limits = [da.min(ds_ms.UVW[:,0]).compute(), da.max(ds_ms.UVW[:,0]).compute(), da.min(ds_ms.UVW[:,1]).compute(), da.max(ds_ms.UVW[:,1]).compute()]

    # Compute the min and max spectral window channel and convert to wavelength
    chan_wavelength_lim = np.array([[scipy.constants.c/np.max(spw_.CHAN_FREQ.data.compute()), scipy.constants.c/np.min(spw_.CHAN_FREQ.data.compute())] for spw_ in spw[0]])

    # Compute the scaled limits of the UV grid by dividing the UV boundaries by the channel boundaries
    uvlimit = [bl_limits[0]/np.min(chan_wavelength_lim), bl_limits[1]/np.min(chan_wavelength_lim)], [bl_limits[2]/np.min(chan_wavelength_lim), bl_limits[3]/np.min(chan_wavelength_lim)]

    logger.info(f"UV limit is {uvlimit[0][0]:.2f} - {uvlimit[0][1]:.2f}, {uvlimit[1][0]:.2f} - {uvlimit[1][1]:.2f}")

    if method=='statistical':
        std_k = [float(uval.reduce(np.std)),
                 float(vval.reduce(np.std))]
        logger.info(f"The calculated STD of the UV distribution is {std_k[0]} by {std_k[1]} lambda.")

        binwidth = [x * (3.5/n**(1./4)) for x in std_k]

        logger.info(f"Using statistical estimation of bin widthds")
        logger.info(f"The calculated UV bin width is {binwidth[0]} {binwidth[1]} lambda.")

        bincount = [int((uvlimit[0][1] - uvlimit[0][0])/binwidth[0]),
                    int((uvlimit[1][1] - uvlimit[1][0])/binwidth[1])]

    if method=='physical':
        '''
        Calculate the field of view of the antenna, given the
        observing frequency and antenna diameter.
        '''
        antennas = msmd.get_antenna_list(verbose=False)

        diameter = np.average([a['diameter'] for a in antennas])
        max_lambda = np.max(chan_wavelength_lim)

        fov = 1.22 * max_lambda/diameter

        binwidth = 1./fov
        binwidth = [int(binwidth), int(binwidth)]

        logger.info(f"Using physical estimation of bin widthds")
        logger.info(f"The calculated FoV is {np.rad2deg(fov):.2f} deg.")
        logger.info(f"The calculated UV bin width is {binwidth[0]} {binwidth[1]} lambda.")

        bincount = [int(bin_count_factor*(uvlimit[0][1] - uvlimit[0][0])/binwidth[0]),
                    int(bin_count_factor*(uvlimit[1][1] - uvlimit[1][0])/binwidth[1])]

    uvbins = [np.linspace( uvlimit[0][0], uvlimit[0][1], bincount[0] ),
              np.linspace( uvlimit[1][0], uvlimit[1][1], bincount[1] )]

    # Reload the Main table grouped by DATA_DESC_ID
    ms = get_data_column(msfile, datacolumn, group_cols=['FIELD_ID', 'DATA_DESC_ID'])
    #ms = xds_from_ms(msfile, columns=[datacolumn, 'UVW', 'FLAG'], group_cols=['FIELD_ID', 'DATA_DESC_ID'], table_keywords=True, column_keywords=True)
    ds_ms, ms_attrs = ms[0], ms[1:]

    dd = xds_from_table(f"{msfile}::DATA_DESCRIPTION")
    dd = dd[0].compute()
    ndd = len(dd.ROWID)
    nrows = 0

    # Use the DATA_DESCRIPTION table to process each subset of data (different ddids can have
    # a different number of channels). The subsets will be stacked after the channel scaling.
    ds_bindex = []

    logger.info(f"Creating a UV-grid with ({bincount[0]}, {bincount[1]}) bins with bin size {binwidth[0]:.1f} by {binwidth[1]:.1f} lambda.")

    logger.info(f"\nField, Data ID, SPW ID, Channels")

    for ds_ in ds_ms:
        fid = ds_.attrs['FIELD_ID']
        ddid = ds_.attrs['DATA_DESC_ID']

        if fid != fieldid:
            logger.info(f"Skipping channel: {fid}.")
            continue

        spwid = int(dd.SPECTRAL_WINDOW_ID[ddid].data)
        chan_freq = spw[0][spwid].CHAN_FREQ.data[0]
        chan_wavelength = scipy.constants.c/chan_freq

#             chan_wavelength = chan_wavelength.squeeze()
        chan_wavelength = xr.DataArray(chan_wavelength, dims=['chan'])

        logger.info(f"{fieldid:<5}  {ddid:<7}  {spwid:<6}  {len(chan_freq):<8}")

        # I think we can remove the W channel part of this to save some compute (ds_.UVW[:,2])
        uvw_chan = xr.concat([ds_.UVW[:,0] / chan_wavelength, ds_.UVW[:,1] / chan_wavelength, ds_.UVW[:,2] / chan_wavelength], 'uvw')
        uvw_chan = uvw_chan.transpose('row', 'chan', 'uvw')

        uval_dig = xr.apply_ufunc(da.digitize, uvw_chan[:,:,0], uvbins[0], dask='allowed', output_dtypes=[np.int32])
        vval_dig = xr.apply_ufunc(da.digitize, uvw_chan[:,:,1], uvbins[1], dask='allowed', output_dtypes=[np.int32])

#             ds_ind = xr.Dataset(data_vars = {'DATA': ds_[datacolumn], 'FLAG': ds_['FLAG'], 'UV': uvw_chan[:,:,:2]}, coords = {'U_bins': uval_dig.astype(np.int32), 'V_bins': vval_dig.astype(np.int32)})
#
#             return ds_ind
#
#             ds_ind = ds_ind.stack(newrow=['row', 'chan']).transpose('newrow', 'uvw', 'corr')
#             ds_ind = ds_ind.drop('ROWID')
#             ds_ind = ds_ind.chunk({'corr': 4, 'uvw': 2, 'newrow': chunksize})
#             ds_ind = ds_ind.unify_chunks()

        # Avoid calling xray.dataset.stack, as it leads to an intense multi-index shuffle
        # that does not seem to be dask-backed and runs on the scheduler.

        da_data = ds_[datacolumn].data.reshape(-1, ncorr)
        da_flag = ds_.FLAG.data.reshape(-1, ncorr)

        ds_ind = xr.Dataset(data_vars = {'DATA': (("newrow", "corr"), da_data), 'FLAG': (("newrow", "corr"), da_flag)}, coords = {'U_bins': (("newrow"), uval_dig.astype(np.int32).data.ravel()), 'V_bins': (("newrow"), vval_dig.astype(np.int32).data.ravel())})
        ds_ind = ds_ind.chunk({'corr': ncorr, 'newrow': chunksize})
        ds_ind = ds_ind.unify_chunks()

        nrows+=len(ds_ind.newrow)

        ds_bindex.append(ds_ind)

    logger.info(f"\nProcessed {ndd} unique data description IDs comprising {nrows} rows.")

    ds_ind = xr.concat(ds_bindex, dim="newrow")
    ds_ind.attrs = {'Measurement Set': msfile, 'Field': fieldid}

    return ds_ind, uvbins



def load_ms_file_splitspw(msfile, fieldid=None, datacolumn='DATA', method='physical', ddid=0, chunksize:int=10**7, bin_count_factor=1):
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
    bin_count_factor : float
        A factor to control binning if the automatic binning doesn't work
        right. A factor of 0.5 results in half the bins in u and v.
        default: 1.

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

    # Print uv-grid information to console
    title_string = "Compute UV bins"
    print(f"{title_string:^80}")
    print('_'*80)

    ms = xds_from_ms(msfile, columns=[datacolumn, 'UVW', 'FLAG'], group_cols=['FIELD_ID'], table_keywords=True, column_keywords=True)

    # Split the dataset and attributes
    try:
        ds_ms, ms_attrs = ms[0][fieldid], ms[1:]
    except:
        list_fields(msmd)
        raise ValueError(f"The fieldid value of {fieldid} is not a valid field in this dataset.")


    # Get spectral window table information
    spw_table_name = 'SPECTRAL_WINDOW'
    spw_col = 'CHAN_FREQ'

    spw = xds_from_table(f'{msfile}::{spw_table_name}', columns=['NUM_CHAN', spw_col], column_keywords=True, group_cols="__row__")
    ds_spw, spw_attrs = spw[0][0], spw[1]

    col_units = spw_attrs['CHAN_FREQ']['QuantumUnits'][0]
    print(f"Selected column {spw_col} from {spw_table_name} with units {col_units}.")

    nchan = int(ds_spw.NUM_CHAN.data.compute()[0])
    nrow = len(ds_ms.ROWID)
    ncorr = len(ds_ms.corr)

    relfile = os.path.basename(msfile)
    if not len(relfile):
        relfile = os.path.basename(msfile.rstrip('/'))

    print(f'Processing dataset {relfile} with {nchan} channels and {nrow} rows.')

#     maxuv = compute_uv_from_ms(msfile, fieldid, ds_spw)
#     uvlimit = [0, maxuv],[0, maxuv]
#     print("UV limit is ", uvlimit)
    # Compute the min and max of the unscaled UV coordinates to calculate the grid boundaries
    bl_limits = [da.min(ds_ms.UVW[:,0]).compute(), da.max(ds_ms.UVW[:,0]).compute(), da.min(ds_ms.UVW[:,1]).compute(), da.max(ds_ms.UVW[:,1]).compute()]

    # Compute the min and max spectral window channel and convert to wavelength
    chan_wavelength_lim = np.array([[scipy.constants.c/np.max(spw_.CHAN_FREQ.data.compute()), scipy.constants.c/np.min(spw_.CHAN_FREQ.data.compute())] for spw_ in spw[0]])

    # Compute the scaled limits of the UV grid by dividing the UV boundaries by the channel boundaries
    uvlimit = [bl_limits[0]/np.min(chan_wavelength_lim), bl_limits[1]/np.min(chan_wavelength_lim)], [bl_limits[2]/np.min(chan_wavelength_lim), bl_limits[3]/np.min(chan_wavelength_lim)]

    print(f"UV limit is {uvlimit[0][0]:.2f} - {uvlimit[0][1]:.2f}, {uvlimit[1][0]:.2f} - {uvlimit[1][1]:.2f}")


    if method=='statistical':
        std_k = [float(uval.reduce(np.std)),
                 float(vval.reduce(np.std))]
        print(f"The calculated STD of the UV distribution is {std_k[0]} by {std_k[1]} lambda.")

        binwidth = [x * (3.5/n**(1./4)) for x in std_k]

        bincount = [int((uvlimit[0][1] - uvlimit[0][0])/binwidth[0]),
                    int((uvlimit[1][1] - uvlimit[1][0])/binwidth[1])]

    if method=='physical':
        '''
        Calculate the field of view of the antenna, given the
        observing frequency and antenna diameter.
        '''
        antennas = msmd.get_antenna_list(verbose=False)

        diameter = np.average([a['diameter'] for a in antennas])
        max_lambda = np.max(chan_wavelength_lim)

        fov = 1.22 * max_lambda/diameter

        binwidth = 1./fov
        binwidth = [int(binwidth), int(binwidth)]
        print(f"The calculated FoV is {np.rad2deg(fov):.2f} deg.")

        bincount = [int(bin_count_factor*(uvlimit[0][1] - uvlimit[0][0])/binwidth[0]),
                    int(bin_count_factor*(uvlimit[1][1] - uvlimit[1][0])/binwidth[1])]


    uvbins = [np.linspace( uvlimit[0][0], uvlimit[0][1], bincount[0] ),
              np.linspace( uvlimit[1][0], uvlimit[1][1], bincount[1] )]



    # Reload the Main table grouped by DATA_DESC_ID
    ms = xds_from_ms(msfile, columns=[datacolumn, 'UVW', 'FLAG'], group_cols=['FIELD_ID', 'DATA_DESC_ID'], table_keywords=True, column_keywords=True)
    ds_ms, ms_attrs = ms[0], ms[1:]

    dd = xds_from_table(f"{msfile}::DATA_DESCRIPTION")
    dd = dd[0].compute()
    ndd = len(dd.ROWID)
    nrows = 0

    ds_bindex = []

    print(f"Creating a UV-grid with ({bincount[0]}, {bincount[1]}) bins with bin size {binwidth[0]:.1f} by {binwidth[1]:.1f} lambda.")

    print(f"\nField, Data ID, SPW ID, Channels")



    nddid = [len(ds.chan) for ds in ds_ms]
    (ddids, dd_inv) = np.unique(nddid, return_inverse=True)

    # Outer loop through SPW's that have the same channel count for efficiency reasons
    for dind, nchan in enumerate(ddids):

            ddid_list = np.array(np.where(dd_inv==dind)[0])
            ms = xds_from_ms(msfile, columns=[datacolumn, 'UVW', 'FLAG', 'DATA_DESC_ID'], group_cols=['FIELD_ID'], taql_where="DATA_DESC_ID IN ({})".format(",".join(np.array(ddid_list, dtype='str'))) )

            ds_ = ms[0]

            uvw_chan_list = []

            # Loop through DATA_DESC_ID to compute UVW channels for each spectral window
            for ddid in ddid_list:

                spwid = int(dd.SPECTRAL_WINDOW_ID[ddid].data)

                chan_freq = spw[0][spwid].CHAN_FREQ.data[0]
                chan_wavelength = scipy.constants.c/chan_freq

                print(f"{fieldid:<5}  {ddid:<7}  {spwid:<6}  {nchan:<8}")

                chan_wavelength = chan_wavelength.squeeze()
                chan_wavelength = xr.DataArray(chan_wavelength, dims=['chan'])

                ds_mask = ds_.where(ds_.DATA_DESC_ID==ddid, other=0)

                uvw_chan_list.append(xr.concat([ds_mask.UVW[:,0] / chan_wavelength, ds_mask.UVW[:,1] / chan_wavelength], 'uv'))

            uv_sum = sum(uvw_chan_list)

            uv_chan = uv_sum.transpose('row', 'chan', 'uv')

            uval_dig = xr.apply_ufunc(da.digitize, uv_chan[:,:,0], uvbins[0], dask='allowed', output_dtypes=[np.int32])
            vval_dig = xr.apply_ufunc(da.digitize, uv_chan[:,:,1], uvbins[1], dask='allowed', output_dtypes=[np.int32])

            da_data = ds_.DATA.data.reshape(-1, ncorr)
            da_flag = ds_.FLAG.data.reshape(-1, ncorr)

            ds_ind = xr.Dataset(data_vars = {'DATA': (("newrow", "corr"), da_data), 'FLAG': (("newrow", "corr"), da_flag)}, coords = {'U_bins': (("newrow"), uval_dig.astype(np.int32).data.ravel()), 'V_bins': (("newrow"), vval_dig.astype(np.int32).data.ravel())})
            ds_ind = ds_ind.chunk({'corr': ncorr, 'newrow': chunksize})
            ds_ind = ds_ind.unify_chunks()

            nrows+=len(ds_ind.newrow)

            ds_bindex.append(ds_ind)

    print(f"\nProcessed {ndd} unique data description IDs comprising {nrows} rows.")

    ds_ind = xr.concat(ds_bindex, dim="newrow")
    ds_ind.attrs = {'Measurement Set': msfile}

    return ds_ind, uvbins





def write_ms_file(msfile, ds_ind, flag_ind_list, fieldid, stokes='I', overwrite=False, datacolumn="DATA", chunk_size=10**6, client=None):
    """ Convert flags to the correct format and write flag column to a
    measurement set (MS).

    Parameters
    ----------
    msfile : string
        Location of the MS file.
    ds_ind: xarray.Dataset
        The contents of the main table of the MS columns DATA and UVW flattened
        and scaled by wavelength (from compute_uv_bins.load_ms_file function).
    flag_ind_list : array-like
        One dimensional array with indicies of flagged visibilities.
    fieldid : int
        The unique identifier of the field to be flagged.
    chunksize : int
        Size of chunks to be used with Dask.
    """

    print("Load ms file for writing.")
    ms = xds_from_ms(msfile, columns=[datacolumn, 'UVW', 'FLAG'], group_cols=['FIELD_ID', 'DATA_DESC_ID'])

    print("Load existing flags.")
    flag_ind_ms = ds_ind.FLAG.data.compute()

    print("Create empty flag column.")
    flag_ind_col = np.zeros((len(ds_ind.newrow)), dtype=bool)
    flag_ind_col[flag_ind_list] = True

    if stokes=='I':
        data_columns = [0, -1]

    print("Rearange and combine flags for pol states and cols.")
    for col in data_columns:
        if overwrite:
            flag_ind_ms[:,col] = flag_ind_col
        else:
            flag_ind_ms[:,col] = flag_ind_col | flag_ind_ms[:,col]

    print("Done with that.")
    da_flag_rows = da.from_array(flag_ind_ms)
    ds_ind_flag = ds_ind.assign(FLAG=(("newrow", "corr"), da_flag_rows))

    print(f"nMS  Field  DDID  nPol  nChan  nRows      Row-span")

    start_row = 0

    for i_ms, ds_ms in enumerate(ms):
        fid = ds_ms.attrs['FIELD_ID']
        if not(fieldid==fid):
            print("skipping field {}".format(fid))
            continue
        ddid = ds_ms.attrs['DATA_DESC_ID']
        npol = ds_ms.FLAG.data.shape[2]
        nrow = len(ds_ms.FLAG)
        nchan = len(ds_ms.chan)
        end_row = start_row + nrow*nchan

        print(f"{i_ms:<3}  {fid:<5}  {ddid:<4}  {npol:<4}  {nchan:<5}  {nrow:<9}  {start_row}-{end_row}")
        ds_iflg = ds_ind_flag.isel(newrow=slice(start_row, end_row)).FLAG.data.reshape(nrow, nchan, npol)
        start_row = end_row
        ds_ms = ds_ms.assign(FLAG=( ("row", "chan", "corr"), ds_iflg ) )
        ds_ms = ds_ms.unify_chunks()

        ms[i_ms] = ds_ms

    print("Saving to file.")
    writes = xds_to_table(ms, msfile, ["FLAG"])
    if client:
        client.compute(writes)
    else:
        dask.compute(writes)
