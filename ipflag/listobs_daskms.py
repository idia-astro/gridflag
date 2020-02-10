#!/usr/bin/python
"""List the summary of a data set.

This function attempts to reproduce the CASA 'listobs' task using python, astropy, daskms,
and xarray.  It is written in a modular fashion so that some or all of the summary data
can be returned by the class as a data structure.

Authors: Joseph Bochenek, Jeremy Smith
"""

__author__ = 'Joseph Bochenek'

import numpy as np

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u

from daskms import xds_from_table, xds_from_ms

StokesTypes= ['Undefined', 'I', 'Q', 'U', 'V', 'RR', 'RL', 'LR', 'LL', 'XX', 'XY', 'YX', 'YY', 'RX', 'RY', 'LX', 'LY', 'XR', 'XL', 'YR', 'YL', 'PP', 'PQ', 'QP', 'QQ', 'RCircular', 'LCircular', 'Linear', 'Ptotal', 'Plinear', 'PFtotal', 'PFlinear', 'Pangle']

def deg_min_sec(degrees = 0.0):
    """Format RA Decl output from astropy to CASA style

    Author: Jeremy Smith
    """
    if type(degrees) != 'float':
        try:
            degrees = float(degrees)
        except:
            print(f'\nERROR: Could not convert {type(degrees)} to float.')
            return 0
    minutes = degrees%1.0*60
    seconds = minutes%1.0*60

    return '%s.%s.%s' %(int(degrees), int(minutes), '{0:.2f}'.format(seconds))

class ListObs:
    """List metadata and summary data of Measurement Set.

    Load a measurement set (MS) file and get summary data or return properties for use in
    other programs.

    Parameters
    ----------
    filename : String
        The path to a measurement set to be used by the class.

    Example:
        myms = listobs(filename)
        myms.listobs()
        myfields = myms.get_summary()
    """

    def __init__(self, filename):
        self.filename = filename
        self.ms = xds_from_ms(filename, columns=['DATA', 'TIME', 'OBSERVATION_ID', 'ARRAY_ID'], group_cols=["DATA_DESC_ID"], table_keywords=True, column_keywords=True)
        #self.ms = xds_from_ms(filename, table_keywords=True, column_keywords=True, group_cols=["DATA_DESC_ID"]) # Changed to above for split-by-spw data

        self.ds_ms = self.ms[0]
        self.table_attr = self.ms[1]
        self.col_attr = self.ms[2]

        self.dd = xds_from_table(f"{filename}::DATA_DESCRIPTION")
        self.spw = xds_from_table(f"{filename}::SPECTRAL_WINDOW", group_cols="__row__", table_keywords=True, column_keywords=True)
        self.pol = xds_from_table(f"{filename}::POLARIZATION",  table_keywords=True, column_keywords=True)
        self.fields = xds_from_table(f"{filename}::FIELD", column_keywords=True)
        self.state = xds_from_table(f"{filename}::STATE",  table_keywords=True, column_keywords=True)
        try:
            self.sources = xds_from_table(f"{filename}::SOURCE", group_cols=['__row__'], columns=['SOURCE_ID', 'NAME', 'SPECTRAL_WINDOW_ID', 'REST_FREQUENCY', 'SYSVEL'])
        except:
            self.sources = xds_from_table(f"{filename}::SOURCE", group_cols=['__row__'], columns=['SOURCE_ID', 'NAME', 'SPECTRAL_WINDOW_ID'])

        self.ds_pol = self.pol[0]
        self.ds_spw = self.spw[0]
        self.ds_state = self.state[0][0]

        self.dashlin2 = dashlin2 = '='*80
        self.document = {}

    def get_summary(self, return_data=False):
        """ Print all metadata for the measurement sent to the standard output.

        Parameters
        ----------
        return_data : Boolean
            If true then return a dictionary containing the metadata from all
            the methods in the

        Returns
        -------
        document : dictionary
            A dictionary of lists, each list contains metadata returned by one
            method of the class.  The keys in the dictionary define the
            information in the dictionary.
        """
        _ = self.get_ms_info()
        print(f"{self.dashlin2}\n           MeasurementSet Name:  {self.document['table_name']}      MS Version {self.document['ms_version']}\n{self.dashlin2}\n")

        _ = self.get_observation_info()
        print(f"   Observer: {self.document['observer']}     Project: {self.document['project']}  ")
        print(f"Observation: {self.document['telescope_name']}")

        _ = self.get_main_table()
        print(f"Data records: {self.document['nrows']}       Total elapsed time = {self.document['exposetime']:.2f} seconds\n   Observed from   {self.document['obstime'][0][0]}/{self.document['obstime'][0][1]}   to   {self.document['obstime'][1][0]}/{self.document['obstime'][1][1]} ({self.document['timeref']})\n" )
        print(f"   ObservationID = {self.document['obsid']}         ArrayID = {self.document['arrid']}")
        _ = self.get_scan_list()

        _ = self.get_fields()

        _ = self.get_spectral_window()

        _ = self.get_source_list()

        _ = self.get_antenna_list()
        if return_data:
            return self.document

    def get_ms_info(self):
        ms_version = self.table_attr['MS_VERSION']
        table_name = self.filename
        self.document['ms_version'] = ms_version
        self.document['table_name'] = table_name
        return (ms_version, table_name)

    def get_observation_info(self):
        ds_obs = xds_from_table(f"{self.filename}::OBSERVATION")
        observer = ds_obs[0].OBSERVER.data.compute()[0]
        project = ds_obs[0].PROJECT.data.compute()[0]
        tn = ds_obs[0].TELESCOPE_NAME.data.compute()[0]
        self.document['observer'] = observer
        self.document['telescope_name'] = tn
        self.document['project'] = project
        return observer, project, tn

    def get_main_table(self):
        nrows, exposetimes, obstimes, timerefs, obsids, arrids = [], [], [], [], [], []
        for msds in self.ds_ms:
            nrow = len(msds.row)
            times_ = [float(msds.TIME[0].compute()), float(msds.TIME[-1].compute())]
            times = [Time(times_[0]/86400.0, format='mjd', scale='utc').datetime, Time(times_[-1]/86400.0, format='mjd', scale='utc').datetime]
            times = [[times[0].strftime('%d-%b-%Y'), times[0].strftime('%H:%M:%S')], [times[1].strftime('%d-%b-%Y'), times[1].strftime('%H:%M:%S')]]
            exposetime = times_[1]-times_[0]
            timeref = self.col_attr['TIME']['MEASINFO']['Ref']
            obsid = msds.OBSERVATION_ID[-1].data.compute()
            arrid = msds.ARRAY_ID[-1].data.compute()
            self.document['nrows'] = nrow
            self.document['exposetime'] = exposetime
            self.document['obstime'] = times
            self.document['timeref'] = timeref
            self.document['obsid'] = obsid
            self.document['arrid'] = arrid
            return (nrows, exposetimes, obstimes, timerefs)

    def get_scan_list(self, verbose=True):
        scans = xds_from_ms(self.filename, group_cols=['SCAN_NUMBER', "FIELD_ID"], index_cols=["SCAN_NUMBER", "TIME"])

        scan_list = []
        for scan in scans:
            scan_dict = {}
            times_ = [float(scan.TIME[0].compute()), float(scan.TIME[-1].compute())]
            times = [Time(times_[0]/86400.0, format='mjd', scale='utc').datetime, Time(times_[-1]/86400.0, format='mjd', scale='utc').datetime]
            scan_dict['times'] = [[times[0].strftime('%d-%b-%Y'), times[0].strftime('%H:%M:%S')], [times[1].strftime('%d-%b-%Y'), times[1].strftime('%H:%M:%S')]]
            scan_dict['scan_id'] = scan.attrs['SCAN_NUMBER']
            field_id = scan.attrs['FIELD_ID']
            scan_dict['field_id'] = field_id
            scan_dict['field_name'] = self.fields[0][0].to_dict()['data_vars']['NAME']['data'][field_id]
            scan_dict['nrows'] = scan.row.count().to_dict()['data']
            scan_dict['spwids'] = 0
            scan_dict['interval'] = round(scan.INTERVAL.data.compute()[-1])
            state_id = scan.STATE_ID[0].to_dict()['data']
            scan_dict['state_id'] = state_id
            scan_dict['intent'] = ''
            if state_id > -1:
                scan_dict['intent'] = self.ds_state.OBS_MODE.to_dict()['data'][state_id]
            scan_list.append(scan_dict)

        self.document['scan_list'] = scan_list
        if verbose:
            print("Date        Timerange (UTC)          Scan  FldId FieldName             nRows     SpwIds   Average Interval(s)    ScanIntent")
            for scan in scan_list:
                times = scan['times']
                if scan['scan_id'] == 1:
                    time_string = f"{times[0][0]}/{times[0][1]} - {times[1][1]}"
                else:
                    time_string = f"{times[0][1]} - {times[1][1]}"
                print(f"{time_string:>31}\t{scan['scan_id']:>9}{scan['field_id']:>7} {scan['field_name']:<22}{scan['nrows']:<10}{scan['spwids']:<9}{scan['interval']:<23}{scan['intent']:<10}")

            print("           (nRows = Total number of rows per scan) ")
        return scan_list


    def get_fields(self, verbose=True):
        """ Get the metadata for the fields in the measurement set.

        Parameters
        ----------
        verbose : Boolean
            If true then return a list of dicts, each one with metadata for one
            field.

        Returns
        -------
        antenna_list : list-like()
            A list of dictionary objects, each dictionary contains metadata for
            one field.
        """
        fields_ = self.fields[0][0].compute()
        fields = fields_.to_dict()

        code = fields['data_vars']['CODE']['data']
        fieldName = fields['data_vars']['NAME']['data']
        sourceID = fields['data_vars']['SOURCE_ID']['data']
        ref_dir = fields['data_vars']['REFERENCE_DIR']['data']
        epoch = self.fields[1]['REFERENCE_DIR']['MEASINFO']['Ref']

        nrow = np.shape(fieldName)[0]

        nfields = nrow
        fields_attrs = []

        group_cols = ["FIELD_ID"]
        ds_grouped = xds_from_table(self.filename, column_keywords=True, group_cols=group_cols)

        field_row_count = {dg.attrs['FIELD_ID']: dg.row.count().to_dict()['data'] for dg in ds_grouped[0]}

        for i in range(0,nrow):
            coords = SkyCoord(ra = ref_dir[i][0][0]*u.radian, dec = ref_dir[i][0][1]*u.radian )
            fields_attrs.append({'ID': i,
                                 'Code' : code[i],
                                 'Name' : fieldName[i],
                                 'RA' : str(int(coords.ra.hms[0]))+':'+str(int(coords.ra.hms[1]))+':'+'{0:.2f}'.format(coords.ra.hms[2]),
                                 'Decl' : deg_min_sec(coords.dec.degree),
                                 'Epoch' : epoch,
                                 'SrcId': sourceID[i],
                                 'nRows' : field_row_count[i]
                                })

        self.document['fields'] = fields_attrs

        if verbose:
            print(f"Fields: {len(fields_attrs)}")
            print("  ID   Code Name                RA               Decl           Epoch   SrcId      nRows")
            for field in fields_attrs:
                print(f"  {field['ID']:<4} {field['Code']:<4} {field['Name']:<19} {field['RA']:<15} {field['Decl']:<15} {field['Epoch']:<7} {field['SrcId']:<10} {field['nRows']}")

        return fields_attrs

    def get_spectral_window(self, verbose=True):

        print(f"Spectral Windows: ({len(self.ds_spw)} unique spectral windows and {len(self.ds_pol)} unique polarization setups)")
        print("  SpwID  Name   #Chans   Frame   Ch0(MHz)  ChanWid(kHz)  TotBW(kHz) CtrFreq(MHz)  Corrs")

        spw_attrs = []

        dd_spw_id = self.dd[0].SPECTRAL_WINDOW_ID.data.compute()
        dd_pol_id = self.dd[0].POLARIZATION_ID.data.compute()
        for msds in self.ds_ms:
            ddid = msds.attrs['DATA_DESC_ID']
            spw_id = dd_spw_id[ddid]
            spw_frame = self.spw[2]['CHAN_FREQ']['MEASINFO']['TabRefTypes'][self.ds_spw[spw_id].MEAS_FREQ_REF.data.compute()[0]]

            spw_name = self.ds_spw[spw_id].NAME.data.compute()[0]
            nchan = self.ds_spw[spw_id].NUM_CHAN.data.compute()[0]
            chan_zero = self.ds_spw[spw_id].CHAN_FREQ.data.compute()[0][0]/(10**6)
            chan_wid = self.ds_spw[spw_id].CHAN_WIDTH.data.compute()[0][0]/10**3
            total_BW = self.ds_spw[spw_id].TOTAL_BANDWIDTH.data.compute()[0]/10**3
            ctrfreq = (self.ds_spw[spw_id].CHAN_FREQ.data.compute()[0][0] + (self.ds_spw[spw_id].CHAN_FREQ.data.compute()[0][-1] - self.ds_spw[spw_id].CHAN_FREQ.data.compute()[0][0])/2)/10**6

            pol_id = dd_pol_id[ddid]
            corr_type = self.ds_pol[pol_id].CORR_TYPE.data.compute()[0]
            corr_type = [StokesTypes[i] for i in corr_type]

            corr_type_print = ' '.join(corr_type)
            spw_attrs.append({'id': spw_id, 'name': spw_name, 'nchan': nchan, 'spw_frame': spw_frame, 'chan_zero': chan_zero, 'chan_width': chan_wid, 'total_BW': total_BW, 'ctrfreq': ctrfreq, 'corr_type_': corr_type_print })
            print(f"  {spw_id}\t{spw_name}\t{nchan}\t{spw_frame}\t{chan_zero:.3f}\t{chan_wid:.3f}\t{total_BW}\t{ctrfreq:.4f}\t{corr_type_print}")

        return spw_attrs

    def get_source_list(self, verbose=True):
        ds_source = self.sources[0]
        sources = ds_source.to_dict()

        source_id = sources['data_vars']['SOURCE_ID']['data']
        names = sources['data_vars']['NAME']['data']
        spw_id = sources['data_vars']['SPECTRAL_WINDOW_ID']['data']
        try:
            rest_freq = sources['data_vars']['REST_FREQUENCY']['data']
            rest_freq = [r_[0] for r_ in rest_freq]
            sysvel = sources['data_vars']['SYSVEL']['data']
            sysvel = [r_[0] for r_ in sysvel]
        except:
            rest_freq = [0]*len(source_id)
            sysvel = [0]*len(source_id)

        source_list = [{'id': source_id[i], 'name': names[i], 'spw_id': spw_id[i], 'rest_freq': rest_freq[i], 'sysvel': sysvel[i]} for i in range(0, len(source_id))]
        self.document['sources'] = source_list

        if verbose:
            print(f"Sources: {len(sources)}")
            print("  ID   Name                SpwId RestFreq(MHz)  SysVel(km/s) ")
            for source in source_list:
                print(f"  {source['id']:<4} {source['name']:<19} {source['spw_id']:<5} {source['rest_freq']/10**6:<14.0f} {source['sysvel']/10**3:<13}")

        return source_list


    def get_antenna_list(self, verbose=True):
        """ Get the metadata for the antennas used in the measurement set.

        Parameters
        ----------
        verbose : Boolean
            If true then return a list of dicts, each one with metadata for one
            antenna.

        Returns
        -------
        antenna_list : list-like()
            A list of dictionary objects, each dictionary contains metadata for
            one antenna.
        """
        antenna = xds_from_table(f"{self.filename}ANTENNA",  table_keywords=True, column_keywords=True)
        ds_ant = antenna[0][0]
        ant = ds_ant.to_dict()

        antenna_ids = ant['coords']['ROWID']['data']
        names = ant['data_vars']['NAME']['data']
        stations = ant['data_vars']['STATION']['data']
        diameters = ant['data_vars']['DISH_DIAMETER']['data']
        offsets = ant['data_vars']['OFFSET']['data']
        itrf_coords = ant['data_vars']['POSITION']['data']

        ant_coords = []
        for pos in itrf_coords:
            geodetic = EarthLocation.from_geocentric(pos[0], pos[1], pos[2], unit=u.m).geodetic
            ant_coords.append([deg_min_sec(geodetic[0].value), deg_min_sec(geodetic[1].value), geodetic[2].value])

        self.document['antennas'] = [{'id': antenna_ids[i], 'name': names[i], 'station': stations[i], 'diameter': diameters[i], 'coordinates': ant_coords[i], 'offset': offsets[i], 'irtf': itrf_coords[i]} for i in range(0, len(antenna_ids))]

        if verbose:
            print(f"Antennas: {len(self.document['antennas'])}")
            print(f"  ID   Name  Station   Diam.    Long.         Lat.                Offset from array center (m)                ITRF Geocentric coordinates (m)        s")
            print(f"                                                                     East         North     Elevation               x               y               z")
            for antenna in self.document['antennas']:
                print(f"  {antenna['id']:<4} {antenna['name']:<5} {antenna['station']:<9} {antenna['diameter']:<8} {antenna['coordinates'][0]:<13} {antenna['coordinates'][1]:<22} {antenna['offset'][0]:<12} {antenna['offset'][1]:<9} {antenna['coordinates'][2]:<18.2f} {antenna['irtf'][0]:<15.1f} {antenna['irtf'][1]:<1.1f} {antenna['irtf'][2]:<1.1f}")

        return self.document['antennas']
