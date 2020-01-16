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
from astropy.coordinates import SkyCoord
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
    
    Example:
        myms = listobs(filename)
        myms.listobs()
        myfields = myms.get_summary()

    """

    def __init__(self, filename):
        self.filename = filename
        self.ms = xds_from_ms(filename, table_keywords=True, column_keywords=True, group_cols=["DATA_DESC_ID"])

        self.ds_ms = self.ms[0]
        self.table_attr = self.ms[1]
        self.col_attr = self.ms[2]

        self.dd = xds_from_table(f"{filename}/DATA_DESCRIPTION")
        self.spw = xds_from_table(f"{filename}SPECTRAL_WINDOW", group_cols="__row__", table_keywords=True, column_keywords=True)
        self.pol = xds_from_table(f"{filename}POLARIZATION",  table_keywords=True, column_keywords=True)
        self.fields = xds_from_table(f"{filename}/FIELD", column_keywords=True)
        self.state = xds_from_table(f"{filename}STATE",  table_keywords=True, column_keywords=True)
        self.ds_pol = self.pol[0]
        self.ds_spw = self.spw[0]
        self.ds_state = self.state[0][0]
        
        self.dashlin2 = dashlin2 = '='*80
        self.document = {}
        
    def get_summary(self):
        _ = self.get_ms_info()
        print(f"{self.dashlin2}\n           MeasurementSet Name:  {self.document['table_name']}      MS Version {self.document['ms_version']}\n{self.dashlin2}\n")

        _ = self.get_observation_info()
        print(f"   Observer: {self.document['observer']}     Project: {self.document['project']}  ")
        print(f"Observation: {self.document['telescope_name']}")        

        _ = self.get_main_table()
        print(f"Data records: {self.document['nrows']}       Total elapsed time = {self.document['exposetime']:.2f} seconds\n   Observed from   {self.document['obstime'][0][0]}/{self.document['obstime'][0][1]}   to   {self.document['obstime'][1][0]}/{self.document['obstime'][1][1]} ({self.document['timeref']})\n" )
        print(f"   ObservationID = {self.document['obsid']}         ArrayID = {self.document['arrid']}")
        self.get_scan_list()

        _ = self.get_fields()
        fields_attrs = self.document['fields']
        print(f"Fields: {len(fields_attrs)}")
        print("  ID   Code Name                RA               Decl           Epoch   SrcId      nRows")
        for field in fields_attrs:
            print(f"  {field['ID']:<4} {field['Code']:<4} {field['Name']:<19} {field['RA']:<15} {field['Decl']:<15} {field['Epoch']:<7} {field['SrcId']:<10} {field['nRows']}")

        self.get_spectral_window()

        _ = self.get_source_list()
        print(f"Sources: {len(self.document['source_id'])}")
        print("  ID   Name                SpwId RestFreq(MHz)  SysVel(km/s) ")
        for i in range(len(self.document['source_id'])):
            print(f"  {self.document['source_id'][i]:<4} {self.document['source_names'][i]:<19} {self.document['spw_id'][i]:<5} {self.document['rest_freq'][i]/10**6:<14.0f} {self.document['sysvel'][i]/10**3:<13}")
                  
                  
    def get_ms_info(self):
        ms_version = self.table_attr['MS_VERSION']
        table_name = self.filename
        self.document['ms_version'] = ms_version
        self.document['table_name'] = table_name
        return (ms_version, table_name)
    
    def get_observation_info(self):
        ds_obs = xds_from_table(f"{self.filename}/OBSERVATION")
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
    
    def get_scan_list(self):
        scans = xds_from_ms(self.filename, group_cols=['SCAN_NUMBER', "FIELD_ID", "DATA_DESC_ID"], index_cols=["SCAN_NUMBER", "TIME"])   
        print("Date        Timerange (UTC)          Scan  FldId FieldName             nRows     SpwIds   Average Interval(s)    ScanIntent")
        for scan in scans:
            times_ = [float(scan.TIME[0].compute()), float(scan.TIME[-1].compute())]
            times = [Time(times_[0]/86400.0, format='mjd', scale='utc').datetime, Time(times_[-1]/86400.0, format='mjd', scale='utc').datetime]
            times = [[times[0].strftime('%d-%b-%Y'), times[0].strftime('%H:%M:%S')], [times[1].strftime('%d-%b-%Y'), times[1].strftime('%H:%M:%S')]]
            scan_id = scan.attrs['SCAN_NUMBER']
            field_id = scan.attrs['FIELD_ID']
            field_name = self.fields[0][0].to_dict()['data_vars']['NAME']['data'][field_id]
            nrows = scan.row.count().to_dict()['data']
            spwids = 0
            interval = round(scan.INTERVAL.data.compute()[-1])
            state_id = scan.STATE_ID[0].to_dict()['data']
            intent = self.ds_state.OBS_MODE.to_dict()['data'][state_id]
            if scan_id == 1:
                time_string = f"{times[0][0]}/{times[0][1]} - {times[1][1]}"
            else:
                time_string = f"{times[0][1]} - {times[1][1]}"
            print(f"{time_string:>31}\t{scan_id:>9}{field_id:>7} {field_name:<22}{nrows:<10}{spwids:<9}{interval:<23}{intent:<10}")
        print("           (nRows = Total number of rows per scan) ")

    def get_fields(self):
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
        return fields_attrs
                  
    def get_spectral_window(self):
                
        print(f"Spectral Windows: ({len(self.ds_spw)} unique spectral windows and {len(self.ds_pol)} unique polarization setups)")
        print("  SpwID  Name   #Chans   Frame   Ch0(MHz)  ChanWid(kHz)  TotBW(kHz) CtrFreq(MHz)  Corrs")

        spw_attrs = []
                  
        for msds in self.ds_ms:
            ddid = msds.attrs['DATA_DESC_ID']
            spw_id = self.dd[ddid].SPECTRAL_WINDOW_ID.data.compute()[0]
            spw_frame = self.spw[2]['CHAN_FREQ']['MEASINFO']['TabRefTypes'][self.ds_spw[spw_id].MEAS_FREQ_REF.data.compute()[0]]
                  
            spw_name = self.ds_spw[spw_id].NAME.data.compute()[0]
            nchan = self.ds_spw[spw_id].NUM_CHAN.data.compute()[0]
            chan_zero = self.ds_spw[spw_id].CHAN_FREQ.data.compute()[0][0]/(10**6)
            chan_wid = self.ds_spw[spw_id].CHAN_WIDTH.data.compute()[0][0]/10**3
            total_BW = self.ds_spw[spw_id].TOTAL_BANDWIDTH.data.compute()[0]/10**3
            ctrfreq = (self.ds_spw[spw_id].CHAN_FREQ.data.compute()[0][0] + (self.ds_spw[spw_id].CHAN_FREQ.data.compute()[0][-1] - self.ds_spw[spw_id].CHAN_FREQ.data.compute()[0][0])/2)/10**6

            pol_id = self.dd[ddid].POLARIZATION_ID.data.compute()[0]
            corr_type = self.ds_pol[pol_id].CORR_TYPE.data.compute()[0]
            corr_type = [StokesTypes[i] for i in corr_type]

            corr_type_print = ' '.join(corr_type)
            print(f"  {spw_id}\t{spw_name}\t{nchan}\t{spw_frame}\t{chan_zero:.3f}\t{chan_wid:.3f}\t{total_BW}\t{ctrfreq:.4f}\t{corr_type_print}") 
        
                  
    def get_source_list(self):
        sources = xds_from_table(f"{self.filename}/SOURCE", column_keywords=True)
        ds_source = sources[0]
        for source in ds_source:
            source = source.to_dict()
            source_id = source['data_vars']['SOURCE_ID']['data']
            names = source['data_vars']['NAME']['data']
            spw_id = source['data_vars']['SPECTRAL_WINDOW_ID']['data']
            rest_freq = np.squeeze(source['data_vars']['REST_FREQUENCY']['data'])
            sysvel = np.squeeze(source['data_vars']['SYSVEL']['data'])
                  
            self.document['source_id'], self.document['source_names'], self.document['spw_id'], self.document['rest_freq'], self.document['sysvel'] = source_id, names, spw_id, rest_freq, sysvel
        return {'source_id':source_id, 'source_names':names, 'spw_id':spw_id, 'rest_freq':rest_freq, 'sysvel':sysvel}

