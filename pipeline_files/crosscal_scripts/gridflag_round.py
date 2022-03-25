#!/usr/bin/env python3.7

import numpy as np
import os
import time

import dask
import dask.array as da

from distributed import Client
from dask_jobqueue import SLURMCluster

from ipflag import compute_uv_bins, gridflag, listobs_daskms, annulus_stats, plotgrid

from config_parser import validate_args as va
import logging

import bookkeeping

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)-15s %(levelname)s: %(message)s", level=logging.INFO)


def map_field_name_id(fieldname, msfile):
    """Convert field name to field ID.
    
    Parameters
    ----------
    fieldname : string
    
    Returns
    -------
    fieldid : int
    """
    
    lo = listobs_daskms.ListObs(msfile, 'DATA')
    field_info = lo.get_fields()
    
    fnames = {ifield['Name']:ifield['ID'] for ifield in field_info}
    try:
        fid = fnames[fieldname]
    except:
        raise ValueError(f"The field name value of {fieldname} is not a valid field in this dataset.")
    
    return fid


def run_gridflag(visname, fields):

    stokes = 'I'
    n_workers = 1
    username = os.environ["USER"]

    logger.info('GridFlag: Setting up Dask Cluster with {0} workers.'.format(n_workers))

    # Cluster configuration should be moved to ~/.config/dask/jobqueue.yaml
    cluster = SLURMCluster(
        queue="Main",
        cores=16,
        processes=1,
        n_workers=n_workers,
        memory="114GB",
        interface="ens3",
        shebang='#!/usr/bin/env bash',
        walltime="02:00:00",
        local_directory="temp/",
        death_timeout="1m",
        log_directory="logs/",
        project="b03-idia-ag",
        python="singularity exec /users/jbochenek/containers/astro_tools.simg /usr/bin/python3"
    )

    cluster.scale(jobs=n_workers)

    time.sleep(30)

    client = Client(cluster)

    start = time.time()

    fieldname = fields.targetfield
    fieldid = map_field_name_id(fieldname, visname)


    logger.info('Reading measurement set in dask-ms: {0}.'.format(visname))

    ds_ind, uvbins = compute_uv_bins.load_ms_file(visname, 
                                                  fieldid=fieldid, 
                                                  bin_count_factor=1.0, 
                                                  chunksize=10**7
                                                 )

    logger.info('Flagging field {0} ({1}).'.format(fieldid, fieldname, visname))
    # Check existing flags in MS
    # check_exising_flags(ds_ind, stokes=stokes, client=client)

    flag_list, median_grid, median_grid_flg = gridflag.compute_ipflag_grid(ds_ind, 
                                                                           uvbins, 
                                                                           sigma=3.0, 
                                                                           partition_level=3, 
                                                                           stokes=stokes, 
                                                                           client=client
                                                                           )


    flag_vis_percentage = 100*len(flag_list)/len(ds_ind.DATA)
    print("Percentage of rows flagged: {:.1f}% - {}/{} visibilities".format(flag_vis_percentage, len(flag_list), len(ds_ind.DATA)))

    logger.info("Percentage of rows flagged {0:.2f} % conprising {1}/{2} visibilities.".format(flag_vis_percentage, len(flag_list), len(ds_ind.DATA)))

    # Save UV-grid median plot
    annulus_width = annulus_stats.compute_annulus_bins(median_grid, uvbins, 10)
    plotgrid.plot_uv_grid(median_grid, uvbins, annulus_width, filename="uv_grid_unflagged.png")
    plotgrid.plot_uv_grid(median_grid_flg, uvbins, annulus_width, filename="uv_grid_flagged.png")

    compute_uv_bins.write_ms_file(visname, ds_ind, flag_list, fieldid, stokes=stokes, overwrite=True)

    end = time.time()

    print("GridFlag runtime: {} seconds.".format(end-start))
    logger.info("Flagging completed. Runtime: {0} seconds.".format((end-start)))

    client.close()
    cluster.close()


def main(args, taskvals):

    visname = va(taskvals, 'data', 'vis', str)
    
    fields = bookkeeping.get_field_ids(taskvals['fields'])

    run_gridflag(visname, fields)



if __name__ == '__main__':

    bookkeeping.run_script(main)
