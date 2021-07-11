#!/usr/bin/env python3.7

import numpy as np
import os
import time

import dask
import dask.array as da

from distributed import Client
from dask_jobqueue import SLURMCluster

from ipflag import compute_uv_bins, gridflag

from config_parser import validate_args as va


def run_gridflag(visname, fields)

    stokes = 'I'
    n_workers = 1

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
        local_directory=f'/scratch3/users/{os.environ["USER"]}/temp/',
        death_timeout="1m",
        log_directory=f'/scratch3/users/{os.environ["USER"]}/logs/',
        project="b03-idia-ag",
        python="singularity exec /users/jbochenek/containers/astro_tools.simg /usr/bin/python3"
    )

    cluster.scale(jobs=n_workers)

    time.sleep(30)

    client = Client(cluster)

    start = time.time()

    fieldid = fields.targetfield

    ds_ind, uvbins = compute_uv_bins.load_ms_file(visname, 
                                                  fieldid=fieldid, 
                                                  bin_count_factor=1.0, 
                                                  chunksize=10**7
                                                 )

    # Check existing flags in MS
    # check_exising_flags(ds_ind, stokes=stokes, client=client)

    flag_list, median_grid, median_grid_flg = gridflag.compute_ipflag_grid(ds_ind, 
                                                                           uvbins, 
                                                                           sigma=2.5, 
                                                                           partition_level=3, 
                                                                           stokes=stokes, 
                                                                           client=client
                                                                           )

    flag_vis_percentage = len(flag_list)/len(ds_ind.DATA)
    print(f"Percentage of rows flagged: {100*flag_vis_percentage:.2f}% - {len(flag_list)}/{len(ds_ind.DATA)} visibilities")


    # Save UV-grid median plot
    annulus_width = annulus_stats.compute_annulus_bins(median_grid, uvbins, 10)
    plotgrid.plot_uv_grid(median_grid, uvbins, annulus_width, filename="uv_grid_unflagged.png")
    plotgrid.plot_uv_grid(median_grid_flg, uvbins, annulus_width, filename="uv_grid_flagged.png")

    compute_uv_bins.write_ms_file(visname, ds_ind, flag_list, fieldid, stokes=stokes, overwrite=True)

    end = time.time()

    print(f"GridFlag runtime: {end-start} seconds.")

    client.close()
    cluster.close()


def main(args,taskvals):

    visname = va(taskvals, 'data', 'vis', str)
    
    fields = bookkeeping.get_field_ids(taskvals['fields'])

    run_gridflag(visname, fields[0])

if __name__ == '__main__':

    bookkeeping.run_script(main)