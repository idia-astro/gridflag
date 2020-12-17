import numpy as np
import os
import time

import dask
import dask.array as da

from distributed import Client
from dask_jobqueue import SLURMCluster

from ipflag import compute_uv_bins, gridflag

# Job parameters (to be set by pipeline manager or command line arguments)
msfile = '/scratch/users/jbochenek/data/1538856059_sdp_l0_chan100_flagdata.ms/'
data_columns = [0, 3]
fieldid = 1

n_workers = 1

# Set up distributed scheduler and slurm parameters

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
    local_directory=f'/scratch/users/{os.environ["USER"]}/temp/',
    death_timeout="1m",
    log_directory=f'/scratch/users/{os.environ["USER"]}/logs/',
    project="b03-idia-ag",
    python="singularity exec /users/jbochenek/containers/astro_ipflag_tools.simg /usr/bin/python3", 
    job_extra=["--exclude=compute-047"]
)

cluster.scale(jobs=n_workers)

time.sleep(30)

client = Client(cluster)

ds_bindex, uvbins = compute_uv_bins.load_ms_file(msfile, fieldid=fieldid, bin_count_factor=1.0, chunksize=10**8)
ds_ind = ds_bindex[0]

flag_list, median_grid = gridflag.map_grid_partition(ds_ind, data_columns, uvbins, sigma=2.5, partition_level=4, client=client)

flag_vis_percentage = len(flag_list)/len(ds_ind.DATA)
print(f"Percentage of bins flagged: {100*flag_vis_percentage:.2f}% - {len(flag_list)}/{len(ds_ind.DATA)} visibilities")

compute_uv_bins.write_ms_file(msfile, ds_ind, flag_list, fieldid, data_columns, overwrite=True, client=client)

client.close()
cluster.close()