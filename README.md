# IPFlag
An algorithm for the identification of Radio Frequency Interference (RFI) in the UV Plane. This repository contains the IPflag software package which implements the Gridflag algorithm as described in the paper [Two Procedures to Flag Radio Frequency Interference in the UV Plane](https://arxiv.org/abs/1711.00128). The package comprises a statistical method for identifying RFI as well as a tool for running Gridflag in a distributed manner on the Slurm workload manager. 

# Description
The GridFlag package is implemented in Python. The data is imported from CASA [MeasurementSet](https://casadocs.readthedocs.io/en/latest/notebooks/casa-fundamentals.html#MeasurementSet-v2) (MS) or multi-measurement set (MMS) files, which contains data sorted according to the physical observation order. The visibilities in the MS files are transformed and UV bins are calculated using the [Dask-MS](https://github.com/ska-sa/dask-ms) tool. Pre-processing operations are constructed in a DAG using Dask delayed and computed in memory in chunks. Metadata and variables in the resulting N-dimensional datasets are accounted for with [XArray](https://xarray.pydata.org/en/stable/). After UV bins are computed for each visibility, the task graph for all operations is executed in parallel, then the unsorted, chunked binned data is partitioned recursively in to blocks of U and V bins, which can then be further processed in a distributed manner. In the final steps, we compute UV grid and bin statistics which are taken by our flagging logic to compute flags for each visibility in each bin. The flags are then transformed in reverse to the order of the initial MS file, and the FLAG column(s) are written to the MS file.

All numerically intensive code, which consist of the UV-partition and certain statistical calculations, are compiled using [Numba](https://numba.pydata.org) to produce optimized machine level code. 


# Installation 
The software is and dependencies can be installed using the github repository:

```bash
	$ pip3 install --upgrade git+https://github.com/idia-astro/ipflag.git
```

For those using the Ilifu cluster, IPFlag is installed in a singularity container for use as a pipeline step or for stand-alone processing. It has also been integrated with the IDIA MeerKAT pipeline. A fork of the pipeline code is available at `/users/jbochenek/work/pipeline_gridflag/pipelines_casa6`, and can be use by running the corresponding setup script: 

```
source /users/jbochenek/work/pipeline_gridflag/pipelines_casa6/pipelines/setup.sh
```

After setting up the pipeline for your data file, add a gridflag step to the pipeline by editing the automatically generated config script and adding the following to the `scripts` list:

```
('gridflag_round.py', False, '/users/jbochenek/containers/astro_tools.simg')
```

This will run the 


# Usage
The GridFlag algorithm can be used both interactively, for example in Jupyter, and in batch mode. It is also integrated with the [The IDIA MeerKAT pipeline](https://idia-pipelines.github.io/docs/processMeerKAT/Quick-Start/) as a flagging step. 


The RFI mitigation algorithm takes a single parameter to control the threshold for discriminating visibilities. This parameter is the significance or α, which is the probability for rejecting non-RFI measurements, according to the GridFlag model. A value of α = 3 means a row will be flagged if it is at or above the threshold in its UV cell for which 95% of non-RFI measurements are distributed. This value is computed according to the sample size in the cell.  

To test and calibrate the flagging performance, GridFlag can be run in a Jupyter notebook, making it easier to investigate and visualize the results. An demonstration notebook is available in the repository, 

After running the flagging algorithm on a set of code, a set of visualizations are automatically generated. 





# Referneces 

1. : Two Procedures to Flag Radio Frequency Intereference in the UV Plane 
[https://arxiv.org/abs/1711.00128](https://arxiv.org/abs/1711.00128)
2. : Xarray N-D labeled arrays and datasets in Python [https://xarray.pydata.org/en/stable/](https://xarray.pydata.org/en/stable/)
3. : XArray datasets from CASA tables [https://github.com/ska-sa/dask-ms](https://github.com/ska-sa/dask-ms)
4. : Flexible library for parallel computing in Python, Dask delayed [https://docs.dask.org/en/stable/delayed.html](https://docs.dask.org/en/stable/delayed.html)\
5. : Numba: A High Performance Python Compiler [https://numba.pydata.org](https://numba.pydata.org)