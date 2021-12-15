# IPFlag
Repository for the reference implementation of the ipflag software for Radio Frequency Interference (RFI). The package is implemented in Python using the XArray extension Dask-MS [^3] as well as 

Dask 

The algorithm is based on the [UV-plane flagging algorithm][1]



# Installation 
The software is and dependencies are be installed using avaialalbe in configured 

.. code-block:: bash
	$ pip3 install --upgrade git+https://github.com/idia-astro/ipflag.git


# Referneces 

[1]: Two Procedures to Flag Radio Frequency Intereference in the UV Plane 
[https://arxiv.org/abs/1711.00128](https://arxiv.org/abs/1711.00128)
[2]: Xarray N-D labeled arrays and datasets in Python [https://xarray.pydata.org/en/stable/](https://xarray.pydata.org/en/stable/)
[3]: XArray datasets from CASA tables [https://github.com/ska-sa/dask-ms](https://github.com/ska-sa/dask-ms)
[4]: Flexible library for parallel computing in Python, Dask delayed [https://docs.dask.org/en/stable/delayed.html](https://docs.dask.org/en/stable/delayed.html)
