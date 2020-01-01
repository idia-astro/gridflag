#! /usr/bin/env python3

import click

ctx = dict(help_option_names=['-h', '--help'])

@click.group(context_settings = ctx)
def main():
    """
    Usage:

    ipflag gridflag --help
    """

@main.command(short_help='GRIDflag algorithm')
@click.argument('ms', type=click.Path(exists=True))
@click.argument('gridsize', type=int)
@click.option('--uvrange', nargs=2, type=float, default=[None,None],
              help='Min and max UV range within which to flag (default:all)')
@click.option('--corr', type=str,
              help='Correlation to flag on (ex: "XX") (default: all)')
@click.option('--nsigma', type=float, default=3.0,
        help='Sigma to use while determining outliers [default: 3]')
def gridflag(ms, gridsize, nsigma, uvrange, corr):
    """
    Run the GRIDflag flagger on the input MS.

    The gridsize parameter is given by the inverse of the maximum field of
    view expressed in radians. If the field of view is 3 deg (~ 0.05 rad)
    the gridsize is given by ~ 20 lambda.

    The UV range to be specified determines the minimum and maximum
    UV lengths considered while flagging, anything outside this range is
    ignored.

    By default the GRIDflag algorithm flags on all correlations independently,
    successively on the real, imaginary and amplitudes of the visibilities. If
    a correlation is specified via the --corr, only that correlation will be
    operated upon.
    """

    import dask.array as da
    import dask
    #from ipflag.compute_uv_bins import load_ms_file
    import compute_uv_bins
    import groupby_apply

    from dask.distributed import Client
    client = Client()

    ds_ind = compute_uv_bins.load_ms_file(ms)

    # Get dask arrays of UV-bins and visibilities from XArray dataset
    dd_ubins = da.from_array(ds_ind.U_bins)
    dd_vbins = da.from_array(ds_ind.V_bins)

    dd_vals = da.from_array(ds_ind.DATA[:,0])

    # Combine U and V bins into one dask array
    dd_bins = da.stack([dd_ubins, dd_vbins]).T

    # Apply unifrom chunks to both dask arrays
    chunk_size = 1E7
    dd_bins = dd_bins.rechunk([chunk_size, 2])
    dd_vals = dd_vals.rechunk([chunk_size, 1])

    value_group_chunks = [
        dask.delayed(groupby_apply.group_bin_values_wrap)(part[0][0], part[1])
        for part in zip(dd_bins, dd_vals)
    ]
    value_groups_ = \
        dask.delayed(groupby_apply.combine_group_values)(value_group_chunks)

    median_bins = \
        dask.delayed(groupby_apply.apply_to_groups)(value_groups_, np.median)

    std_bins = \
        dask.delayed(groupby_apply.apply_to_groups)(value_groups_, np.std)

if __name__ == '__main__':
    main()
