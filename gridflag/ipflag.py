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
@click.argument('field_id', type=int)
@click.option('--uvrange', nargs=2, type=float, default=[None,None],
              help='Min and max UV range within which to flag (default:all)')
@click.option('--nsigma', type=float, default=3.0,
        help='Sigma to use while determining outliers [default: 3]')
@click.option('--column', 'datacolumn', default='DATA',
        help='Column on which to operate [default: DATA]')
def gridflag(ms, field_id, nsigma, uvrange, datacolumn):
    """
    Run the GRIDflag flagger on the input MS, over the specified FIELD_ID.

    The UV range to be specified determines the minimum and maximum
    UV lengths considered while flagging, anything outside this range is
    ignored.
    """

    _gridflag(ms, field_id, nsigma, uvrange, datacolumn)


def _gridflag(ms, field_id, nsigma, uvrange, datacolumn):
    """
    The 'internal' layer of gridflag - so that it's accessible
    via notebook/ipython without going through the click wrappers.
    """

    import dask.array as da
    import dask

    # Depending on how it was installed one or the other will work.
    try:
        import compute_uv_bins, groupby_apply
    except ModuleNotFoundError:
        from ipflag import compute_uv_bins
        from ipflag import groupby_apply

    #from dask.distributed import Client
    #client = Client()

    ds_ind = compute_uv_bins.load_ms_file(ms, field_id, datacolumn=datacolumn)[0]

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
