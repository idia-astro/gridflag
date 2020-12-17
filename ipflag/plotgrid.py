import numpy as np

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Label
from bokeh.palettes import Spectral6, Spectral4
from bokeh.transform import linear_cmap
from bokeh.layouts import gridplot 
from bokeh.io import export_png

from bokeh.util.hex import hexbin

from scipy.stats import median_absolute_deviation, gamma, chi2, poisson, expon, lognorm, rice, rayleigh

import dask
import dask.array as da

from . import annulus_stats

from .annulus_stats import get_fixed_thresholds, get_rayleigh_thresholds


def plot_uv_grid(median_grid, uvbins, annulus_width, bin_max=None):

    uv_bins = np.asarray(median_grid>0).nonzero()
    (u, v) = uv_bins
    
    bin_median_values = median_grid[uv_bins]
#     bin_counts = bincount_grid[uv_bins]
    binwidth = [uvbins[0][1] - uvbins[0][0], uvbins[1][1] - uvbins[1][0]]
    
    x_bin_range = uvbins[0][u-1]
    y_bin_range = uvbins[1][v-1]

    # Compute color scale to show median values
    if bin_max==None:
        bin_max = np.median(bin_median_values) + 3.*median_absolute_deviation(bin_median_values)
    print(f"Grid median (of medians): {np.median(bin_median_values):.3f};  Range: median<{bin_max:0.3f}")
    
    value_colors = [
        "#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+200*bin_median_values/bin_max, 30+200*bin_median_values/bin_max)
    ]

#     value_colors = [
#         "#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(255*bin_median_values/bin_max, 255-255*bin_median_values/bin_max)
#     ]

    value_colors = np.array(value_colors)

    # Erase all bins with no data
    value_colors[np.where(bin_median_values==0)] = "#DADADA"

    # Black out all bins with "noise" (this is a naive criteria to detect RMS but it works approximately)
    value_colors[np.where(bin_median_values>bin_max)] = "#000000"

    data=dict(
        x_bin_range=x_bin_range,
        y_bin_range=y_bin_range,
        u_bins=u,
        v_bins=v,
        value_colors=value_colors,
        medians=bin_median_values
    )

    TOOLS = "hover,crosshair,pan,zoom_in,zoom_out,box_zoom,reset,save,"

    p = figure(title="UV-grid Median Surface",
               x_axis_label="U bins", y_axis_label="V bins",
               tooltips = [('UV bin', '(@u_bins, @v_bins)'), ('Median', '@medians{1.3f}')]
              )

    p.plot_width = 1000
    p.plot_height = 1000
    p.rect('x_bin_range', 'y_bin_range', width=binwidth[0], height=binwidth[1], source=data, color='value_colors', line_color=None)
    p.arc(x=[0]*len(annulus_width), y=[0]*len(annulus_width), radius=annulus_width, start_angle=0.0, end_angle=2*np.pi, color="navy", line_dash='dashed', line_width=0.5)
    show(p)
    return p, bin_max



def plot_rms_vs_dist(value_grids, names, uvbins, grid_row_map, n_annulus=20, title="RMS"):

    uv_bins = np.asarray(value_grids[0]>0).nonzero()
    uv_bins_ = np.array([uv_bins[0], uv_bins[1]])

    bin_uv_dist = np.sqrt(uvbins[0][uv_bins_[0]-1]**2 + uvbins[1][uv_bins_[1]-1]**2)
    
    TOOLS="hover,crosshair,pan,zoom_in,zoom_out,box_zoom,reset,tap,save"
    TOOLTIPS = [
        ("(u,v)", "(@u,@v)"),
        ("(x,y)", "($x{0.1f}, $y{0.1f})"),
        ("count", "@count")
    ]

    p = figure(title=f"UV Distance vs. {title}", tools=TOOLS, plot_width=1000, plot_height=400, tooltips=TOOLTIPS, x_axis_label="UV Distance", y_axis_label=f"{title} (Jy)")

    annulus_width = annulus_stats.compute_annulus_bins(value_grids[0], uvbins, n_annulus)

    colors = ['red', 'green', 'blue']

    for value_grid, color, name in zip(value_grids, Spectral4, names):
        bin_median_values = value_grid[uv_bins]
    
        for ind, edge in enumerate(annulus_width):
            minuv=0
            if ind:
                minuv = annulus_width[ind-1]
            maxuv = annulus_width[ind]
            ann_bin = bin_median_values[np.where((bin_uv_dist>minuv) & (bin_uv_dist<maxuv))]
            mean_median_amp = np.mean(ann_bin)
            mad_amp = median_absolute_deviation(ann_bin)
        
            p.line([minuv, maxuv], [mean_median_amp, mean_median_amp], line_width=2, color=color, alpha=0.8, 
                    muted_color=color, muted_alpha=0.2, legend_label=name)

            print(f'Annulus ({minuv:<6}- {maxuv:<6} lambda)\tcount: {len(ann_bin):<6}\tmedian: {mean_median_amp:.3f}\tmad: {mad_amp:.3f}')

    p.legend.location = "top_right"
    p.legend.click_policy="mute"

    show(p)
    


def plot_amp_dist_bins(median_grid, uvbins, grid_row_map, annulus_width=[], n=10000, title="Median Amplitude"):

    uv_bins = np.asarray(median_grid>0).nonzero()
    (u, v) = uv_bins
    uv_bins_ = np.array([uv_bins[0], uv_bins[1]])
    
    bin_median_values = median_grid[uv_bins]
    bin_uv_dist = np.sqrt(uvbins[0][uv_bins_[0]-1]**2 + uvbins[1][uv_bins_[1]-1]**2)

    # Compute bin counts
    bin_count_grid = np.zeros(((np.max(grid_row_map[:,0])+1), (np.max(grid_row_map[:,1])+1)), dtype=int)
    for c, (u_, v_) in zip(np.diff(grid_row_map[:,2]), grid_row_map[:,:2]):
        bin_count_grid[u_][v_] = c
    bincounts = np.array([bin_count_grid[u_-1][v_-1] for u_,v_ in zip(u,v)])

#     bincounts = np.array([len(value_groups[u_-1][v_-1]) for u_,v_ in zip(u,v)])
    
    # Take a random sample of visibilities for plotting
    x = np.random.choice(len(bin_uv_dist), n)
    x.sort()
    
    buv = bin_uv_dist[x]
    bmv = bin_median_values[x]
    bc = bincounts[x]
    u_bin = u[x]
    v_bin = v[x]

    source = ColumnDataSource(dict(x=buv,y=bmv, z=x, count=bc, u=u_bin, v=v_bin))
    uv_colors = linear_cmap(field_name='z', palette=Spectral6 ,low=min(x) ,high=max(x))
    
    TOOLS="hover,crosshair,pan,zoom_in,zoom_out,box_zoom,reset,tap,save"
    TOOLTIPS = [
        ("(u,v)", "(@u,@v)"),
        ("(x,y)", "($x{0.1f}, $y{0.1f})"),
        ("count", "@count")
    ]

    p = figure(title=f"UV Distance vs. {title}", tools=TOOLS, plot_width=1000, plot_height=400, tooltips=TOOLTIPS, x_axis_label="UV Distance", y_axis_label=f"{title} (Jy)")

    p.scatter(x='x', y='y', fill_color=uv_colors, fill_alpha=0.7, line_color=None, source=source)

    if len(annulus_width):
        for ind, edge in enumerate(annulus_width):
            minuv=0
            if ind:
                minuv = annulus_width[ind-1]
            maxuv = annulus_width[ind]
            ann_bin = bin_median_values[np.where((bin_uv_dist>minuv) & (bin_uv_dist<maxuv))]
            mean_median_amp = np.median(ann_bin)
            mad_amp = median_absolute_deviation(ann_bin)
            rlower, rupper = get_rayleigh_thresholds(ann_bin)
            p.line([minuv, maxuv], [mean_median_amp, mean_median_amp], line_color='black', line_dash='dashed', line_alpha=0.75)
            p.line([minuv, maxuv], [rlower, rlower], line_color='gray', line_dash='dashed', line_alpha=0.75)
            p.line([minuv, maxuv], [rupper, rupper], line_color='gray', line_dash='dashed', line_alpha=0.75)

            print(f'Annulus ({minuv:<6}- {maxuv:<6} lambda)\tcount: {len(ann_bin):<6}\tmedian: {mean_median_amp:.3f}\tmad: {mad_amp:.3f}')

    show(p)



def plot_rayleigh_fit(bin_values, func=rayleigh, nbins=50, hrange=None):

    bin_median = np.median(bin_values)
    bin_alpha = bin_median/np.sqrt(2*np.log(2))

    n, bins = np.histogram(bin_values, range=hrange, bins=nbins, density=False)
    scale = np.diff(bins)[0]*np.sum(n)

    bincenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])

    y = scale*func.pdf(bincenters, loc=0, scale=bin_alpha) #, 0.68, loc=0., scale=0.22         fscale=0.077  (3.5, -0.009277343750000453, 0.07) 34.183836340904236

    TOOLS="hover,crosshair,pan,box_zoom,reset,tap,save,box_select"
    p = figure(tools=TOOLS, plot_width=600, plot_height=400)

    p.quad(bottom=0, top=n, left=bins[:-1], right=bins[1:], fill_color='navy', alpha=0.7)

    #pdf function fit
    p.line(bincenters, y, line_color='green', line_dash='5', line_width=2)

    bin_count = len(bin_values)
    bin_mean, bin_med = np.mean(bin_values), np.median(bin_values)
    bin_mean_var, bin_med_var = np.std(bin_values), median_absolute_deviation(bin_values)# np.mean(abs(bin_values - bin_med)**2)
    bin_ratio = bin_mean / bin_med

    p.ray(x=[bin_med], y=[0], length=0, angle=[np.pi/2.], color='red', line_dash='dashed', line_alpha=0.75)

    count_text = Label(x=350, y=325, x_units='screen', y_units='screen', text=f"count: {bin_count}")
    mean_text = Label(x=350, y=300, x_units='screen', y_units='screen', text=f"mean: {bin_mean:.3f} ± {bin_mean_var:.3f}")
    med_text = Label(x=350, y=280, x_units='screen', y_units='screen', text=f"median: {bin_med:.3f} ± {bin_med_var:.3f}")
    ratio_text = Label(x=350, y=260, x_units='screen', y_units='screen', text=f"ratio: {bin_ratio:.2f}")

    p.add_layout(count_text)
    p.add_layout(mean_text)
    p.add_layout(med_text)
    p.add_layout(ratio_text)

    show(p)


def plot_bin_distribution(bin_values, func=None, title='', nbins=50, hrange=None):

#     bin_values = value_groups[uvbin[0]][uvbin[1]]
    params = []
    n, bins = np.histogram(bin_values, range=hrange, bins=nbins, density=False)
    scale = np.diff(bins)[0]*np.sum(n)

    TOOLS="hover,crosshair,pan,box_zoom,reset,tap,save,box_select"
    p = figure(title=f'{title}', tools=TOOLS, plot_width=600, plot_height=400)

    p.quad(bottom=0, top=n, left=bins[:-1], right=bins[1:], fill_color='navy', alpha=0.7)

    # Limit the range of values for the fit
    if hrange:
        bin_vals_hrange = bin_values[np.where((bin_values>hrange[0]) & (bin_values<hrange[1]))]
    else:
        bin_vals_hrange = bin_values

    if func:
        params = func.fit(bin_vals_hrange) # , floc=0, 0.68, loc=0., scale=0.22         fscale=0.077  (3.5, -0.009277343750000453, 0.07) 34.183836340904236
    #    params = lognorm.fit(bin_values, f0=0.4, loc=0., fscale=0.28)  # bin: (558, 358)
        fit_median = params[1]*np.sqrt(2*np.log(2))+params[0]
        print(f"Median from fit: {fit_median}")
        bincenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        y = scale*func.pdf(bincenters, *params)    
        print(params, scale)
        #pdf function fit
        p.line(bincenters, y, line_color='green', line_dash='5', line_width=2)

    bin_count = len(bin_values)
    bin_mean, bin_med = np.mean(bin_values), np.median(bin_values)
    bin_mean_var, bin_med_var = np.std(bin_values), median_absolute_deviation(bin_values)# np.mean(abs(bin_values - bin_med)**2)
    bin_ratio = bin_mean / bin_med

    p.ray(x=[bin_med], y=[0], length=0, angle=[np.pi/2.], color='red', line_dash='dashed', line_alpha=0.75)

    count_text = Label(x=350, y=325, x_units='screen', y_units='screen', text=f"count: {bin_count}")
    mean_text = Label(x=350, y=300, x_units='screen', y_units='screen', text=f"mean: {bin_mean:.3f} ± {bin_mean_var:.3f}")
    med_text = Label(x=350, y=280, x_units='screen', y_units='screen', text=f"median: {bin_med:.3f} ± {bin_med_var:.3f}")
    ratio_text = Label(x=350, y=260, x_units='screen', y_units='screen', text=f"ratio: {bin_ratio:.2f}")

    p.add_layout(count_text)
    p.add_layout(mean_text)
    p.add_layout(med_text)
    p.add_layout(ratio_text)

    show(p)
    return params


def make_hist_plot(binname, bin_values, nbins, yrange=None, tools=''):
    '''Make a histogram plot of one bin in the UV grid.
    
    Parameters
    ----------
    binname : string
        Name to print on sub-plot to identify bin.
    bin_values : list-like
        List of bin values for the bin.
    nbins : int
        Number of bins for the histogram
        
    Returns
    -------
    p : bokeh.plotting.figure.Figure
        Figure to be shown or included in multi-plot
    '''
    if len(bin_values)==0:
        return None
    
    n, bins = np.histogram(bin_values, range=yrange, density=False, bins=nbins)
    
    binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    
    scale = np.diff(bins)[0]*np.sum(n)

    params = rayleigh.fit(bin_values)
    y = scale*rayleigh.pdf(binscenters, *params)

    p = figure(title=f'{binname}', tools=tools, plot_width=600, plot_height=400)
    p.quad(bottom=0, top=n, left=bins[:-1], right=bins[1:], fill_color='navy', alpha=0.7)

    p.line(binscenters, y, line_color='green', line_dash='5', line_width=2)
    
    bin_count = len(bin_values)
    bin_mean, bin_med = np.mean(bin_values), np.median(bin_values)
    bin_mean_var, bin_med_var = np.std(bin_values), median_absolute_deviation(bin_values)# np.mean(abs(bin_values - bin_med)**2)
    bin_ratio = bin_mean / bin_med




    count_text = Label(x=65, y=135, x_units='screen', y_units='screen', text_font_size = "10px", text=f"count: {bin_count}")
    mean_text = Label(x=65, y=125, x_units='screen', y_units='screen', text_font_size = "10px", text=f"mean: {bin_mean:.3f}±{bin_mean_var:.3f}")

    if bin_med > 0.209:
        text_color='red'
    else:
        text_color='black'

    med_text = Label(x=65, y=115, x_units='screen', y_units='screen', text_font_size = "10px", text_color=text_color, text=f"med:   {bin_med:.3f}±{bin_med_var:.3f}")

    if bin_ratio > 1.5:
        text_color='red'
    else:
        text_color='black'

    ratio_text = Label(x=65, y=105, x_units='screen', y_units='screen', text_font_size = "10px", text_color=text_color, text=f"ratio: {bin_ratio:.3f}")

    p.add_layout(count_text)
    p.add_layout(mean_text)
    p.add_layout(med_text)
    p.add_layout(ratio_text)
    
    return p


def plot_bin_grid(value_groups, uvbins, nbins=50, ncols=4):

    plots = []
    for sb in uvbins:
        bin_count = len(value_groups[sb[0]][sb[1]])
        bin_name = f"{sb} - {bin_count}"
        plots.append( make_hist_plot(bin_name, value_groups[sb[0]][sb[1]], nbins) )

    show(gridplot(plots, ncols=ncols, plot_width=200, plot_height=200, toolbar_location=None))


def plot_bin_grid_2(bin_groups, bin_names, nbins=25, ncols=5, hrange=None):
    plots = []
    for sb, bin_name in zip(bin_groups, bin_names):
        bin_count = len(sb)
#         bin_name = f"count - {bin_count}"
#         print(len(sb), len(sb[np.where((sb>hrange[0])&(sb<hrange[1]))]))
        sb_sel = sb #[np.where((sb>hrange[0])&(sb<hrange[1]))]
        plots.append( make_hist_plot(bin_name, sb_sel, nbins, yrange=hrange) )
        
    show(gridplot(plots, ncols=ncols, plot_width=200, plot_height=200, toolbar_location=None))


def get_bin_thresholds(median_grid, std_grid, annulus_width, nsigma=3):

    npixu = median_grid.shape[0]
    npixv = median_grid.shape[1]

    for ind, edge in enumerate(annulus_width):
        minuv=0
        if ind:
            minuv = annulus_width[ind-1]    
        maxuv = annulus_width[ind]
        print(ind, edge, minuv, maxuv)
        idx = np.where((uvlen*gridsize))



def sigma_clip_upper(bin_values, alpha:float=0.05, maxiter:int=4):
    '''Rayleigh sigma clipping - upper only'''
    
    bin_mean, bin_med = np.mean(bin_values), np.median(bin_values)
    bin_mean_var, bin_med_var = np.std(bin_values), median_absolute_deviation(bin_values)# np.mean(abs(bin_values - bin_med)**2)
    bin_ratio = bin_mean / bin_med

    if bin_ratio > 1.2:
        low, high = get_rayleigh_thresholds(bin_values, alpha)
        bin_values_new = bin_values[np.where(bin_values < high)]
    else:
        bin_values_new = bin_values
    return bin_values_new
    
def compute_annulus_stats(median_grid, value_groups, annulus_width, uvbins, alpha=0.0027):

    uv_bins = np.asarray(median_grid>0).nonzero()
    uv_bins_ = np.array([uv_bins[0], uv_bins[1]])
    bin_uv_dist = np.sqrt(uvbins[0][uv_bins_[0]-1]**2 + uvbins[1][uv_bins_[1]-1]**2)

    for ind, edge in enumerate(annulus_width):
        minuv=0
        if ind:
            minuv = annulus_width[ind-1]
        maxuv = annulus_width[ind]

        ann_bin_ind = np.where( (bin_uv_dist > minuv) & (bin_uv_dist < maxuv) )
        ann_bin_ind = ann_bin_ind[0]

        ann_vals = np.array([value_groups[x-1][y-1] for x,y in zip(uv_bins_[:,ann_bin_ind][0], uv_bins_[:,ann_bin_ind][1])])
        ann_vals = np.concatenate(ann_vals)
        ann_med = np.median(ann_vals)
        
        rlower, rupper = get_rayleigh_thresholds(ann_vals, alpha)
        ann_mad = median_absolute_deviation(ann_vals)
        
        print(f'Annulus ({minuv:<5}- {maxuv:<5} lambda) median: {ann_med:.3f} mad: {ann_mad:.3f}')


def plot_one_annulus(value_groups, median_grid, annulus_range, uvbins, hrange, nbins=100, func=lognorm):
    
    uv_bins = np.asarray(median_grid>0).nonzero()
    uv_bins_ = np.array([uv_bins[0], uv_bins[1]])
    
    bin_uv_dist = np.sqrt(uvbins[0][uv_bins_[0]-1]**2 + uvbins[1][uv_bins_[1]-1]**2)
    minuv, maxuv = annulus_range[0], annulus_range[1]
    
    ann_bin_ind = np.where( (bin_uv_dist > minuv) & (bin_uv_dist < maxuv) )[0]
    ann_vals = np.array([value_groups[x-1][y-1] for x,y in zip(uv_bins_[:,ann_bin_ind][0], uv_bins_[:,ann_bin_ind][1])])
    ann_vals = np.concatenate(ann_vals)
    
    ann_med = np.median(ann_vals)
    ann_mad = median_absolute_deviation(ann_vals)
    ann_cnt = len(ann_vals)
    
    print(f'Annulus ({minuv:<5}- {maxuv:<5} lambda)\tcount: {ann_cnt}\t median: {ann_med:.3f} mad: {ann_mad:.3f}')

    params = plot_bin_distribution(ann_vals[np.where((ann_vals>hrange[0])&(ann_vals<hrange[1]))], func=func, nbins=100, hrange=hrange)

#     plotgrid.plot_bin_dist(ann_vals[np.where((ann_bin>0.)&(ann_bin<3.3))], func=func, nbins=100, hrange=[0,3.3])
    return params


def plot_grid_for_annulus(value_groups, median_grid, annulus_range, uvbins, hrange=[0,10]):

    uv_bins = np.asarray(median_grid>0).nonzero()
    uv_bins_ = np.array([uv_bins[0], uv_bins[1]])

    bin_uv_dist = np.sqrt(uvbins[0][uv_bins_[0]-1]**2 + uvbins[1][uv_bins_[1]-1]**2)
    minuv, maxuv = annulus_range[0], annulus_range[1]

    ann_bin_ind = np.where( (bin_uv_dist > minuv) & (bin_uv_dist < maxuv) )[0]    
    
    bin_names = [(u,v) for u,v in zip(uv_bins_[:,ann_bin_ind][0], uv_bins_[:,ann_bin_ind][1])]
    
    print(bin_names)
#     closure_uv_dist = np.sqrt(uvbins[0][uv_bins_[:,ann_bin_ind][0]-1]**2 + uvbins[1][uv_bins_[:,ann_bin_ind][1]-1]**2)
    ann_bins = np.array([value_groups[x][y] for x,y in zip(uv_bins_[:,ann_bin_ind][0], uv_bins_[:,ann_bin_ind][1])])

#     for bin_name, ann_bin in zip(bin_names, ann_bins):
#         bin_name_dist = np.sqrt((uvbins[0][bin_name[0]]-uvbins[0][bin_sel[0]])**2+(uvbins[1][bin_name[1]]-uvbins[1][bin_sel[1]])**2)
#         print(f"{bin_name:}\t{bin_name_dist:>7.0f}\t{np.median(ann_bin)})")


    print(f"Number of Bins: {len(ann_bins)}")
    bin_ratio_range = [0, 100]
    ann_bin_ratio = np.array([np.mean(_)/np.median(_) if np.median(_) > 0 else 0 for _ in ann_bins])

    bins = plot_bin_grid_2(ann_bins[np.where((ann_bin_ratio<bin_ratio_range[1]) & (ann_bin_ratio>bin_ratio_range[0]))][:90], bin_names, hrange=hrange)
#     return closure_uv_dist


def plot_grid_binrange(value_groups, bin_range, nbins=100):

    u_range, v_range = np.arange(bin_range[0][0], bin_range[0][1]+1), np.arange(bin_range[1][1], bin_range[1][0]-1, -1)
    
    bingrid = np.meshgrid(u_range, v_range)

    ncols = len(u_range)
    plots = []
    nbins = 50
    for xrow, yrow in zip(bingrid[0], bingrid[1]):
        for xcell,ycell in zip(xrow, yrow):
            bin_name = f"({xcell}, {ycell})"
            bin_group = value_groups[xcell][ycell]
            plots.append(make_hist_plot(bin_name, bin_group, nbins) )

    show(gridplot(plots, ncols=ncols, plot_width=200, plot_height=200, toolbar_location=None))


def dist_amp_hex_plot(ds_ind, annulus_width, data_comp):

    uvdist = np.sqrt(ds_ind.UV[:,0]**2 + ds_ind.UV[:,1]**2)
    imgval = np.absolute( ds_ind.DATA[:,data_comp[0]] + ds_ind.DATA[:,data_comp[1]] )

    bins = hexbin(uvdist[np.where(imgval>0)], imgval[np.where(imgval>0)], size=1., aspect_scale=1/100.)
    
    p = figure(title="Amplitude vs. Radius", tools="wheel_zoom,box_zoom,pan,reset", plot_width=1200, plot_height=600, background_fill_color='#FFFFFF')
    p.hex_tile(q="q", r="r", size=1.0, line_color=None, source=bins, fill_color=linear_cmap('counts', 'Viridis256', 0, max(bins.counts)))
    show(p)

##     Print Annulus Statistics from Points   
#     for ind, edge in enumerate(annulus_width):
#         minuv=0
#         if ind:
#             minuv = annulus_width[ind-1]
#         maxuv = annulus_width[ind]
#         ann_bin = imgval[np.where((uvdist>minuv) & (uvdist<maxuv) & (imgval>0))]
#         mean_median_amp = np.median(ann_bin)
#         mad_amp = median_absolute_deviation(ann_bin)
#         print(f'Annulus ({minuv:<5}- {maxuv:<5} lambda) median: {mean_median_amp:.2f} mad: {mad_amp:.2f}')