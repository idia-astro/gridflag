import numpy as np
from scipy.stats import median_absolute_deviation, sigmaclip

from .plotgrid import get_fixed_thresholds, get_rayleigh_thresholds

length_checker = np.vectorize(len) 


def compute_bin_threshold(bin_val, annulus_threshold, alpha=None):
    
    ci_lower, ci_upper = annulus_threshold[0], annulus_threshold[1]

    if len(bin_val)==0:
        return None

    if not(alpha):
        alpha = 1./len(bin_val)

    bin_med = np.median(bin_val)

    if bin_med > ci_upper:
        bin_thresholds = get_fixed_thresholds(ci_upper, alpha=alpha)
        bin_val_sel = bin_val[np.where( bin_val<bin_thresholds[1]) ]
        bin_ratio = np.mean(bin_val_sel)/np.median(bin_val_sel)
        if bin_ratio > 1.2:
            bin_val_sel, low, high = sigmaclip(bin_val_sel, 3, 3)
            bin_thresholds = (bin_thresholds[0], np.min([high, bin_thresholds[1]]) )
        if np.median(bin_val_sel) > ci_upper:
            return (0,0)
        bin_thresholds = get_rayleigh_thresholds(bin_val_sel, alpha=alpha)
    else:
        bin_thresholds = get_rayleigh_thresholds(bin_val, alpha=alpha) 

    return bin_thresholds


def get_annulus_limits(ann_bin_median, sigma):

    ann_bin_median_log = np.log(ann_bin_median)
    log_limits = [np.median(ann_bin_median_log), median_absolute_deviation(ann_bin_median_log)]

    ci_lower, ci_upper = np.exp(log_limits[0] - sigma*log_limits[1]), np.exp(log_limits[0] + sigma*log_limits[1])

    return ci_lower, ci_upper


def remove_zero_values(bin_val, bin_ind, bin_name):
    '''Remove visibilities with zero amplitude for this component before 
    applying bin filtering.
    
    params
    ------
    bin_val : array-like
        Array of lists of median values for all bins in annulus or other group 
        of bins.
    bin_ind : array-like
        Array of lists of indicies for all bins in annulus or other group of 
        bins.
    bin_name : array-like
        Array of lists of bin names for all bins in annulus or other group of 
        bins.
        
    Returns
    -------
    bin_val
        Array of lists of median values with zeros removed.
    bin_ind
        Array of lists of indicies with zeros removed.
    flag_ind
        Array of indicies for visibilities to flag.
        
    '''
    
    bin_val_c  = np.array([bv[np.where(bv>0)] for bv in bin_val])    
    bin_ind_c  = np.array([bi[np.where(bv>0)] for bv, bi in zip(bin_val, bin_ind)])

    flag_ind = np.array([bi[np.where(bv==0)] for bv, bi in zip(bin_val, bin_ind)])


    if len(flag_ind) > 0:
        flag_ind = np.concatenate(flag_ind)
    else:
        flag_ind = np.array([])

    print(f"\tflag_ind: {len(flag_ind)}\t bin_val_c: {len(bin_val_c)} \t bin_ind_c: {len(bin_ind_c)}")

    if len(bin_val_c):  
        ann_bin_cnt = length_checker(bin_val_c) 

        # Filter out Zero Length bins from Annulus Stats
        bin_val = bin_val_c[np.where(ann_bin_cnt>0)]
        bin_ind = bin_ind_c[np.where(ann_bin_cnt>0)]
        bin_name = bin_name[np.where(ann_bin_cnt>0)]
    else:
        bin_val, bin_ind, bin_name = np.array([]), np.array([]), np.array([])

    return bin_val, bin_ind, bin_name, flag_ind


def compute_annulus_bins(median_grid, uvbins, nbins):

    uv_bins = np.asarray(median_grid>0).nonzero()
    (u, v) = uv_bins
    uv_bins_ = np.array([uv_bins[0], uv_bins[1]])
    bin_uv_dist = np.sqrt(uvbins[0][uv_bins_[0]-1]**2 + uvbins[1][uv_bins_[1]-1]**2)
    bin_uv_dist_sort = np.sort(bin_uv_dist)
    
    annulus_bins = np.quantile(bin_uv_dist_sort, np.linspace(0,1,nbins+1)[1:])
    return annulus_bins


def select_annulus_bins(minuv, maxuv, value_groups, index_groups, median_grid, uvbins):

    # Find grid of non-empty of non-trivial bins - equivalent to np.where(median_grid>0)
    uv_bins = np.asarray(median_grid>0).nonzero()

    # Convert from tuple of arrays to ndarray
    uv_bins_ = np.array([uv_bins[0], uv_bins[1]])

    # Compute uv-distance of each bin
    bin_uv_dist = np.sqrt(uvbins[0][uv_bins_[0]-1]**2 + uvbins[1][uv_bins_[1]-1]**2)

    annulus_bins = np.where((bin_uv_dist > minuv) & (bin_uv_dist < maxuv))[0]

    ann_bin_val = np.array([np.array(value_groups[x-1][y-1]) for x,y in zip(uv_bins_[:,annulus_bins][0], uv_bins_[:,annulus_bins][1])])
    ann_bin_ind = np.array([np.array(index_groups[x-1][y-1], dtype=int) for x,y in zip(uv_bins_[:,annulus_bins][0], uv_bins_[:,annulus_bins][1])])
    ann_bin_name = np.array([(x-1,y-1) for x,y in zip(uv_bins_[:,annulus_bins][0], uv_bins_[:,annulus_bins][1])])

    # Remove zero amplitude values from all bins
    ann_bin_val, ann_bin_ind, ann_bin_name, bin_flag_ind = remove_zero_values(ann_bin_val, ann_bin_ind, ann_bin_name)
    
    return ann_bin_val, ann_bin_ind, ann_bin_name, bin_flag_ind



def compute_annulus_stats(median_grid, value_groups, index_groups, uvbins, annulus_width, sigma=3.):

    median_grid_flg = np.zeros(median_grid.shape)
    value_groups_flg = [[[] for r_ in c_] for c_ in value_groups]
    flag_ind_list = []

    for ind, edge in enumerate(annulus_width):
        minuv=0
        if ind:
            minuv = annulus_width[ind-1]
        maxuv = annulus_width[ind]

        print(f"Annulus ({minuv}-{maxuv}): ")

        ann_bin_val, ann_bin_ind, ann_bin_name, bin_flag_ind = select_annulus_bins(minuv, maxuv, value_groups, index_groups, median_grid, uvbins)

        # Add pre-flagged visibilities
        flag_ind_list.append(bin_flag_ind)
        
        # Re-compute bin medians
        ann_bin_median = np.array([np.median(_) for _ in ann_bin_val])

        ci_lower, ci_upper = get_annulus_limits(ann_bin_median, sigma)
        
        print(f"\t count: {len(ann_bin_val)},\t median: {np.median(ann_bin_median):.3f},\t pre-flags:{len(bin_flag_ind)}\t CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

        for bin_val, bin_ind, bin_name in zip(ann_bin_val, ann_bin_ind, ann_bin_name):
            u, v = bin_name[0], bin_name[1]
            bin_thresholds = compute_bin_threshold(bin_val, (ci_lower, ci_upper))
    
            bin_flg = bin_ind[np.where((bin_val<=bin_thresholds[0]) | (bin_val>=bin_thresholds[1]))]
            bin_val = bin_val[np.where((bin_val>bin_thresholds[0]) & (bin_val<bin_thresholds[1]))] 

            value_groups_flg[u][v] = bin_val
            median_grid_flg[u][v] = np.median(bin_val)
            flag_ind_list.append(bin_flg)
        
        print(f"\t post-flags: {len(flag_ind_list)}")
        
    flag_ind_list = np.concatenate(flag_ind_list)

    return median_grid_flg, value_groups_flg, flag_ind_list
