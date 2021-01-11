import numpy as np
import numba as nb

from scipy.stats import median_absolute_deviation, sigmaclip


length_checker = np.vectorize(len) 

@nb.jit
def median_abs_deviation(x, scale=1.4826):

    # Consistent with `np.var` and `np.std`.
    if not x.size:
        return np.nan

    med = np.median(x)
    mad = np.median(np.abs(x - med))

    return mad * scale

@nb.jit
def sigmaclip(data, siglow=3., sighigh=3., niter=-100, 
              use_median=False):
    """Remove outliers from data which lie more than siglow/sighigh
    sample standard deviations from mean. Adapted from LOFAR 
    implementation: https://tkp.readthedocs.io/en/r3.0/devref/tkp/utility/sigmaclip.html

    Args:

        data (numpy.ndarray): Numpy array containing data values.

    Kwargs:

        niter (int): Number of iterations to calculate mean & standard
            deviation, and reject outliers, If niter is negative,
            iterations will continue until no more clipping occurs or
            until abs('niter') is reached, whichever is reached first.

        siglow (float): Kappa multiplier for standard deviation. Std *
            siglow defines the value below which data are rejected.

        sighigh (float): Kappa multiplier for standard deviation. Std *
            sighigh defines the value above which data are rejected.

        use_median (bool): Use median of data instead of mean.

    Returns:
        tuple: (2-tuple) Boolean numpy array of data clipped data

    """

    ilow, ihigh = 0., 0.
    nniter = -niter if niter < 0 else niter
    i = 0
    for i in range(nniter):
        N = len(data)

        if use_median:
            mean = np.median(data)
        else:
            mean = np.mean(data)

        sigma = np.sqrt((np.sum(data**2) - N*mean**2)/(N-1))
                
        ilow = mean - sigma * siglow
        ihigh = mean + sigma * sighigh

        if N < 2:
            return data, ilow, ihigh
        
        newdata = data[(np.where((data>ilow) & (data<ihigh)))]
        
        if niter < 0:
            # break when no changes
            if (len(newdata) == len(data)):
                break
        data = newdata
        
    return data, ilow, ihigh

@nb.jit
def get_annulus_limits(ann_bin_median, sigma):

    ann_bin_median_log = np.log(ann_bin_median)
    log_limits = [np.median(ann_bin_median_log), median_abs_deviation(ann_bin_median_log)]

    ci_lower, ci_upper = np.exp(log_limits[0] - sigma*log_limits[1]), np.exp(log_limits[0] + sigma*log_limits[1])

    return ci_lower, ci_upper


@nb.jit
def compute_bin_threshold(bin_val, ci_lower, ci_upper, alpha=None, sigma=3.0):
    
    if len(bin_val)==0:
        return None

    if not(alpha):
        alpha = 1./len(bin_val)

    bin_med = np.median(bin_val)

    if bin_med > ci_upper:
        lthreshold, uthreshold = get_fixed_thresholds(ci_upper, alpha=alpha)
        bin_val_sel = bin_val[np.where( bin_val<uthreshold) ]
        if len(bin_val_sel)==0:
            return 0., 0.
        bin_ratio = np.mean(bin_val_sel)/np.median(bin_val_sel)
        if bin_ratio > 1.2:
            bin_val_sel, low, high = sigmaclip(bin_val_sel, sigma, sigma)
            uthreshold =np.min(np.array([high, uthreshold]))
        if len(bin_val_sel)==0:
            return 0., 0.
        if np.median(bin_val_sel) > ci_upper:
            return 0., 0.
        lthreshold, uthreshold = get_rayleigh_thresholds(bin_val_sel, alpha=alpha)
    else:
        lthreshold, uthreshold = get_rayleigh_thresholds(bin_val, alpha=alpha) 

    return lthreshold, uthreshold


@nb.jit
def get_fixed_thresholds(bin_median:float, alpha:float=0., sigma:int=0):
    """
    Get thresholds based on a Rayleigh distribution without fitting. Since there 
    is only one parameter, the median of the bin alone can be used to set the 
    shape and scale.
    
    parameters
    ----------
    bin_medians - array-like
        Median values for each measurement in a bin
    alpha - float
        The cumulative distribution function is alpha/2 at the lower threshold, and 
        1-alpha/2 for the upper threshold (either alpha or sigma is required).
    sigma - int
        Level for two-sided confidence interval. Only integer values 1-5 are 
        accepted.
    """
    
    if (alpha==0 and sigma==0):
        raise Exception("Please set either alpha of sigma.")
    
    if alpha==0:
        if sigma>0:
            if sigma==1:
                alpha=0.31731051
            elif sigma==2:
                alpha=0.04550026
            elif sigma==3:
                alpha=0.00269980
            elif sigma==4:
                alpha=0.00006334
            elif sigma==5:
                alpha=0.00000057        
            else:
                raise Exception("The value of sigma must be an integer less than 6.")
            
    # Determine the single parameter of the Rayleigh distribution from the 
    # median (https://en.wikipedia.org/wiki/Rayleigh_distribution).
    sigma_r = bin_median/np.sqrt(2*np.log(2))
    
    lthreshold = sigma_r*np.sqrt(-2*np.log(1-alpha/2))
    uthreshold = sigma_r*np.sqrt(-2*np.log(alpha/2))
    
    return lthreshold, uthreshold

@nb.jit
def get_rayleigh_thresholds(vals, alpha=0.045):
    '''
    Determine the sigma level thresholds for the Rayleigh distribution. 
    
    Parameters
    ----------
    vals : array-like
        A list of values for a bin or annulus
    alpha : float
        The number to input for the quartile distribution. The output will be
        the location at which the cululative distribution function will equal
        alpha/2 and 1-alpha/2. Corresponds roughly to 1-sigma -> 0.3173, 2-
        sigma = 0.0455, 3-sigma -> 0.0027 for the 68-95-99.7 percent confidence
        levels.
        
    Returns:
    lthreshold, uthreshold : tuple of floats
        Represents a two-sided interval for rejecting outliers at a p-value of 
        alpha/2.
    '''
    
    if (alpha > 1):
        print("Invalid value for alpha.")
    # The sample median
    median = np.median(vals)
    
    # Determine the single parameter of the Rayleigh distribution from the 
    # median (https://en.wikipedia.org/wiki/Rayleigh_distribution).
    sigma = median/np.sqrt(2*np.log(2))
    
    lthreshold = sigma*np.sqrt(-2*np.log(1-alpha/2))
    uthreshold = sigma*np.sqrt(-2*np.log(alpha/2))
    
    return lthreshold, uthreshold


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


# ----------------------------------------------------------------------------------------

@nb.jit
def process_annuli(median_grid, annulus_width, ubins, vbins, sigma=3.):    

    annuli_grid = -1*np.ones(median_grid.shape, dtype=np.int32)
    
    u_bins, v_bins = np.where(median_grid>0)
    uv_bins = np.vstack((u_bins, v_bins))
    
    bin_uv_dist = np.sqrt(ubins[uv_bins[0]]**2 + vbins[uv_bins[1]]**2)

    annuli_limits = np.empty( shape=(0, 2), dtype=np.float64 )
    
    for ind, edge in enumerate(annulus_width):
            minuv=0
            if ind:
                minuv = annulus_width[ind-1]
            maxuv = annulus_width[ind]

            ann_bin_index = np.where((bin_uv_dist > minuv) & (bin_uv_dist < maxuv))[0]
            ann_bin_names = np.array([(uv[0],uv[1]) for uv in uv_bins[:,ann_bin_index].T])
            
            ann_bin_median = np.array([median_grid[u][v] for (u,v) in ann_bin_names])
            
            print("zero values", len(ann_bin_median)-len(ann_bin_median.nonzero()[0]), "bins", len(ann_bin_median))
            print("ann_bin_median", ann_bin_median)

            ci_lower, ci_upper = get_annulus_limits(ann_bin_median, sigma)

            print("\t med: ", np.median(ann_bin_median), " limits: ", ci_lower, " - ", ci_upper)

            for (u,v) in ann_bin_names:
                annuli_grid[u][v] = ind

            annuli_limits = np.append(annuli_limits, np.array([[ci_lower, ci_upper]]), axis=0)

    return annuli_limits, annuli_grid



@nb.jit
def flag_one_annulus(uvbin_group, value_group, grid_row_map, annuli_limits, annuli_grid, sigma=3.):
    
    median_grid_flg = np.zeros(annuli_grid.shape)
    flag_list = np.empty( shape=(0), dtype=np.int64 )

    print("Flag annulus.")
    
    for i_bin, bin_location in enumerate(grid_row_map[:-1]):
        u, v, idx = bin_location
        
        if not(i_bin%10000):
            print(i_bin, "/", len(grid_row_map))
        
        istart, iend =  grid_row_map[i_bin][2], grid_row_map[i_bin+1][2]
        bin_val = value_group[istart:iend]
        bin_ind = uvbin_group[istart:iend,2]
        
        (ci_lower, ci_upper) = annuli_limits[annuli_grid[u][v]]
        
        lthreshold, uthreshold = compute_bin_threshold(bin_val, ci_lower, ci_upper, sigma=sigma)
        
        bin_flg = bin_ind[np.where((bin_val<=lthreshold) | (bin_val>=uthreshold))]
        bin_val = bin_val[np.where((bin_val>lthreshold) & (bin_val<uthreshold))] 
        
        if len(bin_val)>0:
            median_grid_flg[u][v] = np.median(bin_val)

        if len(bin_flg):
            flag_list = np.append(flag_list, bin_flg)
                
    return median_grid_flg, flag_list
