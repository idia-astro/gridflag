import numpy as np
import numba as nb

# from scipy.stats import median_absolute_deviation

length_checker = np.vectorize(len) 

# @nb.jit(nogil=True)
def median_absolute_deviation(x, scale=1.4826):
    """Compute median absolute deviation. Adapted from scipy for use with numba [1].
    
    Parameters
    ----------
    x : array_like
        Input numpy array
    scale : scalar, optional
        The numerical value of scale with be multiplied by the final result. For a normal 
        distribution a value of 1/0.67449 is used (default).
        
    Returns
    -------
    mad : scalar or ndarray
        The scalar output value.
        
    References
    ----------
    [1] https://github.com/scipy/scipy/blob/master/scipy/stats/stats.py
    
    """
    if not x.size:
        return np.nan

    med = np.median(x)
    mad = np.median(np.abs(x - med))

    return mad * scale


@nb.njit(nogil=True)
def sigmaclip(data, siglow=3., sighigh=3., niter=-100, 
              use_median=False):
    """Remove outliers from data which lie more than siglow/sighigh sample standard 
    deviations from mean. Adapted from the scipy [1] and LOFAR tpk implementations [2].

    Parameters
    ----------

        data : numpy.ndarray
            Numpy array containing data values.

        siglow : float, optional
            Kappa multiplier for standard deviation. Std * siglow defines the value below 
            which data are rejected.

        sighigh : float, optional
            Kappa multiplier for standard deviation. Std * sighigh defines the value above
            which data are rejected.

        niter : int, optional
            Number of iterations to calculate mean & standard deviation, and reject 
            outliers, If niter is negative, iterations will continue until no more 
            clipping occurs or until abs('niter') is reached, whichever is reached first.

        use_median : bool, optional
            Use median of data instead of mean.

    Returns
    -------
        data : numpy.array
            Numpy array of data clipped data.
            
        ilow : float
            Lower threshold used for clipping.
            
        ihigh : float
            Upper threshold used for clipping.
            
        
    References
    ----------

    [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sigmaclip.html
    [2] https://tkp.readthedocs.io/en/r3.0/devref/tkp/utility/sigmaclip.html

    """

    ilow, ihigh = 0., 0.
    nniter = -niter if niter < 0 else niter
    i = 0
    for i in range(nniter):
        N = len(data)

        if N < 2:
            return data, ilow, ihigh

        if use_median:
            mean = np.median(data)
        else:
            mean = np.mean(data)

        sigma = np.sqrt((np.sum(data**2) - N*mean**2)/(N-1))
                
        ilow = mean - sigma * siglow
        ihigh = mean + sigma * sighigh
        
        newdata = data[(np.where((data>ilow) & (data<ihigh)))]
        
        if niter < 0:
            # break when no changes
            if (len(newdata) == len(data)):
                break
        data = newdata
        
    return data, ilow, ihigh
    

@nb.jit(nogil=True)
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



@nb.jit(nogil=True)
def get_fixed_thresholds(bin_median:float, alpha:float=0., sigma:int=0):
    """
    Get thresholds based on a Rayleigh distribution without fitting. Since there 
    is only one parameter, the median of the bin alone can be used to set the 
    shape and scale.
    
    Parameters
    ----------
    bin_medians : numpy.array
        Median values for each measurement in a bin
        
    alpha : float, optional
        The cumulative distribution function is alpha/2 at the lower threshold, and 
        1-alpha/2 for the upper threshold (either alpha or sigma is required).
        
    sigma : int, optional
        Level for two-sided confidence interval. Only integer values 1-5 are 
        accepted.
        
    Returns
    -------
    lthreshold, uthreshold : float
        Lower and upper thresholds used for clipping bin values
        
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



# @nb.jit(nogil=True)
def get_annulus_limits(ann_bin_median, sigma):

    ann_bin_median_log = np.log(ann_bin_median)
    log_limits = [np.median(ann_bin_median_log), median_absolute_deviation(ann_bin_median_log)]

    ci_lower, ci_upper = np.exp(log_limits[0] - sigma*log_limits[1]), np.exp(log_limits[0] + sigma*log_limits[1])

    return ci_lower, ci_upper


# ----------------------------------------------------------------------------------------

# @nb.jit
def process_annuli(median_grid, annulus_width, ubins, vbins, sigma=3.): 
    """
    Compute upper and lower bounds for a set of annuli of a two dimensional uv grid.
    
    Parameters
    ----------
    median_grid : numpy.array2d
        A two dimensional array containing the zero-clipped median values of uv-gridded
        observations.
    
    annulus_width : numpy.array
        A set of radial widths used to split a uv grid in to a set of annuli.
    
    ubins : numpy.array
        The lambda values defining the lower bound of each bin representing the x-axis in
        the parameter median_grid.

    vbins : numpy.array
        The lambda values defining the lower bound of each bin representing the y-axis in
        the parameter median_grid.
        
    sigma : int, optional
        The statistical significance used to compute the thresholds for each annulus. An 
        integer below six (default is 3).
        
    Returns
    -------
    
    annuli_limits : list(numpy.array)
        Two lists containing the lower and upper bounds for each annuli. Each list has the
        same dimension as annulus_width.
    
    annuli_grid : numpy.array2d
        A two dimensional list of the same shape as `median_grid`. Each uv-bin contains an 
        index corresponding to the annulus to which it belongs. Used with annuli_limits to
        map thresholds to each uv bin depending on it's radial position.
    
    """

    annuli_grid = -1*np.ones(median_grid.shape, dtype=np.int32)
    
    u_bins, v_bins = np.where(median_grid>0)
    uv_bins = np.vstack((u_bins, v_bins))
    
    #The `-1` in the following line account for the null zeroth row and col in median_grid
    bin_uv_dist = np.sqrt(ubins[uv_bins[0]-1]**2 + vbins[uv_bins[1]-1]**2)

    annuli_limits = np.empty( shape=(0, 2), dtype=np.float64 )
    
    for ind, edge in enumerate(annulus_width):
            minuv=0
            if ind:
                minuv = annulus_width[ind-1]
            maxuv = annulus_width[ind]

            ann_bin_index = np.where((bin_uv_dist > minuv) & (bin_uv_dist < maxuv))[0]
            ann_bin_names = np.array([(uv[0],uv[1]) for uv in uv_bins[:,ann_bin_index].T])
            
            ann_bin_median = np.array([median_grid[u][v] for (u,v) in ann_bin_names])
            
            ci_lower, ci_upper = get_annulus_limits(ann_bin_median, sigma)

            print("Annulus ", ind , " median: ", np.median(ann_bin_median), " limits: ", ci_lower, " - ", ci_upper)

            for (u,v) in ann_bin_names:
                annuli_grid[u][v] = ind

            annuli_limits = np.append(annuli_limits, np.array([[ci_lower, ci_upper]]), axis=0)

    return annuli_limits, annuli_grid


# ----------------------------------------------------------------------------------------


@nb.njit(nogil=True)
def get_rayleigh_thresholds(bin_median, alpha=0.045):
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
        raise Exception("Invalid value for alpha.")
    
    # Determine the single parameter of the Rayleigh distribution from the 
    # median (https://en.wikipedia.org/wiki/Rayleigh_distribution).
    sigma = bin_median/np.sqrt(2*np.log(2))
    
    lthreshold = sigma*np.sqrt(-2*np.log(1-alpha/2))
    uthreshold = sigma*np.sqrt(-2*np.log(alpha/2))
    
    return lthreshold, uthreshold



# @nb.njit(
#     nb.types.Tuple(
#         (nb.float64, nb.float64)
#     )(nb.float32[:,:], nb.float64, nb.float64, nb.float64),
#     locals={
#         "alpha": nb.float64,
#         "bin_median": nb.float32,
#         "bin_ratio": nb.float32
#     },
#     nogil=True
# )
@nb.njit(nogil=True)
def compute_bin_threshold(bin_val, ci_lower, ci_upper, sigma):
    """
    Contains the logic for clipping values for a uv bin. It takes a list of bin values and
    the lower and upper thresholds for the annulus corresponding to the bin. This function
    applies to Rayleigh distributed observations (i.e. the amplitude of any stokes 
    parameter). If the median of the bin is within the annulus limits, the thresholds are 
    computed using the confidence interval for the Rayleigh distribution. If the median is 
    above the annulus threshold, and the distribution is not rayleigh-like, the values are 
    first sigma clipped (see documentation for function `sigmaclip' above), then Rayleigh 
    thresholds are applied. Finally, if the bin's median is still outside the annulus 
    limits, the bin is dropped.
    
    Parameters
    ----------
    bin_val : numpy.array
        Values for one bin (zeros removed).
    ci_lower/ci_upper : float
        Upper and lower bounds for the annulus to which the bin belongs.
    alpha : float, optional
        The fractional significance used to clip outliers.
    sigma : int, optional
        The integer statistical significance used to clip outlier values.
        
    Returns
    -------
    lthreshold/uthreshold : float
        Upper and lower thresholds outside which bin values are flagged

    """
    
    if len(bin_val)==0:
        return -1., -1.

#     if not(alpha):
    alpha = 1./(2*len(bin_val))

    bin_median = np.median(bin_val)

    bin_ratio = np.mean(bin_val)/bin_median

    if (bin_median > ci_upper) or (bin_ratio > 1.2):

#         lthreshold, uthreshold = get_rayleigh_thresholds(ci_upper, alpha)
#         bin_val = bin_val[np.where( bin_val<uthreshold) ]
# 
#         if len(bin_val)==0:
#             return 0., 0.
#         bin_ratio = np.mean(bin_val)/np.median(bin_val)

        bin_val, low, high = sigmaclip(bin_val, sigma, sigma, -100, False)
        uthreshold = np.min(np.array([high, ci_upper]))

        if len(bin_val)==0:
            return 0., 0.
        if np.median(bin_val) > ci_upper:
            return 0., 0.

        bin_median = np.median(bin_val)
        lthreshold, uthreshold = get_rayleigh_thresholds(bin_median, alpha)
    else:
        lthreshold, uthreshold = get_rayleigh_thresholds(bin_median, alpha) 

    return lthreshold, uthreshold



# Change name to annulus_flagger or something better
@nb.njit(nogil=True)
def flag_one_annulus(uvbin_group, value_group, grid_row_map, preflags, annuli_limits, annuli_grid, sigma=3.):
    """
    Apply annulus thresholds and bin thresholds to each bin in a grid.
    
    Parameters
    ----------
    uvbin_group : np.array2d
        List bins and indicies sorted by uv bin.
    value_group : np.array
        List values (observations) corresponding to `uvbin_group`.
    grid_row_map : np.array2d
        A mapping where each row corresponds to one UV bin and its starting position index
        in the `uvbin_group` and `value_group` arrays.
    preflags : numpy.array
        A list of flags for rows with zero values to append to the flags from this 
        function.
    annuli_limits : list(numpy.array)
        Two lists containing the lower and upper bounds for each annuli. Each list has the
        same dimension as annulus_width.    
    annuli_grid : numpy.array2d
        A two dimensional list of uv-bins containing an index to map bins to annuli.
        
    Returns
    -------
    median_grid_flg : numpy.array2d
        A two dimensional list of uv-bins whose value is the median of the flagged bin 
        values
    flag_count_grid : array of shape (ubins, vbins)
        A uv-bin grid with flag count for each cell.
    flag_list : array of int64
        A list of indicies for the position in the input measurement set for flagged visibilities.
    val_flag_pos : array of int64
        The list of flags in the position of the trasnformed, binned data to use for 
        analysis and plotting.
    """
    
    median_grid_flg = np.zeros(annuli_grid.shape)
    flag_count_grid = np.zeros(annuli_grid.shape)
    flag_list = np.empty( shape=(0), dtype=np.int64 )
    
    val_flag_pos = np.empty( shape=(0), dtype=np.int64)

    print("Value group shape (flag_one_annulus): ", value_group.shape)

    print("Flagging ", len(uvbin_group), " rows in ", len(grid_row_map), " bins." )
    
    for i_bin, bin_location in enumerate(grid_row_map[:-1]):
        u, v, idx = bin_location
        
        # if not(i_bin%1000):
        #     print(i_bin, "/", len(grid_row_map))            

        istart, iend =  grid_row_map[i_bin][2], grid_row_map[i_bin+1][2]
        bin_val = value_group[istart:iend]
        bin_ind = uvbin_group[istart:iend,2]

        (ci_lower, ci_upper) = annuli_limits[annuli_grid[u][v]]

        lthreshold, uthreshold = compute_bin_threshold(bin_val, ci_lower, ci_upper, sigma)

        flag_mask = np.where((bin_val<=lthreshold) | (bin_val>=uthreshold))

        # Save unflagged visibilities for analytics on flagged data
        bin_flag_pos = istart + flag_mask[0]
        
        bin_flg = bin_ind[flag_mask]
        bin_val = bin_val[np.where((bin_val>lthreshold) & (bin_val<uthreshold))] 

        if len(bin_val)>0:
            median_grid_flg[u][v] = np.median(bin_val)

        if len(bin_flg):
            flag_list = np.append(flag_list, bin_flg)
            val_flag_pos = np.append(val_flag_pos, bin_flag_pos)

        flag_count_grid[u][v] = len(bin_flg)

    print( "Flagged ", int(100*len(flag_list)/len(value_group)), "% (", len(flag_list), " of ", len(value_group), " values) in ", len(grid_row_map), " bins." ) 

    # print("Flagged ", 100*len(flag_list)/len(value_group), "% of ", len(grid_row_map), " bins (", len(value_group), " values) with ", len(flag_list), " total flags")

    flag_list = np.append(flag_list, preflags)
    
    return median_grid_flg, flag_count_grid, flag_list, val_flag_pos
