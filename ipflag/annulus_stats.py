from scipy.stats import median_absolute_deviation
import numpy as np

def isodd(num):
    """
    Checks if a number is even or odd
    """

    if num % 2 != 0:
        return True
    else:
        return False


def get_bin_thresholds(median_grid, std_grid, nsigma, annulus_width, binwidth):
    """
    Calculates the thresholds per UV-bin given the input median_grid and
    std_grid. The thresholds are calculated at nsigma, per annulus.

    Inputs:
    -------
    median_grid:    Grid of median values, 2d array, np.float
    std_grid :      Grid of standard deviation values, 2d array, np.float
    nsigma :        Sigma cutoff to use while calculating bin threshold values, np.float
    annulus_width : Outer edges of the annuli in the UV plane, list, np.float
    binwidth :      Size of the grid in the U and V dimensions in lambda, list, np.float

    Returns:
    --------
    threshold_min : Min threshold values per UV-bin, 2d array, np.float
    threshold_max : Max threshold values per UV-bin, 2d array, np.float
    """

    npixu = median_grid.shape[0]//2
    npixv = median_grid.shape[1]//2

    if isodd(median_grid.shape[0]):
        u = np.arange(-npixu, npixu+1)
    else:
        u = np.arange(-npixu, npixu)

    if isodd(median_grid.shape[1]):
        v = np.arange(-npixv, npixv+1)
    else:
        v = np.arange(-npixv, npixv)

    print(u.size, v.size)
    print(npixu, npixv)
    uu, vv = np.meshgrid(u, v)
    uvlen = np.hypot(uu, vv).T

    print(median_grid.shape, uvlen.shape)

    ann_med = []
    ann_std = []
    ann_idx = []
    if len(annulus_width) <= 1:
        print("Warning : No annuli specified - stats will be calculated over the whole grid.")
        print("This is not a good thing!")

        ann_med.append(np.median(median_grid))
        ann_std.append(np.std(median_grid))

    else:
        for ind, edge in enumerate(annulus_width):
            if ind == 0:
                minuv = 0
            else:
                minuv = annulus_width[ind-1]

            maxuv = annulus_width[ind]
            idx = np.where((uvlen*gridsize > minuv) & (uvlen*gridsize < maxuv))
            print(np.asarray(idx).shape)
            ann_idx.append(idx)
            tmpmed = median_grid[idx]
            #tmpstd = std_grid[idx]

            std = 1.4826 * median_absolute_deviation(tmpmed)

            ann_med.append(np.median(tmpmed))
            ann_std.append(std)

    flagged_grid = reject_outlier_bins(tmpmed, ann_med, ann_std, ann_idx)
    np.save('flagged_med_grid.npy', flagged_grid)

    min_grid = np.zeros_like(flagged_grid)
    max_grid = np.zeros_like(flagged_grid)

    return min_grid, max_grid




def reject_outlier_bins(grid, ann_med, ann_std, ann_idx, clip_sigma=3.0):
    """
    Given the input grid of values and the annular
    median and std values, rejects the outliers bins
    above clip_sigma.

    The rejected outliers will be set to 0.

    Inputs:
    -------
    grid :     Grid of standard deviation/median values, 2d array, np.float
    ann_med :  List of annulus median values, 2d array, np.float
    ann_std :  List of annulus std values, 2d array, np.float
    ann_idx :  List of indices to index into grid to obtain annulus values, 2d array, bool
    clip_sigma: Sigma at which to reject outliers, float

    Returns:
    -------
    flagged_grid : Grid where the outliers per annulus have been rejected, 2d array, np.float
    """

    flagged_grid = np.copy(grid)

    for med, std, idx in zip(ann_med, ann_std, ann_idx):
        print(med, std, np.asarray(idx).shape)
        min_thresh = med - clip_sigma*std
        max_thresh = med + clip_sigma*std

        idx2 = np.where((grid[idx] < min_thresh) | (grid[idx] > max_thresh))

        flagged_grid[idx][idx2] = 0.0

    return flagged_grid






