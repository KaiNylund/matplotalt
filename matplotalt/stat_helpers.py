import numpy as np
from matplotalt_helpers import format_float, idx_pt_desc


# each function should take: chart_dict, var_name, ax_name, sig_figs


def _max_sf(chart_dict, var_name, ax_name, sig_figs, **kwargs):
    var_ax_data = chart_dict["var_info"][var_name]["data"][ax_name]
    max_idx = np.nanargmax(var_ax_data)
    max_pt_desc = idx_pt_desc(max_idx, chart_dict, var_name, ax_name, sig_figs=sig_figs)
    max_pt_desc = f" at {max_pt_desc}" if max_pt_desc != "" else ""
    return f"a maximum value of {ax_name}={format_float(var_ax_data[max_idx], sig_figs)}{max_pt_desc}"

def _min_sf(chart_dict, var_name, ax_name, sig_figs, **kwargs):
    var_ax_data = chart_dict["var_info"][var_name]["data"][ax_name]
    min_idx = np.nanargmin(var_ax_data)
    min_pt_desc = idx_pt_desc(min_idx, chart_dict, var_name, ax_name, sig_figs=sig_figs)
    min_pt_desc = f" at {min_pt_desc}" if min_pt_desc != "" else ""
    return f"a minimum value of {ax_name}={format_float(var_ax_data[min_idx], sig_figs)}{min_pt_desc}"


def _mean_sf(chart_dict, var_name, ax_name, sig_figs, **kwargs):
    var_ax_data = chart_dict["var_info"][var_name]["data"][ax_name]
    return f"an average of {ax_name}={format_float(np.nanmean(var_ax_data), sig_figs)}"


def _median_sf(chart_dict, var_name, ax_name, sig_figs, **kwargs):
    var_ax_data = chart_dict["var_info"][var_name]["data"][ax_name]
    return f"a median of {ax_name}={format_float(np.nanmedian(var_ax_data), sig_figs)}"


def _std_sf(chart_dict, var_name, ax_name, sig_figs, **kwargs):
    var_ax_data = chart_dict["var_info"][var_name]["data"][ax_name]
    return f"a standard deviation of {ax_name}={format_float(np.nanstd(var_ax_data), sig_figs)}"


def _diff_sf(chart_dict, var_name, ax_name, sig_figs, **kwargs):
    var_ax_data = chart_dict["var_info"][var_name]["data"][ax_name]
    return f"a difference from start to end of {ax_name}={format_float(var_ax_data[-1] - var_ax_data[0], sig_figs)}"


def _num_slope_changes_sf(chart_dict, var_name, ax_name, sig_figs, **kwargs):
    var_ax_data = chart_dict["var_info"][var_name]["data"][ax_name]
    num_slope_changes = np.count_nonzero(np.diff(np.sign(np.diff(var_ax_data))))
    if num_slope_changes > 0:
        return f"data on the {ax_name}-axis change from increasing to decreasing or vice versa {num_slope_changes} times"
    return ""


def _max_inc_sf(chart_dict, var_name, ax_name, sig_figs, **kwargs):
    var_ax_data = chart_dict["var_info"][var_name]["data"][ax_name]
    arr_diff = np.diff(var_ax_data)
    max_inc_idx = np.argmax(arr_diff)
    max_inc_pt1 = idx_pt_desc(max_inc_idx, chart_dict, var_name, ax_name, sig_figs=sig_figs)
    max_inc_pt2 = idx_pt_desc(max_inc_idx + 1, chart_dict, var_name, ax_name, sig_figs=sig_figs)
    return f"a max increase of {ax_name}={format_float(arr_diff[max_inc_idx], sig_figs)} from {max_inc_pt1} to {max_inc_pt2}"


def _max_dec_sf(chart_dict, var_name, ax_name, sig_figs, **kwargs):
    var_ax_data = chart_dict["var_info"][var_name]["data"][ax_name]
    arr_diff = np.diff(var_ax_data)
    max_dec_idx = np.argmin(arr_diff)
    max_dec_pt1 = idx_pt_desc(max_dec_idx, chart_dict, var_name, ax_name, sig_figs=sig_figs)
    max_dec_pt2 = idx_pt_desc(max_dec_idx + 1, chart_dict, var_name, ax_name, sig_figs=sig_figs)
    return f"a max decrease of {ax_name}={format_float(arr_diff[max_dec_idx], sig_figs)} from {max_dec_pt1} to {max_dec_pt2}"


# Shouldn't change based on the given axis
def _numpts_sf(chart_dict, var_name, ax_name, sig_figs, **kwargs):
    return f"{len(next(iter(chart_dict['var_info'][var_name]['data'].values())))} {chart_dict['var_info'][var_name]['mark_type']}s"


def _linearfit_sf(chart_dict, var_name, ax_name, sig_figs, **kwargs):
    var_data = chart_dict["var_info"][var_name]["data"]
    linear_fit = np.polyfit(np.squeeze(var_data["x"]), np.squeeze(var_data["y"]), deg=1)
    return f"a linear fit of y={format_float(linear_fit[0], sig_figs)}x+{format_float(linear_fit[1], sig_figs)}"


def get_quartile_outlier_idxs(arr):
    """
    Given an array of floats, returns the indices of outliers defined as observations
    that fall below Q1 − 1.5 IQR or above Q3 + 1.5 IQR.
    """
    Q1 = np.quantile(arr, 0.25)
    Q3 = np.quantile(arr, 0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_idxs = np.squeeze(np.argwhere((arr > upper) | (arr < lower)))
    if outlier_idxs.ndim == 0:
        outlier_idxs = [outlier_idxs]
    if len(outlier_idxs) > 0:
        return np.sort(outlier_idxs)
    return []


def _outliers_sf(chart_dict, var_name, ax_name, max_outliers_desc, sig_figs):
    var_data = chart_dict["var_info"][var_name]["data"]
    xyz = list(var_data.values())
    outlier_idxs_arr = [get_quartile_outlier_idxs(ax_data) for ax_data in xyz]
    outlier_idxs = np.unique(np.concatenate(outlier_idxs_arr)).astype(int)
    outlier_word = "outlier" if len(outlier_idxs) == 1 else "outliers"
    if len(outlier_idxs) == 0:
        return "no outliers"
    elif len(outlier_idxs) < max_outliers_desc:
        outlier_pts = idx_pt_desc(outlier_idxs, chart_dict, var_name, ax_name, sig_figs=sig_figs)
        return f"{len(outlier_idxs)} {outlier_word} at {outlier_pts}"
    else:
        return f"{len(outlier_idxs)} {outlier_word}"



BASE_STAT_NAME_TO_FUNC = {
    "max": _max_sf,
    "min": _min_sf,
    "mean": _mean_sf,
    "median": _median_sf,
    "std": _std_sf,
    "diff": _diff_sf,
    "num_slope_changes": _num_slope_changes_sf,
    "maxinc": _max_inc_sf,
    "maxdec": _max_dec_sf,
    "numpts": _numpts_sf,
    "linearfit": _linearfit_sf,
    "outliers": _outliers_sf,
}