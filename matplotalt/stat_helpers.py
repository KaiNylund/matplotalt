import numpy as np
from matplotalt_helpers import format_float, idx_pt_desc


def _max_sf(var_ax_data, ax_name_to_ticklabels=None, stat_axis=None, var_idx=None, sig_figs=None, **kwargs):
    max_idx = np.nanargmax(var_ax_data[stat_axis])
    max_pt = idx_pt_desc(max_idx, ax_name_to_ticklabels, stat_axis, var_idx=var_idx, sig_figs=sig_figs)
    return f"{stat_axis}={format_float(var_ax_data[stat_axis][max_idx], sig_figs)}{max_pt}"


def _min_sf(var_ax_data, ax_name_to_ticklabels=None, stat_axis=None, var_idx=None, sig_figs=None, **kwargs):
    min_idx = np.nanargmin(var_ax_data[stat_axis])
    min_pt = idx_pt_desc(min_idx, ax_name_to_ticklabels, stat_axis, var_idx=var_idx, sig_figs=sig_figs)
    return f"{stat_axis}={format_float(var_ax_data[stat_axis][min_idx], sig_figs)}{min_pt}"


def _mean_sf(var_ax_data, stat_axis=None, sig_figs=None, **kwargs):
    return f"{stat_axis}={format_float(np.nanmean(var_ax_data[stat_axis]), sig_figs)}"


def _median_sf(var_ax_data, stat_axis=None, sig_figs=None, **kwargs):
    return f"{stat_axis}={format_float(np.nanmedian(var_ax_data[stat_axis]), sig_figs)}"


def _std_sf(var_ax_data, stat_axis=None, sig_figs=None, **kwargs):
    return f"{stat_axis}={format_float(np.nanstd(var_ax_data[stat_axis]), sig_figs)}"


def _diff_sf(var_ax_data, stat_axis=None, sig_figs=None, **kwargs):
    return f"{stat_axis}={format_float(var_ax_data[stat_axis][-1] - var_ax_data[stat_axis][0], sig_figs)}"


def _num_slope_changes_sf(var_ax_data=None, stat_axis=None, **kwargs):
    num_slope_changes = np.count_nonzero(np.diff(np.sign(np.diff(var_ax_data[stat_axis]))))
    if num_slope_changes > 0:
        return f"data on the {stat_axis}-axis change from increasing to decreasing or vice versa {num_slope_changes} times"
    return ""


def _max_inc_sf(var_ax_data, ax_name_to_ticklabels=None, stat_axis=None, var_idx=None, sig_figs=None, **kwargs):
    arr_diff = np.diff(var_ax_data[stat_axis])
    max_inc_idx = np.argmax(arr_diff)
    max_inc_pt1 = idx_pt_desc(max_inc_idx, ax_name_to_ticklabels, stat_axis, var_idx=var_idx, sig_figs=sig_figs).replace("at ", "")
    max_inc_pt2 = idx_pt_desc(max_inc_idx + 1, ax_name_to_ticklabels, stat_axis, var_idx=var_idx, sig_figs=sig_figs).replace("at ", "")
    return f"a max increase of {stat_axis}={format_float(arr_diff[max_inc_idx], sig_figs)} from {max_inc_pt1} to {max_inc_pt2}"


def _max_dec_sf(var_ax_data, ax_name_to_ticklabels=None, stat_axis=None, var_idx=None, sig_figs=None, **kwargs):
    arr_diff = np.diff(var_ax_data[stat_axis])
    max_dec_idx = np.argmin(arr_diff)
    max_dec_pt1 = idx_pt_desc(max_dec_idx, ax_name_to_ticklabels, stat_axis, var_idx=var_idx, sig_figs=sig_figs).replace("at ", "")
    max_dec_pt2 = idx_pt_desc(max_dec_idx + 1, ax_name_to_ticklabels, stat_axis, var_idx=var_idx, sig_figs=sig_figs).replace("at ", "")
    return f"a max decrease of {stat_axis}={format_float(arr_diff[max_dec_idx], sig_figs)} from {max_dec_pt1} to {max_dec_pt2}"


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
}


BASE_STAT_NAME_TO_DESC_INTRO = {
    "max": "a maximum value of",
    "min": "a minimum value of",
    "mean": "an average of",
    "median": "a median of",
    "std": "a standard deviation of",
    "diff": "a difference from start to end of",
}