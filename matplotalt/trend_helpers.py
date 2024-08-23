import numpy as np
from scipy.stats import pearsonr
from matplotalt_helpers import format_float, format_list, idx_pt_desc



# TODO: update to use lines of fit instead of arbitrary heuristics
def _shape_tf(chart_dict, var_name, ax_name, generally_thresh=0.65, strictly_thresh=1.0, sig_figs=4, **kwargs):
    shape_desc = ""
    var_ax_data = chart_dict["var_info"][var_name]["data"][ax_name]
    num_pts = len(var_ax_data)
    if num_pts < 3:
        return shape_desc
    arr_diff = np.diff(var_ax_data)
    max_idx = np.argmax(var_ax_data)
    min_idx = np.argmin(var_ax_data)
    pct_flat = (arr_diff == 0).sum() / (num_pts - 1)
    if max_idx > 1:
        pct_inc_up_to_max = (arr_diff[:(max_idx - 1)] > 0).sum() / (max_idx - 1)
        pct_flat_up_to_max = (arr_diff[:(max_idx - 1)] == 0).sum() / (max_idx - 1)
    else:
        pct_inc_up_to_max = 0
        pct_flat_up_to_max = 0
    if min_idx > 1:
        pct_dec_up_to_min = (arr_diff[:(min_idx - 1)] < 0).sum() / (min_idx - 1)
        pct_flat_up_to_min = (arr_diff[:(min_idx - 1)] == 0).sum() / (min_idx - 1)
    else:
        pct_dec_up_to_min = 0
        pct_flat_up_to_min = 0

    inc_dec_modifier1 = "generally"
    inc_dec_modifier2 = "generally"
    inc_dec_desc1 = ""
    inc_dec_desc2 = ""
    # Data is constant
    if pct_flat >= 1.0:
        shape_desc = f"{var_name} are constant"
    else:
        # Decreasing up to the min
        if pct_dec_up_to_min >= generally_thresh:
            inc_dec_desc1 = f"decrease to a min at {idx_pt_desc(min_idx, chart_dict, var_name, ax_name, sig_figs=sig_figs)}"
            if (pct_dec_up_to_min + pct_flat_up_to_min) >= strictly_thresh:
                inc_dec_modifier1 = "strictly"
            if pct_flat_up_to_min > 0:
                inc_dec_modifier1 = "are constant or " + inc_dec_modifier1
        # Increasing up to the max
        elif pct_inc_up_to_max >= generally_thresh:
            inc_dec_desc1 = f"increase to a max at {idx_pt_desc(max_idx, chart_dict, var_name, ax_name, sig_figs=sig_figs)}"
            if (pct_inc_up_to_max + pct_flat_up_to_max) >= strictly_thresh:
                inc_dec_modifier1 = "strictly"
            if pct_flat_up_to_max > 0:
                inc_dec_modifier1 = "are constant or " + inc_dec_modifier1
        # If there's a clear inc/dec trend up to the min/max
        if inc_dec_desc1 != "":
            shape_desc += f"{var_name} {inc_dec_modifier1} {inc_dec_desc1}"
            # Decreasing after the max
            if max_idx != num_pts - 1:
                pct_dec_past_max = (arr_diff[(max_idx - 1):] < 0).sum() / (num_pts - max_idx - 1)
                pct_flat_past_max = (arr_diff[(max_idx - 1):] == 0).sum() / (num_pts - max_idx - 1)
                if pct_dec_past_max >= generally_thresh:
                    inc_dec_desc2 = "decrease"
                    if (pct_dec_past_max + pct_flat_past_max) >= strictly_thresh:
                        inc_dec_modifier2 = "strictly"
                    if pct_flat_past_max > 0:
                        inc_dec_modifier2 = "are constant or " + inc_dec_modifier2
            # Increasing after the min
            elif min_idx != num_pts - 1:
                pct_inc_past_min = (arr_diff[(min_idx - 1):] > 0).sum() / (num_pts - min_idx - 1)
                pct_flat_past_min = (arr_diff[(min_idx - 1):] == 0).sum() / (num_pts - min_idx - 1)
                if pct_inc_past_min >= generally_thresh:
                    inc_dec_desc2 = "increase"
                    if (pct_inc_past_min + pct_flat_past_min) >= strictly_thresh:
                        inc_dec_modifier2 = "strictly"
                    if pct_flat_past_min > 0:
                        inc_dec_modifier2 = "are constant or " + inc_dec_modifier2
        if inc_dec_desc2 != "":
            shape_desc += f", then {inc_dec_modifier2} {inc_dec_desc2}"
    #  No clear trends so just describe the pct. increasing, decreasing, and constant
    #if shape_desc == "":
    #    pct_increasing = (arr_diff > 0).sum() / (num_pts - 1)
    #    pct_decreasing = (arr_diff < 0).sum() / (num_pts - 1)
    #    shape_desc += f"{var_label} are increasing {format_float(100 * pct_increasing, sig_figs)}% of the time, decreasing {format_float(100 * pct_decreasing, sig_figs)}% of the time, and constant {format_float(100 * pct_flat, sig_figs)}% of the time."
    return shape_desc.strip()



def _correlation_tf(chart_dict, var_name, ax_name, sig_figs, **kwargs):
    var_names = list(chart_dict["var_info"].keys())
    num_vars = len(var_names)
    if num_vars < 2:
        return ""
    elif num_vars == 2:
        var_ax_data1 = chart_dict["var_info"][var_names[0]]["data"][ax_name]
        var_ax_data2 = chart_dict["var_info"][var_names[1]]["data"][ax_name]
        return f"{var_names[0]} and {var_names[1]} have a correlation of {format_float(pearsonr(var_ax_data1, var_ax_data2).statistic, sig_figs)}"
    else:
        max_corr = -2
        min_corr = 2
        min_corr_vars = []
        max_corr_vars = []
        for i in range(num_vars):
            for j in range(num_vars):
                var_ax_datai = chart_dict["var_info"][var_names[i]]["data"][ax_name]
                var_ax_dataj = chart_dict["var_info"][var_names[j]]["data"][ax_name]
                cur_vars_corr = pearsonr(var_ax_datai, var_ax_dataj).statistic
                if cur_vars_corr < min_corr:
                    min_corr = cur_vars_corr
                    min_corr_vars = [var_names[i], var_names[j]]
                if cur_vars_corr > max_corr:
                    min_corr = cur_vars_corr
                    max_corr_vars = [var_names[i], var_names[j]]
        return f"{max_corr_vars[0]} and {max_corr_vars[1]} have the highest correlation (r={format_float(max_corr, sig_figs)}), while {min_corr_vars[0]} and {min_corr_vars[1]} have the lowest (r={format_float(min_corr, sig_figs)})"


BASE_TREND_NAME_TO_FUNC = {
    "shape": _shape_tf,
    "correlation": _correlation_tf
}