import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import webcolors
from IPython.core.getipython import get_ipython
from PIL import Image
import base64
import io
import re
import sys
import warnings
import dateutil.parser
from scipy import stats
from collections import OrderedDict

#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from matplotalt_constants import *

# Store as a global var so we don't have to reload the model between cells
chart_type_cls_model = None
chart_type_cls_processor = None

# Thanks to Gustavo at https://stackoverflow.com/questions/15411967
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter



def infer_single_axis(ax=None):
    if ax is None:
        ax = plt.gcf().get_axes()
    if isinstance(ax, (list, np.ndarray)):
        if len(ax) > 0 and isinstance(ax[0], matplotlib.axes._axes.Axes):
            ax = ax[0]
        else:
            raise ValueError(f"Given blank or multiple axes: {ax}")
    return ax


# Supports any Huggingface ImageClassification model with an ImageProcessor
def infer_model_chart_type(ax=None, model="KaiNylund/chart-classifier-tiny"):
    global chart_type_cls_model
    global chart_type_cls_processor
    ax = infer_single_axis(ax)
    # Filter out text objects since they can occur in any chart
    ax_child_types = [type(c) for c in ax._children if type(c) != matplotlib.text.Text]
    containers = ax.get_legend_handles_labels()
    # If the chart has a quadmesh then it's probably a heatmap
    if matplotlib.collections.QuadMesh in ax_child_types:
        return "heatmap"
    # If legend has Line2D objects, assume that it's a line chart
    elif containers and (len(containers[0]) > 0) and type(containers[0]) == matplotlib.lines.Line2D:
        return "line"
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #print("Cannot infer chart type from properties, using an automatic classifier...")
            if chart_type_cls_model is None:
                #print("Loading classifier...")
                from transformers import AutoImageProcessor, AutoModelForImageClassification
                chart_type_cls_model = AutoModelForImageClassification.from_pretrained(model, use_safetensors=True).to('cpu')
                chart_type_cls_processor = AutoImageProcessor.from_pretrained(model)
                chart_type_cls_model.eval()

            #print("Predicting chart type...")
            fig = plt.gcf()
            fig.canvas.draw()
            fig_img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb());
            inputs = chart_type_cls_processor(fig_img, return_tensors="pt");
            logits = chart_type_cls_model(**inputs).logits;

            # If there's no PathCollection object then it's probably not a scatter plot
            if matplotlib.collections.PathCollection not in ax_child_types:
                logits[0][chart_type_cls_model.config.label2id["scatter"]] = -100000
            # If there are no Line2D objects in get_lines, this probably isn't a line plot.
            # Otherwise, manually upweight "line".
            if len(ax.get_lines()) < 1:
                logits[0][chart_type_cls_model.config.label2id["line"]] = -100000
            else:
                logits[0][chart_type_cls_model.config.label2id["line"]] *= 2
            # Manually downweight "other" (when top logit is positive)
            logits[0][chart_type_cls_model.config.label2id["other"]] /= 2

            #print(logits)
            predicted_label = logits.argmax(-1).item()
            predicted_label = chart_type_cls_model.config.id2label[predicted_label]
            #print(predicted_label)
        return predicted_label


def get_arr_shape(arr, arr_ticklabels, var_label="data", generally_thresh=0.65, strictly_thresh=1.0, sig_figs=4):
    shape_desc = ""
    num_pts = len(arr)
    if num_pts < 3:
        return shape_desc
    arr_diff = np.diff(arr)
    max_idx = np.argmax(arr)
    min_idx = np.argmin(arr)
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
        shape_desc = f"{var_label} are constant."
    else:
        # Decreasing up to the min
        if pct_dec_up_to_min >= generally_thresh:
            inc_dec_desc1 = f"decrease to a min of {format_float(arr_ticklabels[min_idx], sig_figs)}"
            if (pct_dec_up_to_min + pct_flat_up_to_min) >= strictly_thresh:
                inc_dec_modifier1 = "strictly"
            if pct_flat_up_to_min > 0:
                inc_dec_modifier1 = "are constant or " + inc_dec_modifier1
        # Increasing up to the max
        elif pct_inc_up_to_max >= generally_thresh:
            inc_dec_desc1 = f"increase to a max of {format_float(arr_ticklabels[max_idx], sig_figs)}"
            if (pct_inc_up_to_max + pct_flat_up_to_max) >= strictly_thresh:
                inc_dec_modifier1 = "strictly"
            if pct_flat_up_to_max > 0:
                inc_dec_modifier1 = "are constant or " + inc_dec_modifier1
        # If there's a clear inc/dec trend up to the min/max
        if inc_dec_desc1 != "":
            shape_desc += f"{var_label} {inc_dec_modifier1} {inc_dec_desc1}"
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
            shape_desc += f", then {inc_dec_modifier2} {inc_dec_desc2}."
    #  No clear trends so just describe the pct. increasing, decreasing, and constant
    #if shape_desc == "":
    #    pct_increasing = (arr_diff > 0).sum() / (num_pts - 1)
    #    pct_decreasing = (arr_diff < 0).sum() / (num_pts - 1)
    #    shape_desc += f"{var_label} are increasing {format_float(100 * pct_increasing, sig_figs)}% of the time, decreasing {format_float(100 * pct_decreasing, sig_figs)}% of the time, and constant {format_float(100 * pct_flat, sig_figs)}% of the time."
    return shape_desc.strip()


# Nice one-liner from john1024 at https://stackoverflow.com/questions/29643352
def hex_to_rgb(h):
    """Convert hex code to rgb triplet"""
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# From https://stackoverflow.com/questions/60676893
def pillow_image_to_base64_string(img):
    """Convert PIL jpeg to base64 string"""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# Closest color matching from fraxel at https://stackoverflow.com/questions/9694165
def get_color_name(color_input):
    """Converts hex codes and rgb triplets to a color name like 'darkblue'"""
    if isinstance(color_input, str):
        if "#" in color_input:
            rgb_triplet = matplotlib.colors.to_rgb(color_input)
        else:
            return color_input
    elif isinstance(color_input, (tuple, list, np.ndarray)):
        rgb_triplet = color_input
    else:
        raise ValueError("Input of unknown type to get_color_name")

    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = matplotlib.colors.to_rgb(key)
        rd = (r_c - rgb_triplet[0]) ** 2
        gd = (g_c - rgb_triplet[1]) ** 2
        bd = (b_c - rgb_triplet[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


def is_number(s):
    """ Returns True if string is a number. """
    s = s.replace("−", "-")
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_date(s):
    """ Returns True if string is a date. """
    try:
        dateutil.parser.parse(s)
        return True
    except Exception:
        return False


def format_float(f, sig_figs=4, tol=1e-10):
    """returns the float as a string with the given number of signifigant figures"""
    if isinstance(f, str):
        f = f.replace("−", "-")
        try:
            f = float(f)
        except ValueError:
            return f
    if isinstance(f, (int, float)):
        if abs(f) < tol:
            f = 0
        return '{:g}'.format(float('{:.{p}g}'.format(f, p=sig_figs)))
    return f


def get_ax_ticks_type(ax_ticks, r_thresh=0.97):
    formatted_ticks = []
    tick_types = []
    # Get type of each axis tick individually
    for i in ax_ticks:
        if is_number(i):
            tick_types.append("number")
            formatted_ticks.append(float(i.replace("−", "-")))
        elif is_date(i):
            tick_types.append("date")
        else:
            tick_types.append("cat")
    # Get type of full axis
    ax_type = "categorical"
    # If all ticks are of the same type:
    if len(set(tick_types)) == 1:
        # If all ticks are numerical, try to infer whether they fall on a log or linear scale
        if tick_types[0] == "number":
            num_ticks = len(formatted_ticks)
            if stats.linregress(formatted_ticks, range(0, num_ticks)).rvalue > r_thresh:
                ax_type = "linear"
            elif stats.linregress(formatted_ticks, np.logspace(num_ticks, 1, num=num_ticks)).rvalue > r_thresh:
                ax_type = "log-linear"
            else: # Otherwise just say "numerical"
                ax_type = "numerical"
        elif tick_types[0] == "date":
            ax_type = "datetime"
    return ax_type


# Function from braunmagrin at https://stackoverflow.com/questions/54987129
def create_new_cell(contents):
    """Creates a new code cell in jupyter with the given contents"""
    shell = get_ipython()
    payload = dict(
        source='set_next_input',
        text=contents,
        replace=False,
        metadata={}
    )
    shell.payload_manager.write_payload(payload, single=True)


def insert_line_breaks(text, max_line_width=80):
    """Breaks the given text into lines of max length max_line_width"""
    tokens = text.split(" ")
    if len(tokens) > 0:
        cur_line_width = 0
        for i in range(len(tokens) - 1):
            cur_line_width += len(tokens[i])
            if cur_line_width + len(tokens[i+1]) > max_line_width:
                tokens[i] += "\n"
                cur_line_width = 0
    return " ".join(tokens)


def assert_equal_shapes(arrays):
    """Raises a value error if the given arrays do not all have the same shape"""
    first_shape = np.array(arrays[0]).shape
    for arr in arrays:
        if np.array(arr).shape != first_shape:
            raise ValueError("All given arrays must have the same shape")


def format_list(l):
    """Returns the given list as 'a, b, c, ..., and d'"""
    str_l = [str(i) for i in l]
    if len(str_l) > 1:
        return ", ".join(str_l[:-1]) + ", and " + str_l[-1]
    elif len(str_l) == 1:
        return str_l[0]
    else:
        return ""


def format_float_list(l, sig_figs):
    """
    Returns the given list as 'a, b, c, ..., and d' with each element rounded to the
    given number of signifigant figures
    """
    str_l = [format_float(i, sig_figs) for i in l]
    return format_list(str_l)


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


# xyz: tuple of coords in format (x, y, z, ...)
def get_outliers_desc(xyz, outlier_idxs, outlier_axis=None, max_outliers_desc=4, sig_figs=4):
    if outlier_axis:
        axis_desc = f"along the {outlier_axis}-axis"
    else:
        axis_desc = ""
    outlier_word = "outlier" if len(outlier_idxs) == 1 else "outliers"
    if len(outlier_idxs) == 0:
        return "no outliers"
    elif len(outlier_idxs) < max_outliers_desc:
        if np.array(xyz).ndim == 1:
            outlier_pts = [format_float(xyz[i], sig_figs) for i in outlier_idxs]
            return f"{len(outlier_pts)} {outlier_word} {axis_desc} at {outlier_axis}={format_list(outlier_pts)}"
        else:
            outlier_pts = [f"({', '.join([format_float(pt[i], sig_figs) for pt in xyz])})" \
                            for i in outlier_idxs]
            return f"{len(outlier_pts)} {outlier_word} {axis_desc} at {format_list(outlier_pts)}"
    else:
        return f"{len(outlier_idxs)} {outlier_word} {axis_desc}"


def idx_pt_desc(idx, ax_name_to_ticklabels, cur_stat_axis, var_idx=None, sig_figs=4):
    if len(ax_name_to_ticklabels) > 0:
        ax_names = list(ax_name_to_ticklabels.keys())
        if var_idx is not None:
            ax_to_pt_label = OrderedDict([(an, ax_name_to_ticklabels[an][var_idx][idx]) for an in ax_names if an != cur_stat_axis])
        else:
            ax_to_pt_label = OrderedDict([(an, ax_name_to_ticklabels[an][idx]) for an in ax_names if an != cur_stat_axis])
        pt_values = list(ax_to_pt_label.values())
        if len(pt_values) >= 2:
            return f" at ({', '.join([str(format_float(ptv, sig_figs)) for ptv in pt_values])})"
        elif len(pt_values) == 1:
            return f" at {list(ax_to_pt_label.keys())[0]}={format_float(pt_values[0], sig_figs)}"
        else:
            if var_idx is not None:
                idx_val = ax_name_to_ticklabels[ax_names[0]][var_idx][idx]
            else:
                idx_val = ax_name_to_ticklabels[ax_names[0]][idx]
            if not is_number(idx_val):
                return f" ({idx_val})"
    else:
        return ""

# headers_to_data: dict of headers to x/y/z/ticklabels list
# columns must all have the same length
def create_md_table(headers_to_data, sig_figs=4):
    table_str = ""
    if len(headers_to_data) > 0:
        headers = list(headers_to_data.keys())
        header_to_str_len = {h: max(len(h), np.max([len(str(di)) for di in headers_to_data[h]])) for h in headers}
        table_str += f"| {' | '.join([h + ' ' * (header_to_str_len[h] - len(h)) for h in headers])} |\n"
        table_str += f"| {' | '.join(['-' * header_to_str_len[h] for h in headers])} |\n"
        for i in range(len(next(iter(headers_to_data.values())))):
            padded_data = []
            for h in headers:
                rounded_f = str(format_float(headers_to_data[h][i], sig_figs=sig_figs))
                padded_data.append(rounded_f + ' ' * (header_to_str_len[h] - len(rounded_f)))
            table_str += f"| {' | '.join(padded_data)} |\n"
    return table_str


def url_safe(s):
    return re.sub("[^a-z0-9-_]", "", s.lower().replace(" ", "_"))


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


STAT_NAME_TO_FUNC = {
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


STAT_NAME_TO_DESC_INTRO = {
    "max": "a maximum value of",
    "min": "a minimum value of",
    "mean": "an average of",
    "median": "a median of",
    "std": "a standard deviation of",
    "diff": "a difference from start to end of",
}