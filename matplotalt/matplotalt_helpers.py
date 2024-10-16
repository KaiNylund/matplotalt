import io
import re
import base64
import warnings
import numpy as np
from PIL import Image
import dateutil.parser
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict
from IPython.core.getipython import get_ipython

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
        #print("Cannot infer chart type from properties, using an automatic classifier...")
        if chart_type_cls_model is None:
            print("Loading classifier...")
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

        # Manual up/down weighting based on figure properties
        # If there's no PathCollection object then it's probably not a scatter plot
        #if matplotlib.collections.PathCollection not in ax_child_types:
        #    logits[0][chart_type_cls_model.config.label2id["scatter"]] = -100000
        # If there are no Line2D objects in get_lines, this probably isn't a line plot.
        # Otherwise upweight "line".
        #if len(ax.get_lines()) < 1:
        #    logits[0][chart_type_cls_model.config.label2id["line"]] = -100000
        #else:
        #    logits[0][chart_type_cls_model.config.label2id["line"]] *= 2
        # Downweight "other" (when top logit is positive)
        #logits[0][chart_type_cls_model.config.label2id["other"]] /= 2

        #print(logits)
        predicted_label = logits.argmax(-1).item()
        predicted_label = chart_type_cls_model.config.id2label[predicted_label]
        #print(predicted_label)
        return predicted_label



# Nice one-liner from john1024 at https://stackoverflow.com/questions/29643352
def hex_to_rgb(h):
    """Convert hex code to rgb triplet"""
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


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
            color_input = color_input[1:]
        rgb_triplet = hex_to_rgb(color_input)
    elif isinstance(color_input, (tuple, list, np.ndarray)):
        rgb_triplet = color_input
    else:
        raise ValueError("Input of unknown type to get_color_name")

    min_colors = {}
    for hex_val, name in HEX_COLOR_TO_NAME.items():
        r_c, g_c, b_c = hex_to_rgb(hex_val)
        rd = (r_c - rgb_triplet[0]) ** 2
        gd = (g_c - rgb_triplet[1]) ** 2
        bd = (b_c - rgb_triplet[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


def is_number(s):
    """ Returns True if string is a number. """
    if isinstance(s, str):
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
    if isinstance(f, (int, float)) or \
       np.issubdtype(f, np.integer) or \
       np.issubdtype(f, np.floating):
        if abs(f) < tol:
            f = 0
        return '{:g}'.format(float('{:.{p}g}'.format(f, p=sig_figs)))
    return f


def get_ax_ticks_scale(ax_ticks, r_thresh=0.97):
    formatted_ticks = []
    tick_types = []
    # Get type of each axis tick individually
    for i in ax_ticks:
        if isinstance(i, str):
            i = i.replace("−", "-")
        if is_number(i):
            tick_types.append("number")
            formatted_ticks.append(float(i))
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
    str_l = [str(i) for i in l if str(i) != ""]
    num_el = len(str_l)
    if num_el > 2:
        return ", ".join(str_l[:-1]) + ", and " + str_l[-1]
    elif num_el == 2:
        return f"{str_l[0]} and {str_l[1]}"
    elif num_el == 1:
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


def idx_pt_desc(idxs, chart_dict, var_name, excluded_axis, sig_figs=4):
    ax_info = chart_dict["ax_info"]
    if len(ax_info) > 0:
        if not isinstance(idxs, (list, np.ndarray)):
            idxs = [idxs]
        idxs_desc_arr = []
        var_data = chart_dict["var_info"][var_name]["data"]
        ax_names = list(var_data.keys())
        for i, idx in enumerate(idxs):
            ax_name_to_idx_labels = OrderedDict()
            for ax_name, ax_dict in ax_info.items():
                if (ax_name != excluded_axis or len(ax_names) == 1) and ax_name in var_data:
                    if "ticklabels" in ax_dict and len(var_data[ax_name]) <= len(ax_dict["ticklabels"]):
                        ax_name_to_idx_labels[ax_name] = ax_dict["ticklabels"][idx]
                    # Otherwise if data is evenly spaced and linear, use pts from range min to max
                    elif "range" in ax_dict and "scale" in ax_dict and ax_dict["scale"] == "linear":
                        print("scaling")
                        # make sure min/max line up and points are evenly spaced
                        data_ax_diff = np.diff(var_data[ax_name])
                        if ax_dict["range"][0] == np.nanmin(var_data[ax_name]) and \
                           ax_dict["range"][1] == np.nanmax(var_data[ax_name]) and \
                           np.all(np.isclose(data_ax_diff, data_ax_diff[0])):
                            ax_name_to_idx_labels[ax_name] = idx * (ax_dict["range"][1] - ax_dict["range"][0]) / len(var_data[ax_name])
                    # Otherwise use the actual data
                    if ax_name not in ax_name_to_idx_labels and idx < len(var_data[ax_name]):
                        print("Actual data")
                        ax_name_to_idx_labels[ax_name] = var_data[ax_name][idx]
                        # In the specific case where we have a categorical axis that's just a range, use 1 indexing instead of 0
                        print("scale" in ax_dict, ax_dict["scale"] in ["categorical", "datetime"], len(var_data[ax_name]) > 0, np.array_equal(var_data[ax_name], list(range(len(var_data[ax_name])))))
                        if "scale" in ax_dict and ax_dict["scale"] in ["categorical", "datetime"] and len(var_data[ax_name]) > 0 and \
                            np.array_equal(var_data[ax_name], list(range(len(var_data[ax_name])))):
                            ax_name_to_idx_labels[ax_name] += 1
            # Format idx labels along each axis
            idx_labels = list(ax_name_to_idx_labels.values())
            if len(idx_labels) >= 2:
                idxs_desc_arr.append(f"({', '.join([str(format_float(ptv, sig_figs)) for ptv in idx_labels])})")
            elif len(idx_labels) == 1 and i == 0:
                idxs_desc_arr.append(f"{list(ax_name_to_idx_labels.keys())[0]}={format_float(idx_labels[0], sig_figs)}")
            elif len(idx_labels) == 1:
                idxs_desc_arr.append(format_float(idx_labels[0], sig_figs))
            else:
                ax_dict = ax_info[ax_names[0]]
                if "ticklabels" in ax_dict and idx < len(ax_dict["ticklabels"]):
                    idx_val = ax_dict["ticklabels"][idx]
                else:
                    idx_val = var_data[ax_names[0]][idx]
                if not is_number(idx_val):
                    idxs_desc_arr.append(f"({idx_val})")
        print(idxs_desc_arr)
        return format_list(idxs_desc_arr)
    return ""