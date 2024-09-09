import utils_gen
from utils_gen import is_number
import numpy as np
from matplotalt import get_ax_ticks_scale


MONTH_STR_TO_NUM = {
    "Jan 31 ": "01/",
    "Feb 28 ": "02/",
    "Feb 29 ": "02/",
    "Mar 31 ": "03/",
    "Apr 30 ": "04/",
    "May 31 ": "05/",
    "Jun 30 ": "06/",
    "Jul 31 ": "07/",
    "Aug 31 ": "08/",
    "Sep 30 ": "09/",
    "Oct 31 ": "10/",
    "Nov 30 ": "11/",
    "Dec 31 ": "12/",
}


def safe_str(s):
    return s.replace('"','\"').replace("'","\'")


def modified_parse_all_dt(obj):
    chart_title = utils_gen.parse_title(obj)[1]
    chart_x = utils_gen.parse_axes(obj)[5]
    chart_y = utils_gen.parse_axes(obj)[11]
    chart_data_parsed = utils_gen.parse_marks_dt(obj, chart_x, chart_y)
    return chart_title, chart_x, chart_y, chart_data_parsed


def modified_parse_scales(obj):
    axes_scales = utils_gen.json_extract_value(obj, "axis-label", subtree=True)
    firstscale = axes_scales[0][1]['items']
    secondscale = axes_scales[1][1]['items']

    if (firstscale[0]['y'] == firstscale[-1]['y']) & (firstscale[0]['x'] != firstscale[-1]['x']):
        axes = [firstscale, secondscale]
    else:
        axes = [secondscale, firstscale]

    scale_strings = []
    for ax in axes:
        if is_number(ax[0]['text'].replace(',','')) and is_number(ax[1]['text'].replace(',','')) and is_number(ax[-1]['text'].replace(',','')):
            delta = float(ax[1]['text'].replace(',','')) - float(ax[0]['text'].replace(',',''))
            delta = round(delta) if delta.is_integer() else delta
            if (len(ax)-1)*delta == (float(ax[-1]['text'].replace(',','')) - float(ax[0]['text'].replace(',',''))):
                scale_strings.append(["linear", ax[0]['text'], ax[-1]['text']])
            else:
                scale_strings.append(["unknown", ax[0]['text'], ax[-1]['text']])
        else:
            scale_strings.append(["categorical", ax[0]['text'], ax[-1]['text']])
    return {'x-scale': scale_strings[0], 'y-scale': scale_strings[1]}


def parse_ax_scales(obj):
    axes_scales = utils_gen.json_extract_value(obj, "axis-label", subtree=True)
    firstscale = axes_scales[0][1]['items']
    secondscale = axes_scales[1][1]['items']
    #if (firstscale[0]['y'] == firstscale[-1]['y']) & (firstscale[0]['x'] != firstscale[-1]['x']):
    #    axes = [firstscale, secondscale]
    #else:
    #    axes = [secondscale, firstscale]

    first_ax_ticks = [s["text"].replace(',','') for s in firstscale]
    second_ax_ticks = [s["text"].replace(',','') for s in secondscale]
    first_type = get_ax_ticks_scale(first_ax_ticks)
    second_type = get_ax_ticks_scale(second_ax_ticks)
    #print(first_ax_ticks, first_type)
    #print(second_ax_ticks, second_type)
    return {'x-scale': [first_type, first_ax_ticks[0], first_ax_ticks[-1]],
            'y-scale': [second_type, second_ax_ticks[0], second_ax_ticks[-1]]}


def format_chart_data(chart_data, x_scale_type, y_scale_type):
    #print(x_scale_type, y_scale_type, chart_data)
    formatted_chart_data = []
    # If all entries along an axis start with Dec 31, replace with ""
    for xy_ax in [0, 1]:
        if len(chart_data) > 1 and not is_number(chart_data[0][xy_ax]) and \
            all(pt[xy_ax].startswith("Dec 31, ") for pt in chart_data):
            chart_data[:, 0] = [d.replace("Dec 31, ", "") for d in chart_data[:, 0]]

    for i, pt in enumerate(chart_data):
        formatted_pt = []
        for j, ele in enumerate(pt):
            if not is_number(ele):
                ele = ele.replace(",", "") # e.g. '250,000' -> '250000'
                for month_str in MONTH_STR_TO_NUM: # e.g. 'Dec 31, 2023' -> '12/2023'
                    ele = ele.replace(month_str, MONTH_STR_TO_NUM[month_str])
                if ";" in ele:
                    ele = ele.split(";")[0]
                ele = ele.replace("−", "-") # unicode symbol makes a difference when calling float!
                ele = ele.replace("*", "")
                ele = ele.strip()
            if is_number(ele):
                formatted_ele = float(ele)
                if formatted_ele.is_integer():
                    formatted_ele = int(formatted_ele)
            else:
                formatted_ele = ele
            if j == 0 and x_scale_type == "categorical":
                formatted_ele = str(formatted_ele)
            elif j == 1 and y_scale_type == "categorical":
                formatted_ele = str(formatted_ele)
            formatted_pt.append(formatted_ele)
        formatted_chart_data.append(formatted_pt)
    return formatted_chart_data


def generate_mpl_plot_code(chart_type, chart_data, chart_title,
                           x_label, y_label,
                           x_scale_type, y_scale_type,
                           x_tick_min, x_tick_max,
                           y_tick_min, y_tick_max):
    formatted_chart_data = format_chart_data(chart_data, x_scale_type, y_scale_type)
    formatted_x_data = [pt[0] for pt in formatted_chart_data]
    formatted_y_data = [pt[1] for pt in formatted_chart_data]

    if x_scale_type != "categorical" and y_scale_type != "categorical" and \
       is_number(formatted_x_data[0]) and is_number(formatted_y_data[0]):
        formatted_dtype = "float"
    else:
        formatted_dtype = "object"
    plt_code = f'chart_data = np.array({formatted_chart_data}, dtype={formatted_dtype})\n'
    plt_code += f'plt.title("{safe_str(chart_title)}")\n'
    plt_code += f'plt.xlabel("{safe_str(x_label)}")\n'
    plt_code += f'plt.ylabel("{safe_str(y_label)}")\n'

    if (not is_number(formatted_chart_data[0][0])):
        plt_code += "plt.xticks(rotation=90)\n"
    elif (is_number(formatted_chart_data[0][0]) and float(formatted_chart_data[0][0]) >= 1000):
        plt_code += "plt.xticks(rotation=45)\n"

    if x_scale_type == "categorical" and len(formatted_x_data) > 20:
        plt_code += "plt.xticks(fontsize=6)\n"
    if y_scale_type == "categorical" and len(formatted_y_data) > 20:
        plt_code += "plt.yticks(fontsize=6)\n"

    using_barh = False
    if chart_type == "line":
        plt_code += "plt.plot(chart_data[:, 0], chart_data[:, 1])\n"
        #plt_code += f"plt.xlim({x_tick_min}, {x_tick_max})\n"
        #plt_code += f"plt.ylim({y_tick_min}, {y_tick_max})\n"
    elif chart_type == "area":
        plt_code += "plt.plot(chart_data[:, 0], chart_data[:, 1])\n"
        if not is_number(formatted_x_data[0]):
            plt_code += "plt.fill_between(list(range(len(chart_data[:, 0]))), list(chart_data[:, 1]))\n"
        else:
            plt_code += "plt.fill_between(list(chart_data[:, 0]), list(chart_data[:, 1]))\n"
    elif chart_type == "bar":
        if y_scale_type == "categorical" or y_label.lower() in ["year", "month"]:
            using_barh = True
            plt_code += "plt.barh(chart_data[:, 1], chart_data[:, 0])\n"
        else:
            plt_code += "plt.bar(chart_data[:, 0], chart_data[:, 1])\n"
    elif chart_type == "scatter":
        plt_code += "plt.scatter(chart_data[:, 0], chart_data[:, 1])\n"

    #if not is_number(formatted_x_data[0]) and len(np.unique(formatted_x_data)) > 16:
    #    plt_code += "plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))\n"
    #if not is_number(formatted_y_data[0]) and len(np.unique(formatted_y_data)) > 16:
    #    plt_code += "plt.gca().yaxis.set_major_locator(plt.MaxNLocator(12))\n"

    # xlim doesn't work with barh
    if x_scale_type != "categorical" and not using_barh:
        try:
            min_x = np.min(formatted_x_data)
            max_x = np.max(formatted_x_data)
            x_tick_min = x_tick_min.replace(",", "")
            x_tick_max = x_tick_max.replace(",", "")
            x_tick_min = float(x_tick_min) if is_number(x_tick_min) else x_tick_min
            x_tick_max = float(x_tick_max) if is_number(x_tick_max) else x_tick_max
            if x_tick_min < min_x - np.abs(min_x) * 0.05:
                plt_code += f"plt.xlim(left={x_tick_min})\n"
            if x_tick_max > max_x + np.abs(max_x) * 0.05:
                plt_code += f"plt.xlim(right={x_tick_max})\n"
        except Exception as e:
            #print(formatted_x_data)
            #raise e
            #print("Unable to set xlim")
            pass

    if y_scale_type != "categorical":
        try:
            min_y = np.min(formatted_y_data)
            max_y = np.max(formatted_y_data)
            y_tick_min = y_tick_min.replace(",", "")
            y_tick_max = y_tick_max.replace(",", "")
            y_tick_min = float(y_tick_min) if is_number(y_tick_min) else y_tick_min
            y_tick_max = float(y_tick_max) if is_number(y_tick_max) else y_tick_max
            if y_tick_min < min_y - np.abs(min_y) * 0.05:
                plt_code += f"plt.ylim(bottom={y_tick_min})\n"
            if y_tick_max > max_y + np.abs(max_y) * 0.05:
                plt_code += f"plt.ylim(top={y_tick_max})\n"
        except Exception as e:
           #print("Unable to set ylim")
           pass

    if x_scale_type == "log-linear":
        plt_code += "plt.xscale(\'log\')"
    if y_scale_type == "log-linear":
        plt_code += "plt.yscale(\'log\')"

    return plt_code
