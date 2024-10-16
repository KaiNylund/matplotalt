import numpy as np
from copy import deepcopy
from scipy.stats import pearsonr
from collections import OrderedDict, defaultdict

from matplotalt_helpers import *
from matplotalt_constants import *
from stat_helpers import *
from trend_helpers import *


##################################################################################################
# Parent Chart Description Class
##################################################################################################
class ChartDescription():
    """
    The top-level class for chart descriptions. Has functions to generate descriptions for
    encodings, axes, annotations, and the overall chart without assuming the chart type.

    Attributes:
        ax (matplotlib.axes._subplots.AxesSubplot):
            The axis object for the chart to describe
        fig (matplotlib.figure.Figure, optional):
            The figure object for the chart to describe. If no figure is given, it is inferred from plt.gcf()
        chart_type (str, optional):
            The chart type of the figure to describe. Defaults to None
        sig_figs (int):
            The number of signifigant figures to use in chart descriptions. Defaults to 4
        max_color_desc_count (int):
            The max number of color encodings to include in chart descriptions. Defaults to 4

    Returns:
        None
    """
    def __init__(self, ax, fig=None, chart_type=None):
        """
        Initialize the ChartDescription object with the given attributes. Tries to infer x/y/z
        data and labels from the axis.
        """
        self.ax = ax
        if fig != None:
            self.fig = fig
        else:
            self.fig = plt.gcf()
        self.chart_dict = {}
        self.chart_dict["var_info"] = OrderedDict()
        self.chart_dict["ax_info"] = OrderedDict()
        self.chart_dict["annotations"] = []
        self.parse_title()
        self.parse_annotations()
        self.parse_axes()
        self.parse_encodings()
        # User specified fields
        self.chart_dict["chart_type"] = chart_type



    def parse_title(self):
        self.chart_dict["title"] = " ".join(self.ax.get_title().replace("\n", " ").strip().split())
        # Use suptitle if there's no regular title and only one subplot
        if self.chart_dict["title"] == "" and self.fig != None and self.fig.get_suptitle() != None and len(self.fig.get_axes()) == 1:
            self.chart_dict["title"] = " ".join(self.fig.get_suptitle().replace("\n", " ").strip().split())


    def parse_encodings(self, var_labels=None):
        # Infer labels and encoded objects from get_legend_handles_labels if possible
        legend_handles = None
        if self.ax.get_legend_handles_labels():
            legend_handles, legend_labels = self.ax.get_legend_handles_labels()
            if var_labels is None:
                var_labels = legend_labels
        # If we haven't initialized the label to encoding dict, populate it by mapping the legend
        # handle to its color (if possible)
        if var_labels:
            for i, label in enumerate(var_labels):
                if label not in self.chart_dict["var_info"]:
                    self.chart_dict["var_info"][label] = {}
                if legend_handles and i < len(legend_handles):
                    self.chart_dict["var_info"][label]["color"] = get_color_name(legend_handles[i].get_color())


    def parse_axes(self):
        for i, ax_name in enumerate(["x", "y", "z"]):
            ax_info_dict = {}
            # Check if the axes have x/y/z label getters
            if hasattr(self.ax, f"get_{ax_name}label") and callable(getattr(self.ax, f"get_{ax_name}label")):
                label_getter = getattr(self.ax, f"get_{ax_name}label")
                ticklabel_getter = getattr(self.ax, f"get_{ax_name}ticklabels")
                # Get label and ticklabels
                ax_info_dict["label"] = label_getter()
                ax_info_dict["ticklabels"] = [tl.get_text() for tl in ticklabel_getter()]
            # Try and get ax scale and range from ticklabels (e.g. linear, categorical, etc...)
            if "ticklabels" in ax_info_dict and len(ax_info_dict["ticklabels"]) > 0:
                ax_info_dict["scale"] = get_ax_ticks_scale(ax_info_dict["ticklabels"])
            # Try and get axis range
            if "ticklabels" in ax_info_dict and  len(ax_info_dict["ticklabels"]) > 0: # from ticklabels
                ax_info_dict["range"] = [ax_info_dict["ticklabels"][0], ax_info_dict["ticklabels"][-1]]
            elif i < len(self.ax.dataLim._points[0]): # from data lim
                ax_info_dict["range"] = [self.ax.dataLim._points[0][i], self.ax.dataLim._points[1][i]]
            # Add ax info to chart_dict
            if len(ax_info_dict) > 0:
                if ax_name not in self.chart_dict["ax_info"]:
                    self.chart_dict["ax_info"][ax_name] = ax_info_dict
                else:
                    self.chart_dict["ax_info"][ax_name].update(ax_info_dict)


    def parse_annotations(self):
        for child in self.ax.get_children():
            if isinstance(child, matplotlib.text.Annotation):
                self.chart_dict["annotations"].append({"text": child.get_text(), "coords": [child.xy[0], child.xy[1]]})


    # e.g. ax_name_to_data = {"x": [[...]...,[...]], "y": [[...]...,[...]]}
    # We assume that all data arrays will be of the same length on the first dimension
    # e.g. mark_type = "line"
    def parse_data(self, ax_name_to_data, mark_type):
        # populate data for each variable
        var_names = list(self.chart_dict["var_info"].keys())
        num_vars = len(var_names)
        num_data_objects = len(next(iter(ax_name_to_data.values())))
        for i in range(num_data_objects):
            # Use var labels from legend / prev inits if they can cover the number of lines
            cur_var_name = "the data"
            if i < num_vars:
                cur_var_name = var_names[i]
            # Otherwise use "variable {i}" or the ylabel or "the data"
            elif num_data_objects > 1:
                cur_var_name = f"{self.chart_dict['chart_type']} {i + 1}"
            # add var axis data to chart_dict
            if cur_var_name not in self.chart_dict["var_info"]:
                self.chart_dict["var_info"][cur_var_name] = {}
            if "data" not in self.chart_dict["var_info"][cur_var_name]:
                self.chart_dict["var_info"][cur_var_name]["data"] = OrderedDict()
            for ax_name, ax_data in ax_name_to_data.items():
                self.chart_dict["var_info"][cur_var_name]["data"][ax_name] = ax_data[i]
            self.chart_dict["var_info"][cur_var_name]["mark_type"] = mark_type
        # If axes don't have ranges or scales, try and parse them from data directly
        for ax_name, ax_dict in self.chart_dict["ax_info"].items():
            if "range" not in ax_dict or "scale" not in ax_dict:
                cur_ax_data = []
                for var_dict in self.chart_dict["var_info"].values():
                    if "data" in var_dict and ax_name in var_dict["data"]:
                        cur_ax_data.append(var_dict["data"][ax_name])
                if "range" not in ax_dict:
                    ax_dict["range"] = [np.nanmin(np.array(cur_ax_data)), np.nanmax(np.array(cur_ax_data))]
                elif "scale" not in ax_dict:
                    ax_dict["scale"] = get_ax_ticks_scale(np.array(cur_ax_data).flatten())



    def get_data_as_md_table(self, max_rows=20, sig_figs=4):
        if len(self.chart_dict["ax_info"]) > 0:
            table_dict = {}
            for ax_name, ax_dict in self.chart_dict["ax_info"].items():
                # Add data to table
                if len(ax_dict) != 0:
                    if "label" in ax_dict and ax_dict["label"].strip() != "":
                        ax_label = ax_dict["label"].strip()
                    else:
                        ax_label = ax_name

                    var_labels = list(self.chart_dict["var_info"].keys())
                    vars_cur_ax_data = []
                    for var_dict in self.chart_dict["var_info"].values():
                        if "data" in var_dict and ax_name in var_dict["data"]:
                            vars_cur_ax_data.append(var_dict["data"][ax_name])
                    data_len = len(vars_cur_ax_data[0])
                    if data_len > max_rows:
                        return "There are too many data points to fit in a table"
                    # If all variables share the same data for an axis, just include it once
                    if all([len(vars_cur_ax_data[i]) == len(vars_cur_ax_data[j]) and \
                       np.allclose(vars_cur_ax_data[i], vars_cur_ax_data[j]) \
                       for i in range(len(vars_cur_ax_data)) for j in range(i)]):
                        table_dict[ax_label] = vars_cur_ax_data[0]
                    # Otherwise include seperate columns for each variable / axis
                    else:
                        for i, l in enumerate(var_labels):
                            table_dict[f"{l} ({ax_label})"] = vars_cur_ax_data[i]
                    # Add axis tickslabels to table is possible
                    if "ticklabels" in self.chart_dict["ax_info"][ax_name] and \
                    data_len == len(self.chart_dict["ax_info"][ax_name]["ticklabels"]):
                        table_dict[f"{ax_label} ticklabels"] = self.chart_dict["ax_info"][ax_name]["ticklabels"]
            return create_md_table(table_dict, sig_figs=sig_figs)
        return ""


    def get_encodings_desc(self, max_color_desc_count=4, mark_type=None, sig_figs=4):
        """
        Return a description of the color encodings for each variable in the figure of the form:
        '{variable_name} is plotted in {variable_color}'
        If the number of variables to describe exceeds the max_color_desc_count, descriptions
        are of the form:
        '{num_variables} {object_name} are plotted for {[all variable names]}'
        (e.g. '12 groups of points are plotted for Jan, Feb,...')

        Returns:
            str: The description of each variable's color encoding
        """
        colors_desc = ""
        if len(self.chart_dict["var_info"]) > 0:
            if len(self.chart_dict["var_info"]) > max_color_desc_count:
                if mark_type == None:
                    mark_types = format_list(list(set([f"{vi['mark_type']}s" for vi in self.chart_dict["var_info"].items() if "mark_type" in vi])))
                else:
                    mark_types = f"{mark_type}s"
                if mark_types == None or len(mark_types) < 1:
                    mark_types = "variables"
                colors_desc += f"{len(self.chart_dict['var_info'])} {mark_types} are plotted"
                if len(self.chart_dict["var_info"]) < 16:
                    colors_desc += " for " + format_list(self.chart_dict["var_info"].keys())
            else:
                colors_desc_list = []
                for var_name, var_dict in self.chart_dict["var_info"].items():
                    if "color" in var_dict:
                        colors_desc_list.append(f"{var_name} is plotted in {var_dict['color']}")
                colors_desc += format_list(colors_desc_list)
        #else:
        #    warnings.warn(f"Chart is missing a legend or no labels were given")
        if colors_desc != "":
            colors_desc += "."
        return colors_desc.strip()


    def get_axes_desc(self, ax_names=None, sig_figs=4):
        """
        Return a description of the chart's axes in the form:
        '{ax_label} is plotted on the {ax_name}-axis from {min} to {max}.'
        If an axis does not have a label, descriptions are of the form:
        'the {ax_name} ranges from {min} to {max}'

        Attributes:
            ax_names (list[str], optional):
                The names of the axes to describe (e.g. ["x", "y", "z"]). If none are given, all
                axes with data will be described. Defaults to None

        Returns:
            str: The axes description
        """
        # If there are ticklabels, use them to get the range. Otherwise, use the dataLim or axis data
        axes_desc_arr = []
        num_axes = len(self.chart_dict["ax_info"])
        ax_to_scale = {ax_name: ax_dict["scale"] for ax_name, ax_dict in self.chart_dict["ax_info"].items() if "scale" in ax_dict}
        ax_share_a_scale = (num_axes > 1 and len(set(ax_to_scale.values())) == 1)
        cur_ax_names = ax_names if ax_names else list(self.chart_dict["ax_info"].keys())
        for ax_name in cur_ax_names:
            if ax_name in self.chart_dict["ax_info"]:
                cur_axis_desc = ""
                ax_dict = self.chart_dict["ax_info"][ax_name]
                ax_min = format_float(ax_dict['range'][0], sig_figs=sig_figs)
                ax_max = format_float(ax_dict['range'][1], sig_figs=sig_figs)
                # If there is an axis label, use it in the desc
                if "label" in ax_dict and ax_dict["label"] != "":
                    cur_axis_desc += f"{ax_dict['label']} is plotted on the {ax_name}-axis from {ax_min} to {ax_max}"
                else:
                    #warnings.warn(f"The {ax_name}-axis is missing a label")
                    cur_axis_desc += f"The {ax_name}-axis ranges from {ax_min} to {ax_max}"
                if not ax_share_a_scale and "scale" in ax_dict:
                    cur_axis_desc += f" using a {ax_dict['scale']} scale"
                cur_axis_desc = cur_axis_desc.strip()
                axes_desc_arr.append(cur_axis_desc)
        axes_desc = format_list(axes_desc_arr)
        if ax_share_a_scale:
            num_axs_word = "both" if num_axes == 2 else "all"
            axes_desc += f", {num_axs_word} using {next(iter(ax_to_scale.values()))} scales"
        if axes_desc != "":
            axes_desc += "."
        return axes_desc


    def get_annotations_desc(self, include_coords=False, max_annotations_desc=5, sig_figs=4):
        """
        Return descriptions of all annotations in the figure with the form:
        'An annotation reads: {annotation_text}'

        Attributes:
            include_coords (bool, optional):
                Whether the annotations coordinates should be included in the description.
                If True, descriptions add the line 'near x={ano_x_pos}, y={ano_y_pos}' before
                the annotation text.

        Returns:
            str: The descriptions of each annotation
        """
        annotations_desc = ""
        num_annotations = len(self.chart_dict["annotations"])
        if num_annotations > 0:
            if num_annotations == 1:
                annotations_desc += f"An annotation reads "
            else:
                annotations_desc += f"There are {num_annotations} text annotations. "
            if num_annotations <= max_annotations_desc:
                coords_desc_arr = []
                for annotation in self.chart_dict["annotations"]:
                    coords_desc = f"'{annotation['text']}'"
                    if include_coords:
                        ano_x = format_float(annotation['coords'][0], sig_figs=sig_figs)
                        ano_y = format_float(annotation['coords'][1], sig_figs=sig_figs)
                        coords_desc += f" at x={ano_x}, y={ano_y}"
                    coords_desc_arr.append(coords_desc)
                annotations_desc += format_list(coords_desc_arr)
        annotations_desc = annotations_desc.strip()
        if annotations_desc != "":
            annotations_desc += "."
        return annotations_desc



    def get_stats_desc(self, stats=[], max_var_stats=5, max_outliers_desc=4, stat_axis=None, sig_figs=4):
        """
        Given data and labels from the figure, return a description of the provided statistics along the
        given axis.

        Attributes:
            ax_name_to_data (dict[str -> array_like]):
                A dictionary mapping each axis name to its (n, d) data for each variable,
                e.g. {"x": [[1, 2, 3], [1, 2, 3]], "y": [[0.112, 0.415, 0.734], [0.225, 0.123, 0.849]]}
                If there is only one variable data can also be shape (d)
                e.g. {"x": [1, 2, 3], "y": [0.112, 0.415, 0.734]}

            ax_name_to_ticklabels (dict[str -> array_like]):
                A dictionary mapping each axis name to its (n, d) or (d) ticklabels

            stats (array_like):
                The statistics to compute and describe. Currently supported options include:
                ["min", "max", "mean", "median", "std", "outliers", "linearfit", "numpts"]

            var_idx (int, optional):
                The data index of the variable to describe in the given (n, d) axis data. If None
                is given, assumes data is of shape (d)

            stat_axis (str, optional):
                The axis to compute statistics along. E.g. if "y" is given, then statistics will
                be computed using the data in ax_name_to_data["y"]. Axes can also be specified
                for individual statistics by appending '_{ax_name}' to the stat name. For instance,
                'max_x' will compute the max along the x-axis.

            max_outliers_desc (int, optional):
                The maximum number of outlier points to list in descriptions. Defaults to 4

        Raises:
            ValueError: if given an unsupported statistic

        Returns:
            list[str]: A list of the descriptions of each given stat
        """
        if len(self.chart_dict["var_info"]) > max_var_stats:
            return ""

        stats_desc = ""
        # Parse a dict of stat_name -> axes based on stats input
        stat_name_to_axes = defaultdict(list)
        for stat in stats:
            stat = stat.split("_")
            stat_name = stat[0]
            # Users can either specify an axis by adding to the end of the stat (e.g. "max_x")
            # or by passing it in thought the stat_axis param
            cur_stat_axis = "x"
            if len(stat) == 2:
                cur_stat_axis = stat[1]
            elif stat_axis is not None:
                cur_stat_axis = stat_axis
            stat_name_to_axes[stat_name].append(cur_stat_axis)
        # Compute each of the stats along their corresponding axes
        for var_name in self.chart_dict["var_info"].keys():
            var_stats_desc_arr = []
            for stat_name, stat_axes in stat_name_to_axes.items():
                if stat_name in BASE_STAT_NAME_TO_FUNC:
                    axs_stats = ", ".join([BASE_STAT_NAME_TO_FUNC[stat_name](chart_dict=self.chart_dict,
                                                                             var_name=var_name,
                                                                             ax_name=ax_name,
                                                                             max_outliers_desc=max_outliers_desc,
                                                                             sig_figs=sig_figs) for ax_name in stat_axes])
                    var_stats_desc_arr.append(axs_stats)
                else:
                    raise ValueError(f"Statistic {stat_name} is not supported for the current chart type")
            stats_desc += f"{var_name.capitalize()} has {format_list(var_stats_desc_arr)}. "
        return stats_desc.strip()



    def get_trends_desc(self, trends=[], max_var_trends=5, trend_axis=None, sig_figs=4):
        """

        """
        if len(self.chart_dict["var_info"]) > max_var_trends:
            return ""

        trends_desc = ""
        # Parse a dict of trend_name -> axes based on trends input
        trend_name_to_axes = defaultdict(list)
        for trend in trends:
            trend = trend.split("_")
            trend_name = trend[0]
            # Users can either specify an axis by adding to the end of the stat (e.g. "max_x")
            # or by passing it in thought the stat_axis param
            cur_trend_axis = "y"
            if len(trend) == 2:
                cur_trend_axis = trend[1]
            elif trend_axis is not None:
                cur_trend_axis = trend_axis
            trend_name_to_axes[trend_name].append(cur_trend_axis)
        # Compute each of the stats along their corresponding axes
        for var_name in self.chart_dict["var_info"].keys():
            var_trends_desc_arr = []
            for trend_name, trend_axes in trend_name_to_axes.items():
                if trend_name in BASE_TREND_NAME_TO_FUNC:
                    var_trends_desc_arr.extend([BASE_TREND_NAME_TO_FUNC[trend_name](chart_dict=self.chart_dict,
                                                                             var_name=var_name,
                                                                             ax_name=ax_name,
                                                                             sig_figs=sig_figs) for ax_name in trend_axes])
                else:
                    raise ValueError(f"Trend {trend_name} is not supported for the current chart type")
        var_trends_desc_arr = [vt for vt in var_trends_desc_arr if vt != ""]
        trends_desc += ". ".join(var_trends_desc_arr)
        if trends_desc != "":
            trends_desc += "."
        return trends_desc.strip()



    def get_chart_type_desc(self):
        """
        Return a description of the current chart type of the form:
        'A {formatted chart_type} titled {chart_title}'
        """
        chart_type_desc = ""
        if self.chart_dict["chart_type"] in CHART_TYPE_TO_DESC:
            chart_type_desc = CHART_TYPE_TO_DESC[self.chart_dict["chart_type"]]
        else:
            chart_type_desc = f"A {self.chart_dict['chart_type']}"
        if self.chart_dict["title"] != "":
            chart_type_desc += f" titled \'{self.chart_dict['title']}\'"
        chart_type_desc += ". "
        return chart_type_desc


    def get_chart_desc(self, desc_level=2, **kwargs):
        """
        Return a description of the chart of the form:

        '{chart type + title description}
         {axes description}
         {encodings description}
         {annotations description}
         {statistics description}
         {trends description}'

        based on the given description level
        """
        desc_config = deepcopy(DEFAULT_DESC_CONFIG)
        desc_config.update(kwargs)
        #print(self.chart_dict)
        alt_text_arr = []
        alt_text_arr.append(self.get_chart_type_desc())
        # Add axis and encoding descriptions
        if desc_level > 0:
            alt_text_arr.append(self.get_axes_desc(sig_figs=desc_config["sig_figs"]))
            alt_text_arr.append(self.get_encodings_desc(max_color_desc_count=desc_config["max_color_desc_count"], sig_figs=desc_config["sig_figs"]))
        alt_text_arr.append(self.get_annotations_desc(include_coords=desc_config["include_annotation_coords"], sig_figs=desc_config["sig_figs"]))
        # Add stats
        if desc_level > 1:
            # if stats is None, use the default stats from the child class
            if desc_config["stats"] and len(desc_config["stats"]) > 0:
                alt_text_arr.append(self.get_stats_desc(stats=desc_config["stats"], max_var_stats=desc_config["max_var_stats"], max_outliers_desc=desc_config["max_outliers_desc"], sig_figs=desc_config["sig_figs"]).strip().capitalize())
            else:
                alt_text_arr.append(self.get_stats_desc(max_var_stats=desc_config["max_var_stats"], max_outliers_desc=desc_config["max_outliers_desc"], sig_figs=desc_config["sig_figs"]))
        # Add trends if applicable
        if desc_level > 2:
            if desc_config["trends"] and len(desc_config["trends"]) > 0:
                alt_text_arr.append(self.get_trends_desc(trends=desc_config["trends"], max_var_trends=desc_config["max_var_trends"], sig_figs=desc_config["sig_figs"]).strip().capitalize())
            else:
                alt_text_arr.append(self.get_trends_desc(sig_figs=desc_config["sig_figs"]))

        alt_text_arr = [al.strip().capitalize() for al in alt_text_arr if al]
        alt_text_arr = [al for al in alt_text_arr if al]
        alt_text = " ".join(alt_text_arr)
        alt_text.replace(r'\s+', r'\s')
        #alt_text = insert_line_breaks(alt_text, max_line_width=desc_config["max_line_width"])
        return alt_text


    def get_chart_dict(self):
        return deepcopy(self.chart_dict)


##################################################################################################
# Area Plot Description
##################################################################################################
class AreaDescription(ChartDescription):
    """
    The class for generating area plot descriptions. Has functions to automatically generate
    descriptions for encodings, axes, annotations, statistics, trends and the overall chart.
    Infers chart data, encodings, etc... from the figure attributes.

    Attributes:
        ax (matplotlib.axes._subplots.AxesSubplot):
            The axis object for the chart to describe
        fig (matplotlib.figure.Figure, optional):
            The figure object for the chart to describe. If no figure is given, it is inferred
            from plt.gcf()
        **kwargs (optional):
            Used to manually specify chart_type, data, and other details of chart descriptions
            including number of signifigant figures, max line width, etc...
    """
    def __init__(self, ax, fig=None, **kwargs):
        """
        Initialize the AreaDescription with the given attributes. Infers x / y data,
        vertical/horizontal lines, and labels from the axis.
        """
        super().__init__(ax, fig, chart_type="area", **kwargs)
        self.lines = self.ax.get_lines()
        if len(self.lines) < 1:
            raise ValueError("Area plot contains no areas")
        self.vline_xs = []
        self.hline_ys = []
        # x and y that don't belong to vertical or horizontal lines
        x = []
        y = []
        constant_line_idxs = []
        for line in self.lines:
            cur_xs = line._xy[:, 0]
            cur_ys = line._xy[:, 1]
            if np.all(np.isclose(cur_xs, cur_xs[0])):
                self.vline_xs.append(cur_xs[0])
                constant_line_idxs.append(i)
            elif np.all(np.isclose(cur_ys, cur_ys[0])):
                self.hline_ys.append(cur_ys[0])
                constant_line_idxs.append(i)
            else:
                x.append(line._xy[:, 0])
                y.append(line._xy[:, 1])
        # Remove duplicate vertical / horizontal lines
        self.vline_xs = np.unique(self.vline_xs)
        self.hline_ys = np.unique(self.hline_ys)
        # Unless all lines are horizontal / vertical,
        # remove lines that are constant on one axis so they aren't included in stats
        if len(constant_line_idxs) < len(self.lines):
            for i in constant_line_idxs:
                del self.lines[i]
        # populate data in chart_dict for each variable
        self.parse_data({"x": x, "y": y}, mark_type="point")



    def get_encodings_desc(self, max_color_desc_count=4, sig_figs=4, **kwargs):
        """
        See :func:`~ChartDescription.get_stats_desc`. Additionally includes
        descriptions of any vertical and horizontal lines in the form:
        'There are vertical lines at x={vertical_line_xs}'
        """
        encodings_desc = super().get_encodings_desc(max_color_desc_count=max_color_desc_count, **kwargs)
        if encodings_desc == "" and len(self.lines) == 1 and max_color_desc_count > 0:
            line = self.lines[0]
            encodings_desc += f" The data are plotted in {LINE_STYLE_TO_DESC[line.get_linestyle()]}{get_color_name(line._color)}. "
        if len(self.vline_xs) == 1:
            encodings_desc += f" There is a vertical line at x={format_float(self.vline_xs[0], sig_figs=sig_figs)}. "
        elif len(self.vline_xs) > 1:
            encodings_desc += f" There are vertical lines at x={format_float_list(self.vline_xs, sig_figs=sig_figs)}. "
        if len(self.hline_ys) == 1:
            encodings_desc += f" There is a horizontal line at y={format_float(self.hline_ys[0], sig_figs=sig_figs)}. "
        elif len(self.hline_ys) > 1:
            encodings_desc += f" There are horizontal lines at y={format_float_list(self.hline_ys, sig_figs=sig_figs)}. "
        return encodings_desc


    def get_stats_desc(self, stats=["min_y", "max_y", "mean_y"], stat_axis="y", **kwargs):
        """See :func:`~ChartDescription.get_stats_desc`"""
        return super().get_stats_desc(stats, stat_axis=stat_axis, **kwargs)


    def get_trends_desc(self, trends=["shape_y", "correlation_y"], trend_axis="y", **kwargs):
        """See :func:`~ChartDescription.get_trends_desc`"""
        return super().get_trends_desc(trends=trends, trend_axis=trend_axis, **kwargs)



##################################################################################################
# Bar Chart Description
##################################################################################################
class BarDescription(ChartDescription):

    """
    The class for generating bar chart / histogram descriptions. Has functions to automatically
    generate descriptions for encodings, axes, annotations, statistics, trends and the overall
    chart. Infers chart data, encodings, etc... from the figure attributes.

    Attributes:
        ax (matplotlib.axes._subplots.AxesSubplot):
            The axis object for the chart to describe
        fig (matplotlib.figure.Figure, optional):
            The figure object for the chart to describe. If no figure is given, it is inferred
            from plt.gcf()
        **kwargs (optional):
            Used to manually specify chart_type, data, and other details of chart descriptions
            including number of signifigant figures, max line width, etc...
    """
    def __init__(self, ax, fig=None, **kwargs):
        """
        Initialize the BarDescription with the given attributes. Infers x / y data
        and labels from the axis.
        """
        super().__init__(ax, fig, chart_type="bar", **kwargs)
        self.bars = self.ax.containers
        if len(self.bars) < 1:
            raise ValueError("Bar chart contains no bars")

        bar_values = [b.datavalues for b in self.bars]
        bar_ticks = [list(range(len(bv))) for bv in bar_values]
        self.cat_axis = "x"
        self.num_axis = "y"
        if self.chart_dict["ax_info"]["y"]["scale"] in ["categorical", "datetime"]:
            self.cat_axis = "y"
            self.num_axis = "x"
        self.parse_data({self.cat_axis: bar_ticks, self.num_axis: bar_values}, mark_type="point")


    def get_encodings_desc(self, **kwargs):
        """See :func:`~ChartDescription.get_encodings_desc`"""
        return super().get_encodings_desc(**kwargs)


    def get_stats_desc(self, stats=["numpts", "min", "max", "mean"], stat_axis=None, **kwargs):
        """See :func:`~ChartDescription.get_stats_desc`"""
        if stat_axis is None:
            stat_axis = self.num_axis
        return super().get_stats_desc(stats, stat_axis=stat_axis, **kwargs)


    def get_trends_desc(self, trends=["shape", "correlation"], trend_axis=None, **kwargs):
        """See :func:`~ChartDescription.get_trends_desc`"""
        if trend_axis is None:
            trend_axis = self.num_axis
        return super().get_trends_desc(trends=trends, trend_axis=trend_axis, **kwargs)



##################################################################################################
# Boxplot Description
##################################################################################################
class BoxplotDescription(ChartDescription):
    """
    The class for generating boxplot descriptions. Has functions to automatically generate
    descriptions for encodings, axes, annotations, statistics, trends and the overall chart.
    Infers chart data, encodings, etc... from the figure attributes.

    Attributes:
        ax (matplotlib.axes._subplots.AxesSubplot):
            The axis object for the chart to describe
        fig (matplotlib.figure.Figure, optional):
            The figure object for the chart to describe. If no figure is given, it is inferred
            from plt.gcf()
        **kwargs (optional):
            Used to manually specify chart_type, data, and other details of chart descriptions
            including number of signifigant figures, max line width, etc...
    """
    def __init__(self, ax, fig=None, **kwargs):
        """
        Initialize the BoxplotDescription with the given attributes. Infers each box's
        quartiles and outliers from the axis
        """
        super().__init__(ax, fig, chart_type="boxplot", **kwargs)
        self.box_num_to_quartiles = OrderedDict()
        self.box_patches = []

        self.vert_idx = 1 # vertical
        if self.chart_dict["ax_info"]["y"]["scale"] in ["categorical", "datetime"]:
            self.vert_idx = 0 # horizontal

        # Code for parsing quartiles and outliers from matplotlib objects:
        # Basic assumption is that each box will consist of seven objects in the form
        # [PathPatch, Line * 6]. The patch will contain info about the median and middle
        # two quartiles, and the lines will contain the outer quartiles and outliers.
        ax_children = self.ax.get_children()
        cur_idx = 0
        cur_box_num = 0
        # Iterate over all children
        while cur_idx < len(ax_children):
            # If the current child is a PathPatch, then check if the next six elements
            # are Lines and extract the quartiles
            if isinstance(ax_children[cur_idx], matplotlib.patches.PathPatch):
                is_box = True
                # for vertical boxplots, the quartiles are stored in the y coordinate of the patch
                # paths. TODO: update this to work with horizintal boxplots as well
                cur_patch = ax_children[cur_idx]
                cur_box_children = set(cur_patch._path._vertices[:, self.vert_idx][:-1])
                cur_box_outliers = set()
                for i in range(1, 7):
                    cur_patch_line = ax_children[cur_idx + i]
                    if (cur_idx + i >= len(ax_children)) or \
                        not isinstance(cur_patch_line, matplotlib.lines.Line2D):
                        is_box = False
                        break
                    else:
                        cur_vertices = cur_patch_line._path._vertices[:, self.vert_idx]
                        # Assume the last line of the six stores the coordinates of outliers
                        if i == 6:
                            cur_box_outliers.update(cur_vertices)
                        else:
                            cur_box_children.update(cur_vertices)
                if is_box:
                    self.box_num_to_quartiles[cur_box_num] = {"quartiles": sorted(cur_box_children),
                                                              "outliers": sorted(cur_box_outliers)}
                    self.box_patches.append(cur_patch)
                    cur_box_num += 1
                    cur_idx += 7
                else:
                    cur_idx += 1
            else:
                cur_idx += 1
        # If there are the same number of ticklabels as boxes, use them for the box labels
        if self.vert_idx == 1 and "ticklabels" in self.chart_dict["ax_info"]["x"] and \
           len(self.chart_dict["ax_info"]["x"]["ticklabels"]) == len(self.box_num_to_quartiles):
            for box_num in list(self.box_num_to_quartiles.keys()):
                box_label = self.chart_dict["ax_info"]["x"]["ticklabels"][box_num]
                self.box_num_to_quartiles[box_label] = self.box_num_to_quartiles.pop(box_num)
        elif self.vert_idx == 0 and "ticklabels" in self.chart_dict["ax_info"]["y"] and \
           len(self.chart_dict["ax_info"]["y"]["ticklabels"]) == len(self.box_num_to_quartiles):
            for box_num in list(self.box_num_to_quartiles.keys()):
                box_label = self.chart_dict["ax_info"]["y"]["ticklabels"][box_num]
                self.box_num_to_quartiles[box_label] = self.box_num_to_quartiles.pop(box_num)
        # err if we can't infer boxplot values
        if len(self.box_num_to_quartiles) < 1:
            raise ValueError("Unable to infer boxplot values")


    def get_data_as_md_table(self, **kwargs):
        return f"Tables are currently unsupported for charts of type: {self.chart_dict['chart_type']}"


    def get_axes_desc(self, **kwargs):
        """See :func:`~ChartDescription.get_axes_desc`"""
        return super().get_axes_desc(ax_names=["x", "y"], **kwargs)


    def get_encodings_desc(self, **kwargs):
        """See :func:`~ChartDescription.get_encodings_desc`"""
        # If we haven't initialized the label to encoding dict, populate it by mapping the legend
        # handle to its color (if possible)
        #if (len(self.label_to_encoding) == 0) and self.labels and (len(self.labels) > 0) and self.legend_handles:
        #    for i, label in enumerate(self.labels):
        #        self.label_to_encoding[label] = get_color_name(self.legend_handles[i]._facecolors[0])
        return super().get_encodings_desc(mark_type="box", **kwargs)



    def get_stats_desc(self, stats=["median", "iqr", "outliers"], max_outliers_desc=4, sig_figs=4, **kwargs):
        """
        Return a description of the provided statistics for each box.

        Attributes:
            stats (array_like):
                The statistics to compute and describe. Currently supported options include:
                ["median", "quartiles", "iqr", "outliers"]

            max_outliers_desc (int, optional):
                The maximum number of outlier points to list in descriptions. Defaults to 4

        Returns:
            str: The descriptions of each given stat
        """
        # TODO: raise valueerror if given an unsupported stat
        stats_desc = ""
        for box_label, box_quartiles in self.box_num_to_quartiles.items():
            boxplot_stats_desc_arr = []
            if "median" in stats:
                boxplot_stats_desc_arr.append(f"a median of {box_quartiles['quartiles'][2]}")
            if "quartiles" in stats:
                boxplot_stats_desc_arr.append(f"quartiles Q0-Q5 with values {format_float_list(box_quartiles['quartiles'], sig_figs=sig_figs)}")
            if "iqr" in stats:
                boxplot_stats_desc_arr.append(f"an interquartile range of {format_float(box_quartiles['quartiles'][3] - box_quartiles['quartiles'][1], sig_figs=sig_figs)}")
            if "outliers" in stats:
                outlier_pts = box_quartiles["outliers"]
                outlier_word = "outlier" if len(outlier_pts) == 1 else "outliers"
                if len(outlier_pts) == 0:
                    boxplot_stats_desc_arr.append("no outliers")
                elif len(outlier_pts) > max_outliers_desc:
                    boxplot_stats_desc_arr.append(f"{len(outlier_pts)} {outlier_word} along the y-axis")
                else:
                    boxplot_stats_desc_arr.append(f"{len(outlier_pts)} {outlier_word} at y={format_float_list(outlier_pts, sig_figs=sig_figs)}")
            stats_desc += f"Boxplot {box_label} has {format_list(boxplot_stats_desc_arr)}. "
        return stats_desc


    def get_trends_desc(self, **kwargs):
        """TODO"""
        return ""



##################################################################################################
# Contour Plot Description
##################################################################################################
class ContourDescription(ChartDescription):
    """
    The class for generating contour plot descriptions. Has functions to automatically generate
    descriptions for encodings, axes, annotations, statistics, trends and the overall chart.
    Infers chart data, encodings, etc... from the figure attributes.

    Attributes:
        ax (matplotlib.axes._subplots.AxesSubplot):
            The axis object for the chart to describe
        fig (matplotlib.figure.Figure, optional):
            The figure object for the chart to describe. If no figure is given, it is inferred
            from plt.gcf()
        **kwargs (optional):
            Used to manually specify chart_type, data, and other details of chart descriptions
            including number of signifigant figures, max line width, etc...
    """
    def __init__(self, ax, fig=None, **kwargs):
        super().__init__(ax, fig, chart_type="contour", **kwargs)
        self.contour_set = self.ax._children[0]
        if not isinstance(self.contour_set, matplotlib.contour.QuadContourSet):
            raise ValueError("Unable to parse contour lines from chart")
        self.level_values = self.contour_set.labelCValues
        self.level_labels = self.contour_set.labelTexts
        if self.level_values is None or len(self.level_values) < 1:
            raise ValueError("Contour plot is missing contour values")
        self.level_centers = []
        for path in self.contour_set._paths[1:-1]:
            self.level_centers.append(np.mean(path._vertices, axis=0))
        self.parse_data({"x": [], "y": []}, mark_type="line")


    def get_data_as_md_table(self, **kwargs):
        return f"Tables are currently unsupported for charts of type: {self.chart_type}"


    def get_encodings_desc(self, **kwargs):
        """See :func:`~ChartDescription.get_encodings_desc`"""
        if len(self.level_labels) == len(self.level_values):
            cur_level_labels = [l.get_text() for l in self.level_labels]
        else:
            cur_level_labels = self.level_values
        return f"{len(cur_level_labels)} contour lines are plotted with values {format_list(cur_level_labels)}."


    def get_stats_desc(self, stats=["max_center"], sig_figs=4, **kwargs):
        """
        Return a description of the provided statistics for each box.

        Attributes:
            stats (array_like):
                The statistics to compute and describe. Currently supported options include:
                ["centers"]

        Returns:
            str: The descriptions of each given stat
        """
        # TODO: raise valueerror if given an unsupported stat
        stats_desc = ""
        if "max_center" in stats:
            max_center = f"({format_float(self.level_centers[-1][0], sig_figs=sig_figs)}, {format_float(self.level_centers[-1][1], sig_figs=sig_figs)})"
            stats_desc += f"The max contour is centered around {max_center}."
        return stats_desc


    def get_trends_desc(self, **kwargs):
        return ""



##################################################################################################
# Heatmap Description
##################################################################################################
class HeatmapDescription(ChartDescription):
    """
    The class for generating heatmap descriptions. Has functions to automatically generate
    descriptions for encodings, axes, annotations, statistics, trends and the overall chart.
    Infers chart data, encodings, etc... from the figure attributes.

    Attributes:
        ax (matplotlib.axes._subplots.AxesSubplot):
            The axis object for the chart to describe
        fig (matplotlib.figure.Figure, optional):
            The figure object for the chart to describe. If no figure is given, it is inferred
            from plt.gcf()
        **kwargs (optional):
            Used to manually specify chart_type, data, and other details of chart descriptions
            including number of signifigant figures, max line width, etc...
    """
    def __init__(self, ax, fig=None, **kwargs):
        """
        Initialize the HeatmapDescription with the given attributes. Infers x / y / z data
        and labels from the axis.
        """
        super().__init__(ax, fig, chart_type="heatmap", **kwargs)
        self.quadmesh = None
        for ax_coll in self.ax.collections:
            if isinstance(ax_coll, matplotlib.collections.QuadMesh):
                self.quadmesh = ax_coll
        if self.quadmesh is None:
            raise ValueError("Heatmap axis does not contain a QuadMesh")
        coords = self.quadmesh._coordinates[:-1, :-1, :]
        self.shape = coords.shape[:2]
        coords = coords.reshape(-1, 2)
        x = coords[:, 0]
        y = coords[:, 1]
        z = self.quadmesh._A.flatten()
        if len(x) < 1 or len(y) < 1:
            raise ValueError("Heatmap cells are missing coordinates")
        elif len(z) < 1:
            raise ValueError("Heatmap cells are missing values")
        x = np.tile(x, self.shape[1])
        y = np.repeat(y, self.shape[0])
        if self.quadmesh.colorbar is not None:
            if "z" not in self.chart_dict["ax_info"]:
                self.chart_dict["ax_info"]["z"] = {}
            self.chart_dict["ax_info"]["z"]["label"] = self.quadmesh.colorbar.ax.get_ylabel()
            self.chart_dict["ax_info"]["z"]["ticklabels"] = [tl.get_text() for tl in self.quadmesh.colorbar.ax.get_yticklabels()]
            if len(self.chart_dict["ax_info"]["z"]["ticklabels"]) > 0:
                self.chart_dict["ax_info"]["z"]["scale"] = get_ax_ticks_scale(self.chart_dict["ax_info"]["z"]["ticklabels"])
        # add data to chart_dict
        self.parse_data({"x": [x], "y": [y], "z": [z]}, mark_type="cell")


    def get_chart_type_desc(self):
        """
        Return a description of the current heatmap title of the form:
        'A {heatmap_width}x{heatmap_height} titled {heatmap_title}'
        """
        chart_type_desc = f"A {self.shape[1]}x{self.shape[0]} heatmap"
        if self.chart_dict["title"] != "":
            chart_type_desc += f" titled \'{self.chart_dict['title']}\'"
        chart_type_desc += ". "
        return chart_type_desc


    def get_stats_desc(self, stats=["min_z", "max_z", "mean_z"], stat_axis="z", **kwargs):
        """See :func:`~ChartDescription.get_stats_desc`"""
        return super().get_stats_desc(stats=stats, stat_axis=stat_axis, **kwargs)



##################################################################################################
# Image Description
##################################################################################################
class ImageDescription(ChartDescription):
    """
    The class for generating image descriptions (e.g. those created with plt.imshow).
    Has functions to automatically generate descriptions for encodings, axes, annotations,
    statistics, trends and the overall chart. Infers chart data, encodings, etc...
    from the figure attributes.

    Attributes:
        ax (matplotlib.axes._subplots.AxesSubplot):
            The axis object for the chart to describe
        fig (matplotlib.figure.Figure, optional):
            The figure object for the chart to describe. If no figure is given, it is inferred
            from plt.gcf()
        **kwargs (optional):
            Used to manually specify chart_type, data, and other details of chart descriptions
            including number of signifigant figures, max line width, etc...
    """
    def __init__(self, ax, fig=None, **kwargs):
        """
        Initialize the ImageDescription with the given attributes. Infers x / y / z data
        and labels from the axis.
        """
        super().__init__(ax, fig, chart_type="image", **kwargs)
        self.ax_img = None
        for ax_child in self.ax.get_children():
            if isinstance(ax_child, matplotlib.image.AxesImage):
                self.ax_img = ax_child
        if self.ax_img is None:
            raise ValueError("Image axis does not contain an AxesImage object")
        data = self.ax_img._A
        self.shape = data.shape
        x, y = np.indices(self.shape)
        x = x.ravel(order='F')
        y = y.ravel(order='F')
        z = data.ravel(order='F')
        if len(x) < 1 or len(y) < 1:
            raise ValueError("Image is missing coordinates")
        elif len(z) < 1:
            raise ValueError("Image is missing pixel values")
        if self.ax_img.colorbar is not None:
            if "z" not in self.chart_dict["ax_info"]:
                self.chart_dict["ax_info"]["z"] = {}
            self.chart_dict["ax_info"]["z"]["label"] = self.ax_img.colorbar.ax.get_ylabel()
            self.chart_dict["ax_info"]["z"]["ticklabels"] = [tl.get_text() for tl in self.ax_img.colorbar.ax.get_yticklabels()]
            if len(self.chart_dict["ax_info"]["z"]["ticklabels"]) > 0:
                self.chart_dict["ax_info"]["z"]["scale"] = get_ax_ticks_scale(self.chart_dict["ax_info"]["z"]["ticklabels"])
        # add data to chart_dict
        self.parse_data({"x": [x], "y": [y], "z": [z]}, mark_type="pixel")


    def get_chart_type_desc(self):
        """
        Return a description of the current heatmap title of the form:
        'A {heatmap_width}x{heatmap_height} titled {heatmap_title}'
        """
        chart_type_desc = f"A {self.shape[1]}x{self.shape[0]} image"
        if self.chart_dict["title"] != "":
            chart_type_desc += f" titled \'{self.chart_dict['title']}\'"
        chart_type_desc += ". "
        return chart_type_desc


    def get_stats_desc(self, stats=["min_z", "max_z", "mean_z"], stat_axis="z", **kwargs):
        """See :func:`~ChartDescription.get_stats_desc`"""
        return super().get_stats_desc(stats=stats, stat_axis=stat_axis, **kwargs)



##################################################################################################
# Line Plot Description
##################################################################################################
class LineDescription(ChartDescription):
    """
    The class for generating line plot descriptions. Has functions to automatically generate
    descriptions for encodings, axes, annotations, statistics, trends and the overall chart.
    Infers chart data, encodings, etc... from the figure attributes.

    Attributes:
        ax (matplotlib.axes._subplots.AxesSubplot):
            The axis object for the chart to describe
        fig (matplotlib.figure.Figure, optional):
            The figure object for the chart to describe. If no figure is given, it is inferred
            from plt.gcf()
        **kwargs (optional):
            Used to manually specify chart_type, data, and other details of chart descriptions
            including number of signifigant figures, max line width, etc...
    """
    def __init__(self, ax, fig=None, **kwargs):
        """
        Initialize the LineDescription with the given attributes. Infers x / y data,
        vertical/horizontal lines, and labels from the axis.
        """
        # Radial line chart
        if isinstance(ax, matplotlib.projections.polar.PolarAxes):
            super().__init__(ax, fig, chart_type="radial line", **kwargs)
            self.radial = True
        else:
            super().__init__(ax, fig, chart_type="line", **kwargs)
            self.radial = False
        self.lines = self.ax.get_lines()
        if len(self.lines) < 1:
            raise ValueError("Line plot contains no lines")
        self.vline_xs = []
        self.hline_ys = []
        # x and y that don't belong to vertical or horizontal lines
        x = []
        y = []
        constant_line_idxs = []
        for i, line in enumerate(self.lines):
            cur_xs = line._xy[:, 0]
            cur_ys = line._xy[:, 1]
            if np.all(np.isclose(cur_xs, cur_xs[0])):
                self.vline_xs.append(cur_xs[0])
                constant_line_idxs.append(i)
            elif np.all(np.isclose(cur_ys, cur_ys[0])):
                self.hline_ys.append(cur_ys[0])
                constant_line_idxs.append(i)
            else:
                if self.radial: # Skip last repeated index in radial line plots
                    x.append(line._xy[:-1, 0])
                    y.append(line._xy[:-1, 1])
                else:
                    x.append(line._xy[:, 0])
                    y.append(line._xy[:, 1])
        # Remove duplicate vertical / horizontal lines
        self.vline_xs = np.unique(self.vline_xs)
        self.hline_ys = np.unique(self.hline_ys)
        # Unless all lines are horizontal / vertical,
        # remove lines that are constant on one axis so they aren't included in stats
        if len(constant_line_idxs) < len(self.lines):
            for i in constant_line_idxs:
                del self.lines[i]
        # populate data in chart_dict for each variable
        self.parse_data({"x": x, "y": y}, mark_type="line")


    def get_encodings_desc(self, max_color_desc_count=4, sig_figs=4, **kwargs):
        """
        See :func:`~ChartDescription.get_stats_desc`. Additionally includes
        descriptions of any vertical and horizontal lines in the form:
        'There are vertical lines at "x"={vertical_line_xs}'
        """
        encodings_desc = super().get_encodings_desc(max_color_desc_count=max_color_desc_count, **kwargs)
        if encodings_desc == "" and len(self.lines) == 1 and max_color_desc_count > 0:
            line = self.lines[0]
            encodings_desc += f" The data are plotted in {LINE_STYLE_TO_DESC[line.get_linestyle()]}{get_color_name(line._color)}. "
        if len(self.vline_xs) == 1:
            encodings_desc += f" There is a vertical line at x={self.vline_xs[0]}. "
        elif len(self.vline_xs) > 1:
            encodings_desc += f" There are vertical lines at x={format_float_list(self.vline_xs, sig_figs=sig_figs)}. "
        if len(self.hline_ys) == 1:
            encodings_desc += f" There is a horizontal line at y={self.hline_ys[0]}. "
        elif len(self.hline_ys) > 1:
            encodings_desc += f" There are horizontal lines at y={format_float_list(self.hline_ys, sig_figs=sig_figs)}. "
        return encodings_desc


    def get_stats_desc(self, stats=["min_y", "max_y", "mean_y"], stat_axis="y", **kwargs):
        """See :func:`~ChartDescription.get_stats_desc`"""
        return super().get_stats_desc(stats=stats, stat_axis=stat_axis, **kwargs)


    def get_trends_desc(self, trends=["shape_y", "correlation_y"], trend_axis="y", **kwargs):
        """See :func:`~ChartDescription.get_trends_desc`"""
        return super().get_trends_desc(trends=trends, trend_axis=trend_axis, **kwargs)



##################################################################################################
# Pie Chart Description
##################################################################################################
class PieDescription(ChartDescription):
    """
    The class for generating pie chart descriptions. Has functions to automatically generate
    descriptions for encodings, axes, annotations, statistics, trends and the overall chart.
    Infers chart data, encodings, etc... from the figure attributes.

    Attributes:
        ax (matplotlib.axes._subplots.AxesSubplot):
            The axis object for the chart to describe
        fig (matplotlib.figure.Figure, optional):
            The figure object for the chart to describe. If no figure is given, it is inferred
            from plt.gcf()
        max_slices_desc (int, optional):
            The maximum number of slices to name in the description. Defaults to 15.
        **kwargs (optional):
            Used to manually specify chart_type, data, and other details of chart descriptions
            including number of signifigant figures, max line width, etc...
    """
    def __init__(self, ax, fig=None, max_slices_desc=15, **kwargs):
        """
        Initialize the PieDescription with the given attributes. Infers wedges and wedge widths,
        and labels from the axis.
        """
        super().__init__(ax, fig, chart_type="pie", **kwargs)
        self.max_wedges_desc = max_slices_desc
        self.wedges = self.ax.patches
        if len(self.wedges) < 1:
            raise ValueError("Pie chart has no wedges")
        wedge_angles = [(w.theta2 - w.theta1) for w in self.wedges]
        self.wedge_pcts = [(100 * wa / 360) for wa in wedge_angles]
        #self.ax_name_to_label["y"] = self.ax.get_ylabel()
        self.parse_data({"x": [self.wedge_pcts]}, mark_type="slice")
        # Use var names, xticks, or yticks as wedge labels if applicable
        if self.wedge_labels is None or len(self.wedge_labels) != len(self.wedge_pcts):
            if len(self.chart_dict["var_info"]) == len(self.wedge_pcts):
                self.wedge_labels = list(self.chart_dict["var_info"].keys())
            elif "ticklabels" in self.chart_dict["ax_info"]["x"] and \
                len(self.chart_dict["ax_info"]["x"]["ticklabels"]) == len(self.wedge_labels):
                self.wedge_labels = self.chart_dict["ax_info"]["x"]["ticklabels"]
            elif "ticklabels" in self.chart_dict["ax_info"]["y"] and \
                len(self.chart_dict["ax_info"]["y"]["ticklabels"]) == len(self.wedge_labels):
                self.wedge_labels = self.chart_dict["ax_info"]["y"]["ticklabels"]


    def get_data_as_md_table(self, max_rows=20, sig_figs=4):
        md_table_str = super().get_data_as_md_table(max_rows=max_rows, sig_figs=sig_figs)
        md_table_str = md_table_str.split("\n")
        md_table_str[0] = md_table_str[0].replace("x ticklabels", "slice label").replace("x", "slice value")
        return "\n".join(md_table_str)


    def get_axes_desc(self, sig_figs=4):
        """
        Return a description of the pie chart's axes in the form:
        '{ax_label} is plotted with {num_wedges} wedges: {wedge_label} ({wedge_percentage}), ...'
        If an axis does not have a label, descriptions are of the form:
        'There are {num_wedges} wedges: {wedge_label} ({wedge_percentage}), ...'
        If there are no labels for the wedges, does not list the wedges with their percentages

        Returns:
            str: The axes description
        """
        axis_label = None
        if self.chart_dict["ax_info"]["x"]["label"] != "":
            axis_label = self.chart_dict["ax_info"]["x"]["label"]
        elif self.chart_dict["ax_info"]["y"]["label"] != "":
            axis_label = self.chart_dict["ax_info"]["y"]["label"]
        if axis_label:
            axes_desc = f"{axis_label} is plotted with {len(self.wedges)} slices"
        else:
            axes_desc = f"There are {len(self.wedges)} slices"
        if len(self.wedges) <= self.max_wedges_desc:
            if len(self.wedge_pcts) > 0 and len(self.wedge_labels) > 0:
                axes_desc += ": "
                label_pcts = [f"{wedge_label} ({format_float(self.wedge_pcts[i], sig_figs)}%)" \
                              for i, wedge_label in enumerate(self.wedge_labels)]
                axes_desc += format_list(label_pcts)
            # TODO: Does it make sense to list percentages without labels here?
            #else:
            #    wedge_pcts_fmt = [f"{format_float(wp, sig_figs)}%" for wp in self.wedge_pcts]
            #    axes_desc += (" " + format_list(wedge_pcts_fmt))
        axes_desc += "."
        return axes_desc

    # Easier to combine encoding and axis descriptions into a single function
    def get_encodings_desc(self, **kwargs):
        """Returns nothing"""
        return ""

    def parse_encodings(self, var_labels=None):
        _, self.wedge_labels = self.ax.get_legend_handles_labels()


    def get_stats_desc(self, stats=["min", "max", "mean"], stat_axis="x", **kwargs):
        """See :func:`~ChartDescription.get_stats_desc`"""
        return super().get_stats_desc(stats=stats, stat_axis=stat_axis, **kwargs)


    def get_trends_desc(self, trends=["shape_x"], trend_axis="x", **kwargs):
        """See :func:`~ChartDescription.get_trends_desc`"""
        return super().get_trends_desc(trends=trends, trend_axis=trend_axis, **kwargs)



##################################################################################################
# Scatter Plot Description
##################################################################################################
class ScatterDescription(ChartDescription):
    """
    The class for generating 2D scatter plot descriptions. Has functions to automatically generate
    descriptions for encodings, axes, annotations, statistics, trends and the overall chart.
    Infers chart data, encodings, etc... from the figure attributes.

    Attributes:
        ax (matplotlib.axes._subplots.AxesSubplot):
            The axis object for the chart to describe
        fig (matplotlib.figure.Figure, optional):
            The figure object for the chart to describe. If no figure is given, it is inferred
            from plt.gcf()
        **kwargs (optional):
            Used to manually specify chart_type, data, and other details of chart descriptions
            including number of signifigant figures, max line width, etc...
    """
    def __init__(self, ax, fig=None, **kwargs):
        """
        Initialize the ScatterDescription with the given attributes. Infers x / y / z data
        (if it exists) and labels from the axis.
        """
        super().__init__(ax, fig, chart_type="scatter", **kwargs)
        self.point_collections = self.ax.collections
        x = [pc._offsets.data[:, 0] for pc in self.point_collections]
        y = [pc._offsets.data[:, 1] for pc in self.point_collections]
        z = []
        if len(self.point_collections) > 0 and len(self.point_collections[0]._offsets.data[0]) > 2:
            z = [pc._offsets.data[:, 2] for pc in self.point_collections]
        if len(x) < 1 and len(y) < 1 and len(z) < 1:
            raise ValueError("Scatter plot contains no points")
        ax_to_data = {"x": x, "y": y}
        if len(z) > 0:
            ax_to_data["z"] = z
        self.parse_data(ax_to_data, mark_type="point")


    def parse_encodings(self, var_labels=None):
        # Infer labels and encoded objects from get_legend_handles_labels if possible
        if self.ax.get_legend_handles_labels():
            legend_handles, legend_labels = self.ax.get_legend_handles_labels()
            if var_labels is None:
                var_labels = legend_labels
        # If we haven't initialized the label to encoding dict, populate it by mapping the legend
        # handle to its color (if possible)
        if var_labels:
            for i, label in enumerate(var_labels):
                if label not in self.chart_dict["var_info"]:
                    self.chart_dict["var_info"][label] = {}
                if self.ax.collections and i < len(self.ax.collections):
                    self.chart_dict["var_info"][label]["color"] = get_color_name(self.ax.collections[i]._facecolors[0])


    def get_stats_desc(self, stats=["numpts", "mean_x", "mean_y", "linearfit", "outliers"], stat_axis=None, **kwargs):
        """See :func:`~ChartDescription.get_stats_desc`"""
        return super().get_stats_desc(stats=stats, stat_axis=stat_axis, **kwargs)



##################################################################################################
# Strip / Swarm Plot Description
##################################################################################################
class StripDescription(ChartDescription):
    """
    The class for generating strip plot descriptions. Has functions to automatically generate
    descriptions for encodings, axes, annotations, statistics, trends and the overall chart.
    Infers chart data, encodings, etc... from the figure attributes.

    Attributes:
        ax (matplotlib.axes._subplots.AxesSubplot):
            The axis object for the chart to describe
        fig (matplotlib.figure.Figure, optional):
            The figure object for the chart to describe. If no figure is given, it is inferred
            from plt.gcf()
        **kwargs (optional):
            Used to manually specify chart_type, data, and other details of chart descriptions
            including number of signifigant figures, max line width, etc...
    """
    def __init__(self, ax, fig=None, **kwargs):
        """
        Initialize the StripDescription with the given attributes. Infers x / y data and labels
        from the axis.
        """
        super().__init__(ax, fig, chart_type="strip", **kwargs)
        self.point_collections = self.ax.collections
        x = [pc._offsets.data[:, 0] for pc in self.point_collections]
        y = [pc._offsets.data[:, 1] for pc in self.point_collections]
        if len(x) < 1 and len(y) < 1:
            raise ValueError("Strip plot contains no points")
        ax_to_data = {"x": x}
        if self.num_axis == "y":
            ax_to_data = {"y": y}
        self.parse_data(ax_to_data, mark_type="point")


    def parse_encodings(self, var_labels=None):
        # Infer labels and encoded objects from get_legend_handles_labels if possible
        if self.ax.get_legend_handles_labels():
            legend_handles, legend_labels = self.ax.get_legend_handles_labels()
            if var_labels is None:
                var_labels = legend_labels
        # Parse which axis is fixed:
        if self.chart_dict["ax_info"]["y"]["scale"] in ["categorical", "datetime"]:
            self.num_axis = "x"
        elif self.chart_dict["ax_info"]["x"]["scale"] in ["categorical", "datetime"]:
            self.num_axis = "y"
        else:
            # Use the point x or y coords as the strip plot positions depending on which axis is fixed.
            x = next(iter(self.chart_dict["var_info"]))["data"]["x"]
            if (len(x) == 0) or np.all(np.isclose(x, x[0])):
                self.num_axis = "y"
            else:
                self.num_axis = "x"
        # Use x/y ticklabels as var labels if applicable
        if var_labels is None:
            if self.num_axis == "x":
                var_labels = self.chart_dict["ax_info"]["y"]["ticklabels"]
            else:
                var_labels = self.chart_dict["ax_info"]["x"]["ticklabels"]
        # If we haven't initialized the label to encoding dict, populate it by mapping the legend
        # handle to its color (if possible)
        if var_labels:
            for i, label in enumerate(var_labels):
                if label not in self.chart_dict["var_info"]:
                    self.chart_dict["var_info"][label] = {}
                if self.point_collections and i < len(self.point_collections):
                    self.chart_dict["var_info"][label]["color"] = get_color_name(self.point_collections[i]._facecolors[0])


    # Window length as a percentage of the total range of data
    def get_stats_desc(self, stats=["numpts", "median", "outliers"], stat_axis=None, **kwargs):
        """See :func:`~ChartDescription.get_stats_desc`"""
        if stat_axis is None:
            stat_axis = self.num_axis
        return super().get_stats_desc(stats=stats, stat_axis=stat_axis, **kwargs)


    def get_trends_desc(self, **kwargs):
        """TODO"""
        return ""