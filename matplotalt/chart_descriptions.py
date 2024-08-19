import numpy as np
from copy import deepcopy
from scipy.stats import pearsonr
from collections import OrderedDict, defaultdict

from matplotalt_helpers import *
from matplotalt_constants import *
from stat_helpers import *


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
        x, y, z (array_like, optional):
            The data represented by the chart to describe. Can have shape (d) for just the values
            or (n, d) for the values of each of n variables. Used to compute statistics for level
            2+ descriptions. If no data are given, they are inferred from the chart. Defaults to None.
        sig_figs (int):
            The number of signifigant figures to use in chart descriptions. Defaults to 4
        max_color_desc_count (int):
            The max number of color encodings to include in chart descriptions. Defaults to 4

    Returns:
        None
    """
    def __init__(self, ax, fig=None, chart_type=None, x=None, y=None, z=None):
        """
        Initialize the ChartDescription object with the given attributes. Tries to infer x/y/z
        data and labels from the axis.
        """
        self.ax = ax
        if fig != None:
            self.fig = fig
        else:
            self.fig = plt.gcf()

        # axis data and labels
        self.ax_name_to_data = OrderedDict()
        if x:
            self.ax_name_to_data["x"] = x
        if y:
            self.ax_name_to_data["y"] = y
        if z:
            self.ax_name_to_data["z"] = z
        self.ax_name_to_label = OrderedDict()
        self.ax_name_to_ticklabels = OrderedDict()
        self.ax_name_to_type = OrderedDict()

        # Check if the axes have x/y/z label getters
        for ax_name in ["x", "y", "z"]:
            if hasattr(self.ax, f"get_{ax_name}label") and callable(getattr(self.ax, f"get_{ax_name}label")):
                label_getter = getattr(self.ax, f"get_{ax_name}label")
                ticklabel_getter = getattr(self.ax, f"get_{ax_name}ticklabels")
                self.ax_name_to_label[ax_name] = label_getter()
                self.ax_name_to_ticklabels[ax_name] = [tl.get_text() for tl in ticklabel_getter()]
                if len(self.ax_name_to_ticklabels[ax_name]) > 0:
                    self.ax_name_to_type[ax_name] = get_ax_ticks_type(self.ax_name_to_ticklabels[ax_name])
            elif ax_name in self.ax_name_to_data and len(self.ax_name_to_data[ax_name]) > 0:
                self.ax_name_to_type[ax_name] = get_ax_ticks_type(self.ax_name_to_data[ax_name])

        # Infer labels and encoded objects from get_legend_handles_labels if possible
        self.labels = None
        self.legend_handles = None
        self.label_to_encoding = OrderedDict()
        if self.ax.get_legend_handles_labels():
            self.legend_handles, self.labels = self.ax.get_legend_handles_labels()
        self.title = " ".join(self.ax.get_title().replace("\n", " ").strip().split())
        # Use suptitle if there's no regular title and only one subplot
        if self.title == "" and self.fig != None and self.fig._suptitle != None and len(self.fig.get_axes()) == 1:
            self.title = " ".join(self.fig._suptitle.get_text().replace("\n", " ").strip().split())

        # User specified fields
        self.chart_type = chart_type


    def get_chart_type(self):
        return self.chart_type


    def get_axes_data(self):
        """ Returns a dict of ax_name -> ax_data """
        return deepcopy(self.ax_name_to_data)


    def get_axes_ticklabels(self):
        """ Returns a dict of ax_name -> ax_ticklabels """
        return deepcopy(self.ax_name_to_ticklabels)


    def get_axes_labels(self):
        """ Returns a dict of ax_name -> ax_label (e.g. {"x": "hours of sunshine"})"""
        return deepcopy(self.ax_name_to_label)


    def get_axes_types(self):
        """ Returns a dict of ax_name -> ax_type (e.g. {"x": "categorical", "y": "date":, "z": "log-linear"})"""
        return deepcopy(self.ax_name_to_type)


    def get_data_as_md_table(self, max_rows=20):
        if len(self.ax_name_to_data) > 0:
            table_dict = {}
            for ax_name, ax_data in self.ax_name_to_data.items():
                # Add data to table
                if isinstance(ax_data, (list, np.ndarray)) and len(ax_data) != 0:
                    if len(ax_data) == 1:
                        ax_data = np.squeeze(ax_data)
                    if ax_name in self.ax_name_to_label and self.ax_name_to_label[ax_name].strip() != "":
                        ax_label = self.ax_name_to_label[ax_name]
                    else:
                        ax_label = ax_name
                    # If axis data are shape (n, d):
                    if isinstance(ax_data[0], (list, np.ndarray)):
                        num_vars = len(ax_data)
                        data_len = len(ax_data[0])
                        # and we have labels:
                        var_labels = deepcopy(self.labels)
                        if var_labels is None or len(var_labels) != num_vars:
                            var_labels = [f"variable {i}" for i in range(num_vars)]
                        # If all variables share the same data for an axis, just include it once
                        if all([len(ax_data[i]) == len(ax_data[j]) and np.allclose(ax_data[i], ax_data[j]) for i in range(num_vars) for j in range(i)]):
                            if ax_name in self.ax_name_to_ticklabels and data_len == len(self.ax_name_to_ticklabels[ax_name]):
                                table_dict[f"{ax_label} ticklabel"] = self.ax_name_to_ticklabels[ax_name]
                            else:
                                table_dict[ax_label] = ax_data[0]
                        # Otherwise include seperate columns for each variable / axis
                        else:
                            for i, l in enumerate(var_labels):
                                table_dict[f"{l} ({ax_label})"] = ax_data[i]
                    # If axis data are shape (d):
                    else:
                        data_len = len(ax_data)
                        if self.labels and len(self.labels) == 1:
                            table_dict[f"{self.labels[0]} ({ax_label})"] = ax_data
                        else:
                            table_dict[ax_label] = ax_data
                    if data_len > max_rows:
                        return "There are too many data points to fit in a table"
                    # Add axis tickslabels to table is possible
                    if ax_name in self.ax_name_to_ticklabels and data_len == len(self.ax_name_to_ticklabels[ax_name]):
                        table_dict[f"{ax_label} ticklabel"] = self.ax_name_to_ticklabels[ax_name]
            return create_md_table(table_dict)
        return ""


    def get_encodings_desc(self, encoded_obj_name="variables", max_color_desc_count=4):
        """
        Return a description of the color encodings for each variable in the figure of the form:
        '{variable_name} is plotted in {variable_color}'
        If the number of variables to describe exceeds the max_color_desc_count, descriptions
        are of the form:
        '{num_variables} {object_name} are plotted for {[all variable names]}'
        (e.g. '12 groups of points are plotted for Jan, Feb,...')

        Attributes:
            encoded_obj_name (str, optional):
                the name of the encoded objects (e.g. 'groups of points', 'bars', 'lines') to use
                in descriptions when there are more than max_color_desc_count variables.

        Returns:
            str: The description of each variable's color encoding
        """
        colors_desc = ""
        # If we haven't initialized the label to encoding dict, populate it by mapping the legend
        # handle to its color (if possible)
        if (len(self.label_to_encoding) == 0) and self.labels and (len(self.labels) > 0) and self.legend_handles:
            for i, label in enumerate(self.labels):
                self.label_to_encoding[label] = get_color_name(self.legend_handles[i]._color)

        if len(self.label_to_encoding) > 0:
            if len(self.label_to_encoding) > max_color_desc_count:
                colors_desc += f"{len(self.label_to_encoding)} {encoded_obj_name} are plotted for "
                colors_desc += format_list(self.label_to_encoding.keys()) + ". "
            else:
                colors_desc += format_list([f"{label} is plotted in {encoding}" for label, encoding in self.label_to_encoding.items()]) + "."
        #else:
        #    warnings.warn(f"Chart is missing a legend or no labels were given")
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
        num_axes = len(self.ax_name_to_data)
        cur_ax_names = ax_names if ax_names is not None else list(self.ax_name_to_data.keys())
        ax_share_a_type = (num_axes > 1 and len(set(self.ax_name_to_type.values())) == 1)
        for i, ax_name in enumerate(cur_ax_names):
            cur_axis_desc = ""
            if ax_name in self.ax_name_to_ticklabels and \
                len(self.ax_name_to_ticklabels[ax_name]) > 0 and \
                self.ax_name_to_ticklabels[ax_name][0]:
                min_text = self.ax_name_to_ticklabels[ax_name][0]
                max_text = self.ax_name_to_ticklabels[ax_name][-1]
            # Try to get axis range from datalim
            elif i < len(self.ax.dataLim._points[0]):
                min_text = format_float(self.ax.dataLim._points[0][i], sig_figs=sig_figs)
                max_text = format_float(self.ax.dataLim._points[1][i], sig_figs=sig_figs)
            # Otherwise get axis range from the data
            elif ax_name in self.ax_name_to_data:
                min_text = format_float(np.nanmin(np.array(self.ax_name_to_data[ax_name])), sig_figs=sig_figs)
                max_text = format_float(np.nanmax(np.array(self.ax_name_to_data[ax_name])), sig_figs=sig_figs)
            else:
                raise ValueError(f"Given axis name: {ax_name} does not have associated data or labels")
            # If there is an axis label, use it in the desc
            if ax_name in self.ax_name_to_label and self.ax_name_to_label[ax_name] != "":
                cur_axis_desc += f"{self.ax_name_to_label[ax_name]} is plotted on the {ax_name}-axis from {min_text} to {max_text}"
            else:
                #warnings.warn(f"The {ax_name}-axis is missing a label")
                cur_axis_desc += f"The {ax_name}-axis ranges from {min_text} to {max_text}"
            if not ax_share_a_type and ax_name in self.ax_name_to_type:
                cur_axis_desc += f" using a {self.ax_name_to_type[ax_name]} scale"
            cur_axis_desc = cur_axis_desc.strip()
            axes_desc_arr.append(cur_axis_desc)
        axes_desc = format_list(axes_desc_arr)
        if ax_share_a_type:
            num_axs_word = "both" if num_axes == 2 else "all"
            axes_desc += f", {num_axs_word} using {next(iter(self.ax_name_to_type.values()))} scales"
        return axes_desc + "."


    def get_annotations_desc(self, include_coords=False, sig_figs=4):
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
        for child in self.ax._children:
            if isinstance(child, matplotlib.text.Annotation):
                if include_coords:
                    coords_desc = f"near x={format_float(child.xy[0], sig_figs)}, \
                                         y={format_float(child.xy[1], sig_figs)} "
                else:
                    coords_desc = ""
                annotations_desc += f"An annotation {coords_desc}reads: '{child._text}'. "
        return annotations_desc.strip()



    def get_data_stats_arr(self, ax_name_to_data, ax_name_to_ticklabels,
                            stats=["min", "max", "mean", "median", "std",
                                   "numpts", "diff", "num_slope_changes",
                                   "maxinc", "maxdec" "outliers",
                                   "linearfit"],
                            var_idx=None, stat_axis=None,
                            max_outliers_desc=4,
                            encoded_obj_name="points", sig_figs=4):
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

            encoded_obj_name (str, optional):
                The name of the encoded object to use when using 'numpts'.
                E.g. "hours of sunshine have 36 points"
                Defaults to 'points'

        Raises:
            ValueError: if given an unsupported statistic

        Returns:
            list[str]: A list of the descriptions of each given stat
        """
        cur_stats_desc_arr = []
        if var_idx is not None:
            var_ax_data = OrderedDict([(ax_name, ax_name_to_data[ax_name][var_idx]) for ax_name in ax_name_to_data.keys()])
        else:
            var_ax_data = ax_name_to_data

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

        for stat_name, stat_axes in stat_name_to_axes.items():
            if stat_name == "numpts": # Doesnt change based on cur_stat_axis
                cur_stats_desc_arr.append(f"{len(next(iter(var_ax_data.values())))} {encoded_obj_name}")
            elif stat_name == "linearfit": # Doesnt change based on cur_stat_axis
                linear_fit = np.polyfit(np.squeeze(var_ax_data["x"]), np.squeeze(var_ax_data["y"]), deg=1)
                cur_stats_desc_arr.append(f"a linear fit of y={format_float(linear_fit[0], sig_figs)}x+{format_float(linear_fit[1], sig_figs)}")
            elif stat_name == "outliers": # Doesnt change based on cur_stat_axis
                xyz = list(var_ax_data.values())
                outlier_idxs_arr = [get_quartile_outlier_idxs(ax_data) for ax_data in xyz]
                outlier_idxs = np.unique(np.concatenate(outlier_idxs_arr)).astype(int)
                outlier_word = "outlier" if len(outlier_idxs) == 1 else "outliers"
                if len(outlier_idxs) == 0:
                    cur_stats_desc_arr.append("no outliers")
                elif len(outlier_idxs) < max_outliers_desc:
                    if np.array(xyz).ndim == 1:
                        outlier_pts = [format_float(xyz[i], sig_figs) for i in outlier_idxs]
                        cur_stats_desc_arr.append(f"{len(outlier_pts)} {outlier_word} at {stat_axis}={format_list(outlier_pts)}")
                    else:
                        outlier_pts = [f"({', '.join([format_float(pt[i], sig_figs) for pt in xyz])})" \
                                        for i in outlier_idxs]
                        cur_stats_desc_arr.append(f"{len(outlier_pts)} {outlier_word} at {format_list(outlier_pts)}")
                else:
                    cur_stats_desc_arr.append(f"{len(outlier_idxs)} {outlier_word}")
            elif stat_name in STAT_NAME_TO_FUNC:
                axs_stats = ", ".join([STAT_NAME_TO_FUNC[stat_name](var_ax_data,
                                       ax_name_to_ticklabels=ax_name_to_ticklabels, stat_axis=ax,
                                       var_idx=var_idx, sig_figs=sig_figs) for ax in stat_axes])
                if stat_name in STAT_NAME_TO_DESC_INTRO:
                    cur_stats_desc_arr.append(f"{STAT_NAME_TO_DESC_INTRO[stat_name]} {axs_stats}".strip())
                else:
                    cur_stats_desc_arr.append(axs_stats)
            else:
                raise ValueError(f"Statistic {stat_name} cannot be computed for the current chart type")
        return cur_stats_desc_arr


    def get_stats_desc(self, stats=[], max_outliers_desc=4, stat_axis=None, encoded_obj_name="points", sig_figs=4):
        """
        Return a description of the provided statistics for each variable along the given axis.

        See :func:`~ChartDescription.get_data_stats_arr` for more details

        """
        stats_desc = ""
        ax_names = list(self.ax_name_to_data.keys())
        # If we have stats and data for at least one axis:
        if (len(stats) > 0) and (len(ax_names) > 0):
            # If there aren't ticklabels for each point, use the actual data as labels instead
            cur_ax_ticklabels = OrderedDict()
            for ax_name in ax_names:
                # Squeeze arrays if applicable
                if len(self.ax_name_to_data[ax_name]) == 1:
                    self.ax_name_to_data[ax_name] = np.squeeze(self.ax_name_to_data[ax_name])
                # Use data as ticks if there are no ticklabels
                if (ax_name in self.ax_name_to_ticklabels):
                    if len(self.ax_name_to_ticklabels[ax_name]) == 1:
                        self.ax_name_to_ticklabels[ax_name] = np.squeeze(self.ax_name_to_ticklabels[ax_name])
                    if len(self.ax_name_to_data[ax_name]) == len(self.ax_name_to_ticklabels[ax_name]):
                        cur_ax_ticklabels[ax_name] = self.ax_name_to_ticklabels[ax_name]
                    else:
                        cur_ax_ticklabels[ax_name] = self.ax_name_to_data[ax_name]
                else:
                    cur_ax_ticklabels[ax_name] = self.ax_name_to_data[ax_name]

            first_ax_data = self.ax_name_to_data[ax_names[0]]
            if isinstance(first_ax_data, (list, np.ndarray)) and len(first_ax_data) != 0:
                # If x, y,... are shape (n, d):
                if isinstance(first_ax_data[0], (list, np.ndarray)):
                    # and we have labels:
                    if self.labels:
                        #if not (len(first_ax_data) == len(self.labels)):
                        #    warnings.warn("Number of variables in axis data does not match numbers of labels")
                        for i, l in enumerate(self.labels):
                            cur_stats_desc_arr = self.get_data_stats_arr(self.ax_name_to_data, cur_ax_ticklabels,
                                                                    var_idx=i,
                                                                    stats=stats, max_outliers_desc=max_outliers_desc,
                                                                    stat_axis=stat_axis, encoded_obj_name=encoded_obj_name,
                                                                    sig_figs=sig_figs)
                            stats_desc += f"{l.capitalize()} have {format_list(cur_stats_desc_arr)}. "
                    # and we don't have labels:
                    elif len(first_ax_data) > 1:
                        stats_desc += f"{len(first_ax_data)} variables are plotted. "
                        for i in range(len(first_ax_data)):
                            cur_stats_desc_arr = self.get_data_stats_arr(self.ax_name_to_data, cur_ax_ticklabels,
                                                                    var_idx=i,
                                                                    stats=stats, max_outliers_desc=max_outliers_desc,
                                                                    stat_axis=stat_axis, encoded_obj_name=encoded_obj_name,
                                                                    sig_figs=sig_figs)
                            stats_desc += f"Data for variable {i} have {format_list(cur_stats_desc_arr)}. "
                # Otherwise x, y,... are shape (d)
                else:
                    if self.labels and len(self.labels) == 1:
                        var_label = self.labels[0]
                    else:
                        var_label = "The data"
                    cur_stats_desc_arr = self.get_data_stats_arr(self.ax_name_to_data, cur_ax_ticklabels,
                                                            stats=stats,
                                                            max_outliers_desc=max_outliers_desc,
                                                            stat_axis=stat_axis, encoded_obj_name=encoded_obj_name,
                                                            sig_figs=sig_figs)
                    stats_desc += f"{var_label.capitalize()} have {format_list(cur_stats_desc_arr)}. "
        return stats_desc.strip()


    def get_single_var_trends_desc(self, ax_name_to_data, ax_name_to_ticklabels, trend_axis="x",
                                   var_idx=None, var_label="data", trends=["shape"],
                                   generally_thresh=0.65, strictly_thresh=1.0, sig_figs=4):
        """_summary_

        Args:
            ax_name_to_data (_type_): _description_
            ax_name_to_ticklabels (_type_): _description_
            trend_axis (str, optional): _description_. Defaults to "x".
            var_idx (_type_, optional): _description_. Defaults to None.
            var_label (str, optional): _description_. Defaults to "data".
            trends (list, optional): _description_. Defaults to ["shape"].
            generally_thresh (float, optional): _description_. Defaults to 0.65.
            strictly_thresh (float, optional): _description_. Defaults to 1.0.
            sig_figs (int, optional): _description_. Defaults to 4.

        Raises:
            ValueError: if given an unsupported trend type

        Returns:
            _type_: _description_
        """
        cur_trends_desc_arr = []
        for trend in trends:
            trend = trend.split("_")
            trend_name = trend[0]

            # Users can either specify an axis by adding to the end of the stat (e.g. "max_x")
            # or by passing it in thought the stat_axis param
            if len(trend) == 2:
                cur_trend_axis = trend[1]
            elif trend_axis is not None:
                cur_trend_axis = trend_axis
            if var_idx is not None:
                axis_data = np.squeeze(ax_name_to_data[cur_trend_axis][var_idx])
                axis_ticklabels = np.squeeze(ax_name_to_ticklabels[cur_trend_axis][var_idx])
            else:
                axis_data = np.squeeze(ax_name_to_data[cur_trend_axis])
                axis_ticklabels = np.squeeze(ax_name_to_data[cur_trend_axis])

            if trend_name == "shape":
                cur_trends_desc_arr.append(get_arr_shape(axis_data, axis_ticklabels, var_label=var_label,
                                                         generally_thresh=generally_thresh,
                                                         strictly_thresh=strictly_thresh, sig_figs=sig_figs))
            #else:
            #    raise ValueError(f"Trend {trend_name} is unsupported for the current chart type.")
        return cur_trends_desc_arr


    def get_multi_var_trends_desc(self, ax_name_to_data, trend_axis="y",
                                  var_labels=[], trends=["correlation"],
                                  sig_figs=4):
        """_summary_

        Args:
            ax_name_to_data (_type_): _description_
            trend_axis (str, optional): _description_. Defaults to "y".
            var_labels (list, optional): _description_. Defaults to [].
            trends (list, optional): _description_. Defaults to ["correlation"].
            sig_figs (int, optional): _description_. Defaults to 4.

        Returns:
            _type_: _description_
        """
        cur_trends_desc_arr = []
        num_vars = len(var_labels)
        if num_vars < 2:
            return cur_trends_desc_arr

        for trend in trends:
            trend = trend.split("_")
            trend_name = trend[0]
            # Users can either specify an axis by adding to the end of the stat (e.g. "max_correlation_y")
            # or by passing it in thought the stat_axis param
            if len(trend) == 2:
                cur_trend_axis = trend[1]
            elif trend_axis is not None:
                cur_trend_axis = trend_axis
            axis_var_data = ax_name_to_data[cur_trend_axis]

            if trend_name == "correlation":
                if num_vars == 2:
                    cur_trends_desc_arr.append(f"{var_labels[0]} and {var_labels[1]} have a correlation of {format_float(pearsonr(axis_var_data[0], axis_var_data[1]).statistic, sig_figs)}.")
                else:
                    max_corr = -2
                    min_corr = 2
                    min_corr_vars = ()
                    max_corr_vars = ()
                    for i in range(num_vars):
                        for j in range(num_vars):
                            cur_vars_corr = pearsonr(axis_var_data[i], axis_var_data[j]).statistic
                            if cur_vars_corr < min_corr:
                                min_corr = cur_vars_corr
                                min_corr_vars = (var_labels[i], var_labels[j])
                            if cur_vars_corr > max_corr:
                                min_corr = cur_vars_corr
                                max_corr_vars = (var_labels[i], var_labels[j])
                    cur_trends_desc_arr.append(f"{max_corr_vars[0]} and {max_corr_vars[1]} have the highest correlation (r={format_float(max_corr, sig_figs)}), while {min_corr_vars[0]} and {min_corr_vars[1]} have the lowest (r={format_float(min_corr, sig_figs)}).")
        return cur_trends_desc_arr


    def get_trends_desc(self, trends=[], trend_axis="y", sig_figs=4):
        """ Return a description of the trends for each variable along the given axis.

        Args:
            trends (list, optional): _description_. Defaults to [].
            trend_axis (str, optional): _description_. Defaults to "y".
            sig_figs (int, optional): _description_. Defaults to 4.

        Returns:
            _type_: _description_
        """
        trends_desc = ""
        ax_names = list(self.ax_name_to_data.keys())
        # If we have trends and data for at least one axis:
        if (len(trends) > 0) and (len(ax_names) > 0):
            # If there aren't ticklabels for each point, use the actual data as labels instead
            cur_ax_ticklabels = OrderedDict()
            for ax_name in ax_names:
                # Squeeze arrays if applicable
                if len(self.ax_name_to_data[ax_name]) == 1:
                    self.ax_name_to_data[ax_name] = np.squeeze(self.ax_name_to_data[ax_name])
                # Use data as ticks if there are no ticklabels
                if (ax_name in self.ax_name_to_ticklabels):
                    if len(self.ax_name_to_ticklabels[ax_name]) == 1:
                        self.ax_name_to_ticklabels[ax_name] = np.squeeze(self.ax_name_to_ticklabels[ax_name])
                    if len(self.ax_name_to_data[ax_name]) == len(self.ax_name_to_ticklabels[ax_name]):
                        cur_ax_ticklabels[ax_name] = self.ax_name_to_ticklabels[ax_name]
                    else:
                        cur_ax_ticklabels[ax_name] = self.ax_name_to_data[ax_name]
                else:
                    cur_ax_ticklabels[ax_name] = self.ax_name_to_data[ax_name]

            first_ax_data = self.ax_name_to_data[ax_names[0]]
            if isinstance(first_ax_data, (list, np.ndarray)) and len(first_ax_data) != 0:
                cur_trends_desc_arr = []
                # If x, y,... are shape (n, d):
                if isinstance(first_ax_data[0], (list, np.ndarray)):
                    # if self.labels is None:
                    #    warnings.warn("Variables have no labels")
                    #if not (len(first_ax_data) == len(self.labels)):
                    #    warnings.warn("Number of variables in axis data does not match numbers of labels")
                    num_vars = len(first_ax_data)
                    var_labels = deepcopy(self.labels)
                    if var_labels is None or len(var_labels) != num_vars:
                        var_labels = [f"variable {i}" for i in range(num_vars)]

                    for i, l in enumerate(var_labels):
                        cur_trends_desc_arr.extend(self.get_single_var_trends_desc(self.ax_name_to_data,
                                                                                   cur_ax_ticklabels,
                                                                                   trend_axis=trend_axis,
                                                                                   var_idx=i,
                                                                                   var_label=l,
                                                                                   trends=trends,
                                                                                   sig_figs=sig_figs))
                    cur_trends_desc_arr.extend(self.get_multi_var_trends_desc(self.ax_name_to_data,
                                                                              trend_axis=trend_axis,
                                                                              var_labels=var_labels,
                                                                              trends=trends,
                                                                              sig_figs=sig_figs))
                # Otherwise x, y,... are shape (d)
                else:
                    if self.labels and len(self.labels) == 1:
                        var_label = self.labels[0]
                    else:
                        var_label = "The data"
                    cur_trends_desc_arr.extend(self.get_single_var_trends_desc(self.ax_name_to_data,
                                                                               cur_ax_ticklabels,
                                                                               trend_axis=trend_axis,
                                                                               var_label=var_label,
                                                                               trends=trends,
                                                                               sig_figs=sig_figs))
                trends_desc = " ".join(cur_trends_desc_arr)
        return trends_desc.strip()


    def get_chart_type_desc(self):
        """
        Return a description of the current chart type of the form:
        'A {formatted chart_type} titled {chart_title}'
        """
        chart_type_desc = ""
        if self.chart_type in CHART_TYPE_TO_DESC:
            chart_type_desc = CHART_TYPE_TO_DESC[self.chart_type]
        else:
            chart_type_desc = f"A {self.chart_type}"
        if self.title != "":
            chart_type_desc += f" titled \'{self.title}\'"
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

        alt_text_arr = []
        alt_text_arr.append(self.get_chart_type_desc())
        # Add axis and encoding descriptions
        if desc_level > 0:
            alt_text_arr.append(self.get_axes_desc(sig_figs=desc_config["sig_figs"]))
            alt_text_arr.append(self.get_encodings_desc(max_color_desc_count=desc_config["max_color_desc_count"]))
        alt_text_arr.append(self.get_annotations_desc(include_coords=desc_config["include_annotation_coords"], sig_figs=desc_config["sig_figs"]))
        # Add stats
        if desc_level > 1:
            # if stats is None, use the default stats from the child class
            if desc_config["stats"] and len(desc_config["stats"]) > 0:
                alt_text_arr.append(self.get_stats_desc(stats=desc_config["stats"], max_outliers_desc=desc_config["max_outliers_desc"], sig_figs=desc_config["sig_figs"]).strip().capitalize())
            else:
                alt_text_arr.append(self.get_stats_desc(max_outliers_desc=desc_config["max_outliers_desc"], sig_figs=desc_config["sig_figs"]))
        # Add trends if applicable
        if desc_level > 2:
            if desc_config["trends"] and len(desc_config["trends"]) > 0:
                alt_text_arr.append(self.get_trends_desc(trends=desc_config["trends"], sig_figs=desc_config["sig_figs"]))
            else:
                alt_text_arr.append(self.get_trends_desc(sig_figs=desc_config["sig_figs"]))

        alt_text_arr = [al.strip().capitalize() for al in alt_text_arr]
        alt_text_arr = [al for al in alt_text_arr if al]
        alt_text = " ".join(alt_text_arr)
        alt_text.replace(r'\s+', r'\s')
        #alt_text = insert_line_breaks(alt_text, max_line_width=desc_config["max_line_width"])
        return alt_text



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
        self.wedges = self.ax.patches
        if len(self.wedges) < 1:
            raise ValueError("Pie chart has no wedges")
        self.wedge_angles = [(w.theta2 - w.theta1) for w in self.wedges]
        self.wedge_pcts = [(100 * wa / 360) for wa in self.wedge_angles]
        self.ax_name_to_data["x"] = self.wedge_pcts
        self.ax_name_to_label["y"] = self.ax.get_ylabel()
        self.max_wedges_desc = max_slices_desc
        if self.labels and len(self.labels) == len(self.wedge_pcts):
            self.ax_name_to_ticklabels["x"] = self.labels


    def get_data_as_md_table(self, max_rows=20):
        md_table_str = super().get_data_as_md_table(max_rows=max_rows)
        md_table_str = md_table_str.split("\n")
        md_table_str[0] = md_table_str[0].replace("x ticklabel", "wedge label").replace("x", "wedge value")
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
        if self.ax_name_to_label["x"] != "":
            axis_label = self.ax_name_to_label["x"]
        elif self.ax_name_to_label["y"] != "":
            axis_label = self.ax_name_to_label["y"]
        if axis_label:
            axes_desc = f"{axis_label} is plotted with {len(self.wedges)} wedges"
        else:
            axes_desc = f"There are {len(self.wedges)} wedges"
        if len(self.wedges) <= self.max_wedges_desc:
            if self.labels:
                axes_desc += ": "
                label_pcts = [f"{label} ({format_float(self.wedge_pcts[i], sig_figs)}%)" \
                              for i, label in enumerate(self.labels)]
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


    def get_stats_desc(self, stats=["min", "max", "mean"], stat_axis="x", **kwargs):
        """See :func:`~ChartDescription.get_stats_desc`"""
        return super().get_stats_desc(stats=stats, stat_axis=stat_axis, encoded_obj_name="slices", **kwargs)


    def get_trends_desc(self, trends=["shape_x"], trend_axis="x", **kwargs):
        """See :func:`~ChartDescription.get_trends_desc`"""
        return super().get_trends_desc(trends=trends, trend_axis=trend_axis, **kwargs)



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
        self.ax_name_to_data["x"] = [pc._offsets.data[:, 0] for pc in self.point_collections]
        self.ax_name_to_data["y"] = [pc._offsets.data[:, 1] for pc in self.point_collections]
        if len(self.ax_name_to_data["x"]) < 1 and len(self.ax_name_to_data["y"]) < 1:
            raise ValueError("Strip plot contains no points")

        if self.ax_name_to_type["y"] in ["categorical", "datetime"]:
            self.num_axis = "x"
            self.labels = self.ax_name_to_ticklabels["y"]
        elif self.ax_name_to_type["x"] in ["categorical", "datetime"]:
            self.num_axis = "y"
            self.labels = self.ax_name_to_ticklabels["x"]
        else:
            # Use the point x or y coords as the strip plot positions depending on which axis is fixed.
            if (len(self.ax_name_to_data["x"]) == 0) or \
            (len(self.ax_name_to_data["x"][0]) == 0) or \
            np.all(np.isclose(self.ax_name_to_data["x"][0], self.ax_name_to_data["x"][0][0])):
                self.num_axis = "y"
                self.labels = self.ax_name_to_ticklabels["x"]
            else:
                self.num_axis = "x"
                self.labels = self.ax_name_to_ticklabels["y"]


    def get_encodings_desc(self, max_color_desc_count=4, **kwargs):
        """See :func:`~ChartDescription.get_encodings_desc`"""
        # Need to have a seperate function here because we get the colors with
        # .facecolors[0] instead of .color
        colors_desc = ""
        if self.labels and (len(self.labels) > 0):
            if len(self.labels) > max_color_desc_count:
                colors_desc += f"{len(self.labels)} collections of points are plotted for "
                colors_desc += format_list(self.labels)
            else:
                colors_desc += format_list([f"{self.labels[i]} are plotted in {get_color_name(self.point_collections[i]._facecolors[0])}" for i in range(len(self.labels))]) + "."
        #else:
        #    warnings.warn(f"Chart is missing a legend or no labels were given")
        return colors_desc.strip()


    # Window length as a percentage of the total range of data
    def get_stats_desc(self, stats=["numpts", "median", "outliers"], stat_axis=None, **kwargs):
        """See :func:`~ChartDescription.get_stats_desc`"""
        if stat_axis is None:
            stat_axis = self.num_axis
        return super().get_stats_desc(stats=stats, stat_axis=stat_axis, **kwargs)


    def get_trends_desc(self, **kwargs):
        """TODO"""
        return ""



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
        self.x = []
        self.y = []
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
                    self.x.append(line._xy[:-1, 0])
                    self.y.append(line._xy[:-1, 1])
                else:
                    self.x.append(line._xy[:, 0])
                    self.y.append(line._xy[:, 1])
        # Unless all lines are horizontal / vertical,
        # remove lines that are constant on one axis so they aren't included in stats
        if len(constant_line_idxs) < len(self.lines):
            for i in constant_line_idxs:
                del self.lines[i]
        if len(self.x) > 0:
            self.ax_name_to_data["x"] = self.x
        if len(self.y) > 0:
            self.ax_name_to_data["y"] = self.y


    def get_encodings_desc(self, max_color_desc_count=4, **kwargs):
        """
        See :func:`~ChartDescription.get_stats_desc`. Additionally includes
        descriptions of any vertical and horizontal lines in the form:
        'There are vertical lines at "x"={vertical_line_xs}'
        """
        encodings_desc = super().get_encodings_desc(encoded_obj_name="lines", max_color_desc_count=max_color_desc_count, **kwargs)
        if encodings_desc == "" and len(self.lines) == 1 and max_color_desc_count > 0:
            line = self.lines[0]
            encodings_desc += f" The data are plotted in {LINE_STYLE_TO_DESC[line.get_linestyle()]}{get_color_name(line._color)}. "
        if len(self.vline_xs) == 1:
            encodings_desc += f" There is a vertical line at x={self.vline_xs[0]}. "
        elif len(self.vline_xs) > 1:
            encodings_desc += f" There are vertical lines at x={self.vline_xs}. "
        if len(self.hline_ys) == 1:
            encodings_desc += f" There is a horizontal line at y={self.hline_ys[0]}. "
        elif len(self.hline_ys) > 1:
            encodings_desc += f" There are horizontal lines at y={self.hline_ys}. "
        return encodings_desc


    def get_stats_desc(self, stats=["min_y", "max_y", "mean_y"], stat_axis="y", **kwargs):
        """See :func:`~ChartDescription.get_stats_desc`"""
        return super().get_stats_desc(stats=stats, stat_axis=stat_axis, encoded_obj_name="lines", **kwargs)


    def get_trends_desc(self, trends=["shape_y", "correlation_y"], trend_axis="y", **kwargs):
        """See :func:`~ChartDescription.get_trends_desc`"""
        return super().get_trends_desc(trends=trends, trend_axis=trend_axis, **kwargs)



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
        self.x = []
        self.y = []
        for line in self.lines:
            cur_xs = line._xy[:, 0]
            cur_ys = line._xy[:, 1]
            if np.all(np.isclose(cur_xs, cur_xs[0])):
                self.vline_xs.append(cur_xs[0])
            elif np.all(np.isclose(cur_ys, cur_ys[0])):
                self.hline_ys.append(cur_ys[0])
            else:
                self.x.append(line._xy[:, 0])
                self.y.append(line._xy[:, 1])
        if len(self.x) > 0:
            self.ax_name_to_data["x"] = self.x
        if len(self.y) > 0:
            self.ax_name_to_data["y"] = self.y


    def get_encodings_desc(self, max_color_desc_count=4, **kwargs):
        """
        See :func:`~ChartDescription.get_stats_desc`. Additionally includes
        descriptions of any vertical and horizontal lines in the form:
        'There are vertical lines at x={vertical_line_xs}'
        """
        encodings_desc = super().get_encodings_desc(encoded_obj_name="areas", max_color_desc_count=max_color_desc_count, **kwargs)
        if encodings_desc == "" and len(self.lines) == 1 and max_color_desc_count > 0:
            line = self.lines[0]
            encodings_desc += f" The data are plotted in {LINE_STYLE_TO_DESC[line.get_linestyle()]}{get_color_name(line._color)}. "
        if len(self.vline_xs) == 1:
            encodings_desc += f" There is a vertical line at x={self.vline_xs[0]}. "
        elif len(self.vline_xs) > 1:
            encodings_desc += f" There are vertical lines at x={self.vline_xs}. "
        if len(self.hline_ys) == 1:
            encodings_desc += f" There is a horizontal line at y={self.hline_ys[0]}. "
        elif len(self.hline_ys) > 1:
            encodings_desc += f" There are horizontal lines at y={self.hline_ys}. "
        return encodings_desc


    def get_stats_desc(self, stats=["min_y", "max_y", "mean_y"], stat_axis="y", **kwargs):
        """See :func:`~ChartDescription.get_stats_desc`"""
        return super().get_stats_desc(stats, stat_axis=stat_axis, encoded_obj_name="areas", **kwargs)


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
        Initialize the LineDescription with the given attributes. Infers x / y data
        and labels from the axis.
        """
        super().__init__(ax, fig, chart_type="bar", **kwargs)
        self.bars = self.ax.containers
        if len(self.bars) < 1:
            raise ValueError("Bar chart contains no bars")

        self.bar_values = [b.datavalues for b in self.bars]
        self.bar_ticks = [list(range(len(bv))) for bv in self.bar_values]

        self.cat_axis = "x"
        self.num_axis = "y"
        if self.ax_name_to_type["y"] in ["categorical", "datetime"]:
            self.cat_axis = "y"
            self.num_axis = "x"
        self.ax_name_to_data[self.num_axis] = self.bar_values
        self.ax_name_to_data[self.cat_axis] = self.bar_ticks


    def get_encodings_desc(self, **kwargs):
        """See :func:`~ChartDescription.get_encodings_desc`"""
        return super().get_encodings_desc(encoded_obj_name="bars", **kwargs)


    def get_stats_desc(self, stats=["numpts", "min", "max", "mean"], stat_axis=None, **kwargs):
        """See :func:`~ChartDescription.get_stats_desc`"""
        if stat_axis is None:
            stat_axis = self.num_axis
        return super().get_stats_desc(stats, stat_axis=stat_axis, encoded_obj_name="bars", **kwargs)


    def get_trends_desc(self, trends=["shape", "correlation"], trend_axis=None, **kwargs):
        """See :func:`~ChartDescription.get_trends_desc`"""
        if trend_axis is None:
            trend_axis = self.num_axis
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
        self.ax_name_to_data["x"] = [pc._offsets.data[:, 0] for pc in self.point_collections]
        self.ax_name_to_data["y"] = [pc._offsets.data[:, 1] for pc in self.point_collections]
        if len(self.point_collections) > 0 and len(self.point_collections[0]._offsets.data[0]) > 2:
            self.ax_name_to_data["z"] = [pc._offsets.data[:, 2] for pc in self.point_collections]
        if len(self.ax_name_to_data["x"]) < 1 and \
           len(self.ax_name_to_data["y"]) < 1 and \
           len(self.ax_name_to_data["z"]) < 1:
            raise ValueError("Scatter plot contains no points")


    def get_encodings_desc(self, max_color_desc_count=4, **kwargs):
        """See :func:`~ChartDescription.get_encodings_desc`"""
        # Need to have a seperate function here because we get the colors with
        # .facecolors[0] instead of .color
        colors_desc = ""
        if self.labels and (len(self.labels) > 0):
            if len(self.labels) > max_color_desc_count and max_color_desc_count > 1:
                colors_desc += f"{len(self.labels)} collections of points are plotted for "
                colors_desc += format_list(self.labels)
            elif self.legend_handles:
                colors_desc += format_list([f"{self.labels[i]} are plotted in {get_color_name(self.legend_handles[i]._facecolors[0])}" for i in range(len(self.legend_handles))]) + "."
            #else:
            #    warnings.warn(f"Labels provided but could not infer encoded objects")
        #else:
        #    warnings.warn(f"Chart is missing a legend or no labels were given")
        return colors_desc.strip()


    def get_stats_desc(self, stats=["numpts", "mean_x", "mean_y", "linearfit", "outliers"], stat_axis=None, **kwargs):
        """See :func:`~ChartDescription.get_stats_desc`"""
        return super().get_stats_desc(stats=stats, stat_axis=stat_axis, **kwargs)



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

        self.vert_idx = 1
        if self.ax_name_to_type["y"] in ["categorical", "datetime"]:
            self.vert_idx = 0

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
        # If there are the same number of xticklabels as boxes, use them for the box labels
        if "x" in self.ax_name_to_ticklabels and len(self.ax_name_to_ticklabels["x"]) == len(self.box_num_to_quartiles):
            for box_num in list(self.box_num_to_quartiles.keys()):
                box_label = self.ax_name_to_ticklabels["x"][box_num]
                self.box_num_to_quartiles[box_label] = self.box_num_to_quartiles.pop(box_num)

        if len(self.box_num_to_quartiles) < 1:
            raise ValueError("Unable to infer boxplot values")


    def get_data_as_md_table(self, **kwargs):
        return f"Tables are currently unsupported for charts of type: {self.chart_type}"


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
        return super().get_encodings_desc(encoded_obj_name="boxplots", **kwargs)



    def get_stats_desc(self, stats=["median", "iqr", "outliers"], max_outliers_desc=4, sig_figs=4):
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
        self.coords = self.quadmesh._coordinates[:-1, :-1, :]
        self.shape = self.coords.shape[:2]
        self.coords = self.coords.reshape(-1, 2)
        self.x = self.coords[:, 0]
        self.y = self.coords[:, 1]
        self.z = self.quadmesh._A.flatten()
        self.ax_name_to_data["x"] = self.x
        self.ax_name_to_data["y"] = self.y
        self.ax_name_to_data["z"] = self.z
        if len(self.x) < 1 or len(self.y) < 1:
            raise ValueError("Heatmap cells are missing coordinates")
        elif len(self.z) < 1:
            raise ValueError("Heatmap cells are missing values")
        self.ax_name_to_ticklabels["x"] = np.tile(self.ax_name_to_ticklabels["x"], self.shape[1])
        self.ax_name_to_ticklabels["y"] = np.repeat(self.ax_name_to_ticklabels["y"], self.shape[0])
        if self.quadmesh.colorbar is not None:
            self.ax_name_to_label["z"] = self.quadmesh.colorbar.ax.get_ylabel()
            self.ax_name_to_ticklabels["z"] = [tl.get_text() for tl in self.quadmesh.colorbar.ax.get_yticklabels()]
        if "z" in self.ax_name_to_ticklabels and len(self.ax_name_to_ticklabels["z"]) > 0:
            self.ax_name_to_type["z"] = get_ax_ticks_type(self.ax_name_to_ticklabels["z"])


    def get_chart_type_desc(self):
        """
        Return a description of the current heatmap title of the form:
        'A {heatmap_width}x{heatmap_height} titled {heatmap_title}'
        """
        chart_type_desc = f"A {self.shape[1]}x{self.shape[0]} heatmap"
        if self.title != "":
            chart_type_desc += f" titled \'{self.title}\'"
        chart_type_desc += ". "
        return chart_type_desc


    def get_stats_desc(self, stats=["min_z", "max_z", "mean_z"], stat_axis="z", **kwargs):
        """See :func:`~ChartDescription.get_stats_desc`"""
        return super().get_stats_desc(stats=stats, stat_axis=stat_axis, encoded_obj_name="cells", **kwargs)



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
        self.data = self.ax_img._A
        self.shape = self.data.shape
        x, y = np.indices(self.shape)
        self.x = x.ravel(order='F')
        self.y = y.ravel(order='F')
        self.z = self.data.ravel(order='F')
        self.ax_name_to_data["x"] = self.x
        self.ax_name_to_data["y"] = self.y
        self.ax_name_to_data["z"] = self.z
        if len(self.x) < 1 or len(self.y) < 1:
            raise ValueError("Image is missing coordinates")
        elif len(self.z) < 1:
            raise ValueError("Image is missing values")
        self.ax_name_to_ticklabels["x"] = np.tile(self.ax_name_to_ticklabels["x"], self.shape[1])
        self.ax_name_to_ticklabels["y"] = np.repeat(self.ax_name_to_ticklabels["y"], self.shape[0])
        if self.ax_img.colorbar is not None:
            self.ax_name_to_label["z"] = self.ax_img.colorbar.ax.get_ylabel()
            self.ax_name_to_ticklabels["z"] = [tl.get_text() for tl in self.ax_img.colorbar.ax.get_yticklabels()]
        if "z" in self.ax_name_to_ticklabels and len(self.ax_name_to_ticklabels["z"]) > 0:
            self.ax_name_to_type["z"] = get_ax_ticks_type(self.ax_name_to_ticklabels["z"])


    def get_chart_type_desc(self):
        """
        Return a description of the current heatmap title of the form:
        'A {heatmap_width}x{heatmap_height} titled {heatmap_title}'
        """
        chart_type_desc = f"A {self.shape[1]}x{self.shape[0]} image"
        if self.title != "":
            chart_type_desc += f" titled \'{self.title}\'"
        chart_type_desc += ". "
        return chart_type_desc


    def get_stats_desc(self, stats=["min_z", "max_z", "mean_z"], stat_axis="z", **kwargs):
        """See :func:`~ChartDescription.get_stats_desc`"""
        return super().get_stats_desc(stats=stats, stat_axis=stat_axis, encoded_obj_name="pixels", **kwargs)





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
        self.level_centers = []
        for path in self.contour_set._paths[1:-1]:
            self.level_centers.append(np.mean(path._vertices, axis=0))
        self.ax_name_to_data["x"] = []
        self.ax_name_to_data["y"] = []


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

