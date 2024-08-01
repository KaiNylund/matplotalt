LINE_STYLE_TO_DESC = {
    "-": "",
    "solid": "",
    "--": "dashed ",
    "dashed": "dashed ",
    "-.": "dash-dotted ",
    "dashdot": "dash-dotted ",
    ":": "dotted ",
    "dotted": "dotted "
}

DEFAULT_DESC_CONFIG = {
    "stats": [],
    "trends": [],
    "sig_figs": 3,
    "max_color_desc_count": 5,
    "max_outliers_desc": 3,
    #"max_line_width": 80,
    "include_annotation_coords": False,
    "include_warnings": False
}

CHART_TYPE_TO_DESC = {
    "other":       "An unknown chart type",
    "line":        "A line plot",
    "area":        "An area chart",
    "scatter":     "A scatter plot",
    "bar":         "A bar chart",
    "heatmap":     "A heatmap",
    "boxplot":     "A boxplot",
    "sankey":      "A sankey diagram",
    "radial line": "A radial line plot",
    "pie":         "A pie chart",
    "strip":       "A strip plot",
    "choropleth":  "A choropleth map",
    "contour":     "A contour plot"
}

CHART_TYPE_TO_ID = {
    "other":      0,
    "line":       1,
    "scatter":    2,
    "bar":        3,
    "heatmap":    4,
    "boxplot":    5,
    "sankey":     6,
    "radial":     7,
    "pie":        8,
    "choropleth": 9,
    "contour":    11
}

CHART_TYPE_ANSWER_PATTERNS = [
    r"it's a (.*) (chart)|(diagram)|(plot)",
    r"a (.*) (chart)|(diagram)|(plot)",
    r"the (.*) (chart)|(diagram)|(plot)",
]