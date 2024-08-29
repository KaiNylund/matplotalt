import os
import re
import secrets
import pyexiv2
import warnings
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML

from matplotalt_constants import *
from matplotalt_helpers import *
from chart_descriptions import *
from api_helpers import *


CHART_TYPE_TO_CLASS = {
    "line":    LineDescription,
    "bar":     BarDescription,
    "scatter": ScatterDescription,
    "image":   ImageDescription,
    "heatmap": HeatmapDescription,
    "boxplot": BoxplotDescription,
    "pie":     PieDescription,
    "radial":  LineDescription,
    "strip":   StripDescription,
    "contour": ContourDescription,
    "area":    AreaDescription,
    "other":   ChartDescription,
}


def infer_chart_type(ax=None):
    """ Infer the chart type of the given matplotlib axis

    Args:
        ax (matplotlib.axis.Axis, optional):
            The axis to infer from. If none is given, the current matplotlib axis is used.
            Defaults to None.

    Returns:
        str: The chart type of the given or current matplotlib axis.
    """
    if ax is None:
        ax = infer_single_axis(ax)
    warnings.filterwarnings("ignore", category=UserWarning)
    for chart_type, chart_type_class in CHART_TYPE_TO_CLASS.items():
        try:
            chart_desc_class = chart_type_class(ax)
            #chart_desc = chart_desc_class.get_chart_desc()
            warnings.filterwarnings("default", category=UserWarning)
            return chart_type
        except Exception as e:
            continue
    warnings.filterwarnings("default", category=UserWarning)
    return "unknown"


def get_cur_chart_desc_class(ax=None, chart_type=None, include_warnings=False,
                             chart_type_classifier="auto"):
    """ Returns a ChartDescription object for the current or given matplotlib axis

    Args:
        ax (matplotlib.axis.Axis, optional):
            The axis used to create the ChartDescription object.
            If none is given, the current matplotlib axis is usedDefaults to None.
        chart_type (str, optional):
            The chart type of the current or given axis. If none is given, it is inferred using
            the given chart_type_classifier method Defaults to None.
        include_warnings (bool, optional):
            Whether to display warnings when creating  the ChartDescription. Defaults to False.
        chart_type_classifier (str, optional):
            How to infer the given axis' chart type. Currently supported methods are

            * "auto": Use the first ChartDescription class that is created without errors
            * "model": Use a finetuned vision model to classify the chart type

            Defaults to "auto".

    Raises:
        ValueError: If given an unsupported chart_type_classifier, or unable to infer chart_type

    Returns:
        ChartDescription: description class instantiated for the given matplotlib axis
    """
    if ax is None:
        ax = infer_single_axis(ax)
    if not include_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)
    if not chart_type:
        if chart_type_classifier == "auto":
            chart_type = infer_chart_type(ax)
        elif chart_type_classifier == "model":
            chart_type = infer_model_chart_type(ax)
        else:
            raise ValueError(f"Unknown chart_type_classifier: {chart_type_classifier}; supported options are 'auto' and 'model'")
        if chart_type == "unknown":
            raise ValueError("Unable to infer chart type, please pass a chart_type parameter to generate_alt_text")
    return CHART_TYPE_TO_CLASS[chart_type](ax)


# TODO: Add markdown heading that says the output is alt text (take optional heading level parameter)
#       'Alt text for {chart title}'
def add_alt_text(alt_text, methods=["html"], output_file=None):
    """
    Surfaces given alt text in a Jupyter notebook using the given methods.

    Args:
        alt_text (str):
            The given text to display
        methods (List[str], optional):
            The methods used to display alt text in Jupyter. Currently supported methods are:

            * "html": displays the last figure in html with an alt property containing the given text.
            * "markdown": display text in markdown in the current cell output.
            * "new_cell": create a new (code) cell after this one containing the markdown magic followed by the given text.
            * "txt_file": writes the given text to a text file at output_file.
            * "img_file": saves the last matplotlib figure to output_file with the given text in its alt property.

            Defaults to "html" in notebooks and None in non-interactive environments.
            NOTE: markdown data tables are excluded from all but the "markdown" and "new_cell" methods
            NOTE: "html", "markdown", and "new_cell" are only supported in notebooks
        output_file (str|None, optional):
            The output file name to use for html and txt_file methods. If the file already exists,
            defaults to output_file plus a hash. If None is given, defaults to "mpa_" plus the
            title of the last matplotlib chart. If None is given and there's no title,
            defaults to "mpa_" plus a hash.


    Returns:
        None
    """
    if methods is None or len(methods) < 1:
        return
    elif isinstance(methods, str):
        methods = [methods]
    is_env_notebook = is_notebook()
    #if not is_notebook() and any(m in methods for m in ["html", "markdown", "new_cell"]):
    #    raise ValueError("The methods 'html', 'markdown', and 'new_cell' are only supported in notebook environments")
    if is_env_notebook and "markdown" in methods:
        # Display the alt text in markdown before the plot
        display(Markdown(alt_text))
    if is_env_notebook and "new_cell" in methods:
        # Create a new (code) cell with the given alt text
        create_new_cell("%%markdown\n" + alt_text)
    # Exclude markdown data tables and newlines from the following methods:
    dt_start_idx = alt_text.lower().find("data table:")
    if dt_start_idx != -1:
        alt_text = alt_text[:dt_start_idx]
    alt_text = alt_text.replace("\n", "")
    # Create output file if needed
    if "txt_file" in methods or "img_file" in methods:
        if not output_file:
            chart_title = re.findall(r"titled \'(.*?)\'\.", alt_text.replace("\n", ""))
            if len(chart_title) > 0:
                output_file = f"mpa_{url_safe(chart_title[0])}"
            else:
                output_file = "matplotalt_tmp_" + secrets.token_urlsafe(16)
        else:
            output_file = os.path.abspath(output_file).split(".")[0]
        if os.path.exists(output_file + ".txt") or os.path.exists(output_file + ".jpg"):
            output_file += "_" + secrets.token_urlsafe(16)
    # Methods that need to draw the figure
    if "html" in methods or "img_file" in methods:
        fig = plt.gcf()
        fig.canvas.draw()
        pil_img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        if "img_file" in methods:
            pil_img.save(output_file + ".jpg", 'JPEG')
            with open(output_file + ".jpg", 'rb+') as img_f:
                with pyexiv2.ImageData(img_f.read()) as pyexif_img:
                    pyexif_img.modify_exif({"Exif.Image.ImageDescription": alt_text})
                    # Empty the original file
                    img_f.seek(0)
                    img_f.truncate()
                    # Get the bytes data of the image and save it to the file
                    img_f.write(pyexif_img.get_bytes())
        # Display in HTML with the given alt text using a dataURL
        if is_env_notebook and "html" in methods:
            data_url = "data:image/png;base64," + pillow_image_to_base64_string(pil_img)
            plt.close()
            display(HTML(f'<img src="{data_url}" alt="{alt_text}"/>'))
    if "txt_file" in methods:
        # Save alt text to the given output file
        with open(f"{output_file}.txt", "w") as output_file:
            output_file.write(alt_text)
    plt.show()


# for backwards compatability
def surface_alt_text(**kwargs):
    add_alt_text(**kwargs)


# TODO: Add option to output alt text as a latex command
def generate_alt_text(axs=None, fig=None, chart_type=None, desc_level=2, chart_type_classifier="auto",
                      max_subplots=9, include_warnings=False, include_table=False, max_table_rows=20, **kwargs):
    """
    Args:
        axs (matplotlib.axis.Axis|List[matplotlib.axis.Axis], optional):
            The matplotlib axis object (or list of axes) containing figure labels, data and other
            properties used in the alt text. If multiple axes are passed in a list then
            alt text is provided for each subplot along with the figure suptitle. If no axes are
            provided, they are inferred from the last matplotlib figure.
        fig (matplotlib.figure.Figure, optional):
            The matplotlib figure object used to generate alt text. If no fig is provided,
            it is inferred from the last matplotlib figure.
        chart_type (str, optional):
            The type of chart to describe. If no chart type is given, it is inferred using
            the chart_type_classifier method.
        desc_level (int, optional):
            The semantic level to use in alt text descriptions based on Lundgard and
            Satyanarayan 2021. Defaults to 2. Currently supported description levels are:

            1. axis and encoding descriptions
            2. level 1 plus statistics about the chart's data
            3. level 2 plus trends in the data and relationships between variables

        include_warnings (bool, optional):
            Whether to display warnings when creating  the ChartDescription. Defaults to False.
        chart_type_classifier (str, optional):
            How to infer the given axis' chart type. Currently supported methods are

            * "auto": Use the first ChartDescription class that is created without errors
            * "model": Use a finetuned vision model to classify the chart type

            Defaults to "auto".
        max_subplots (int, optional):
            If there are more than max_subplots subplots, only the number of plots and suptitle
            will be included in alt text.
        include_table (bool, optional):
            Whether to include a markdown table with the chart's data in the generated alt text.
            Defaults to False.
        max_table_rows (int, optional):
            The maximum length markdown table to include in generated alt text. Defaults to 20.
        kwargs (optional):
            Extra config options for the generated alt text, including the number of signifigant
            figures, max color descriptions, etc...

    Returns:
        str: the starter alt text for the given fig and axs. If there are multiple axis
        objects in axs, returns the suptitle followed by alt text for each axis and their number.
    """
    if not axs:
        axs = plt.gcf().get_axes()
    elif isinstance(axs, matplotlib.axes._axes.Axes):
        axs = [axs]
    if not fig:
        fig = plt.gcf()

    alt_text = ""
    # Dont generate alt text for colorbars
    colorbar_idxs = []
    for i in range(len(axs)):
        if hasattr(axs[i], "_colorbar") or hasattr(axs[i], "_colorbar_info"):
            colorbar_idxs.append(i)
    for idx in sorted(colorbar_idxs, reverse=True):
        del axs[idx]

    # Create alt text for all subplots + suptitle
    if isinstance(axs, (list, np.ndarray)) and len(np.array(axs).flatten()) > 1:
        flattened_axs = np.array(axs).flatten()
        chart_title = fig.get_suptitle()
        if chart_title is not None:
            chart_title = " ".join(chart_title.replace("\n", " ").strip().split())
        alt_text += f"A figure with {len(flattened_axs)} subplots"
        if chart_title != None and chart_title != "":
            alt_text += f" titled \'{chart_title}\'"
        alt_text += "."
        # If there are more than max_subplots subplots, only return the suptitle + number
        if len(flattened_axs) > max_subplots:
            return alt_text
        alt_text += "\n\n"
        for ax_idx, ax in enumerate(flattened_axs):
            alt_text += f" Subplot {ax_idx + 1}: "
            chart_desc_class = get_cur_chart_desc_class(ax=ax, chart_type=chart_type, chart_type_classifier=chart_type_classifier, include_warnings=include_warnings)
            alt_text += chart_desc_class.get_chart_desc(desc_level=desc_level, **kwargs)
            if include_table:
                alt_text += f" Data table:\n\n{chart_desc_class.get_data_as_md_table(max_rows=max_table_rows)}"
            if ax_idx < len(flattened_axs) - 1:
                alt_text += "\n\n"
    # Create alt text for a single plot
    else:
        if isinstance(axs, (list, np.ndarray)):
            if len(axs) == 1:
                ax = axs[0]
            else:
                #raise ValueError("Given axes are blank")
                return "The figure is blank"
        else:
            ax = axs
        chart_desc_class = get_cur_chart_desc_class(ax=ax, chart_type=chart_type, chart_type_classifier=chart_type_classifier, include_warnings=include_warnings)
        alt_text += chart_desc_class.get_chart_desc(desc_level=desc_level, **kwargs)
        if include_table:
            alt_text += f" Data table:\n\n{chart_desc_class.get_data_as_md_table(max_rows=max_table_rows)}"
    alt_text = ". ".join([sent.capitalize() for sent in alt_text.split(". ")])
    return alt_text


# Main function that should replace plt.show()
# TODO: Add "block" param so people can pass that in if they had it in plt.show() before
# TODO: Is there a way to alias plt.show to this function (e.g. plt.show() = show_with_alt())?
def show_with_alt(alt_text=None, axs=None, fig=None, methods=["html"], chart_type=None,
                  desc_level=2, context="", output_file=None,
                  return_alt=False, **kwargs):
    """
    Generates and surfaces starter alt text describing the given figure and axis.

    NOTE behavior with plt.show(): show_with_alt() should replace calls to plt.show()...
    If plt.show() called after add_alt_text(methods=["html",...]) then the displayed
    image will overwrite the version with embedded alt text. If plt.show() is called before
    generate_alt_text, then it will not be able to create alt text because the axs are cleared.

    Args:
        axs (matplotlib.axis.Axis|List[matplotlib.axis.Axis], optional):
            The matplotlib axis object (or list of axes) containing figure labels, data and other
            properties used in the alt text. If multiple axes are passed in a list then
            alt text is provided for each subplot along with the figure suptitle. If no axes are
            provided, they are inferred from the last matplotlib figure.
        fig (matplotlib.figure.Figure, optional):
            The matplotlib figure object used to generate alt text. If no fig is provided,
            it is inferred from the last matplotlib figure
        chart_type (str, optional):
            The type of chart to describe. If no chart type is given, it is inferred using
            the chart_type_classifier method.
        methods (list[str], optional):
            The methods used to display the generated alt text for screen readers.
            Defaults to ["html"]. Currently supported methods are "html", "markdown", "new_cell",
            "txt_file", "img_file". See add_alt_text for more details about each method.
            If "md_table" or "table" is included in methods, a markdown table with the chart's data
            will be included in the alt text.
        desc_level (int, optional):
            The description level to use in alt text descriptions based on Lundgard and
            Satyanarayan 2021. Defaults to 2. Currently supported description levels are:

            1. axis and encoding descriptions
            2. level 1 plus statistics about the chart's data
            3. level 2 plus trends in the data and relationships between variables

        context (str, optional):
            Extra context which will be appended to automatically generated alt text.
        output_file (str|None, optional):
            The output file name to use for img_file and txt_file display methods. If None
            is given, defaults to "mpa_" plus the title of the last matplotlib chart.
            If None is given and there's no title, defaults to "matplotalt_tmp_" plus a hash.
        return_alt (bool, optional):
            Whether this function should return the generated alt text. Otherwise returns None.
            Defaults to False.
        kwargs (optional):
            Extra config options for the generated alt text, including the number of signifigant
            figures, max color descriptions, max markdown table rows, etc...

    Returns:
        str|None: if return_alt, returns the starter alt text for the given fig and axs.
        If there are multiple axis objects in axs, returns the suptitle followed by alt text
        for each axis and their number. Otherwise returns None.
    """
    if "table" in methods or "md_table" in methods:
        kwargs["include_table"] = True
    if not alt_text:
        alt_text = generate_alt_text(axs=axs, fig=fig, chart_type=chart_type,
                                     desc_level=desc_level, **kwargs)
    add_alt_text(alt_text + context, methods=methods, output_file=output_file)
    # So returned string is not displayed as cell output by default
    plt.clf()
    if return_alt or len(methods) == 0:
        return alt_text


def get_api_chart_type(api_key, base64_img, model="gpt-4-vision-preview", use_azure=False):
    """ Returns the type of the current chart image using the given model and credentials.

    Args:
        api_key (str):
            The OpenAI or Azure API key to use when querying models
        base64_img (str):
            The base64 encoded image to classify
        model (str, optional):
            The OpenAI / Azure model name to query to classify the given image.
            Defaults to "gpt-4-vision-preview".
        use_azure (bool, optional):
            Whether the model to query is on Azure. Defaults to False.

    Returns:
        str: The chart type of the given image. Returns "unknown" if chart type could not be inferred.
    """
    chart_types = list(CHART_TYPE_TO_CLASS.keys())
    prompt = f"You are an expert at classifying charts into one of {len(chart_types)} types: {format_list(chart_types)}. What is the type of this chart?"
    api_response =  get_openai_vision_response(api_key, prompt, base64_img, model=model,
                                              use_azure=use_azure, max_tokens=10,
                                              return_full_response=False)
    # Check if the chart type is the first token:
    api_response = api_response.lower().replace(".", "").strip()
    if len(api_response) > 0:
        first_word = api_response.split()[0]
        if first_word in chart_types:
            return first_word
    else:
        return "unknown"
    # Otherwise try and extract the chart type:
    for pattern in CHART_TYPE_ANSWER_PATTERNS:
        matches = re.findall(pattern, api_response)
        if len(matches) > 1:
            return matches[0]
    return "unknown"


def get_desc_level_prompt(desc_level, starter_desc=None, max_tokens=225):
    """ Returns a prompt to generate alt text from a figure based on the given description level.

    Args:
        desc_level (int):
            The description level to use in alt text descriptions based on Lundgard and
            Satyanarayan 2021. Defaults to 2
        starter_desc (str, optional):
            A starter description to use in the prompt. Defaults to None.
        max_tokens (int, optional):
            Adds the sentence 'Limit you response to {max_tokens} tokens.' to the returned
            prompt. Defaults to 200.

    Raises:
        ValueError: if desc_level is not an int from 1-4.

    Returns:
        str: the prompt to use when querying models to generate alt text from a figure.
    """
    # TODO: provide examples for encodings, chart type, axis ranges,
    base_prompt = "You are a helpful assistant that describes figures. Here are two example descriptions:\n"
    base_prompt += "1. 'This is a vertical bar chart entitled \'COVID-19 mortality rate by age\' that plots Mortality rate by Age. Mortality rate is plotted on the vertical y-axis from 0 to 15%. Age is plotted on the horizontal x-axis in bins: 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80+. The highest COVID-19 mortality rate is in the 80+ age range, while the lowest mortality rate is in 10-19, 20-29, 30-39, sharing the same rate. COVID-19 mortality rate does not linearly correspond to the demographic age. The mortality rate increases with age, especially around 40-49 years and upwards. This relates to people’s decrease in their immunity and the increase of co-morbidity with age. The mortality rate increases exponentially with older people.'\n"
    base_prompt += "2. 'This is a line chart titled \'Big Tech Stock Prices\' that plots price by date. The corporations include AAPL (Apple), AMZN (Amazon), GOOG (Google), IBM (IBM), and MSFT (Microsoft). The years are plotted on the horizontal x-axis from 2000 to 2010 with an increment of 2 years. The prices are plotted on the vertical y-axis from 0 to 800 with an increment of 200. GOOG has the greatest price over time. MSFT has the lowest price over time. Prices of particular Big Tech corporations seem to fluctuate but nevertheless increase over time. Years 2008-2009 are exceptions as we can see an extreme drop in prices of all given corporations. The big drop in prices was caused by financial crisis of 2007-2008. The crisis culminated with the bankruptcy of Lehman Brothers on September 15, 2008 and an international banking crisis.'\n\n"
    #base_prompt = "You are a helpful assistant that describes figures. Here is an example description:\n"
    #base_prompt += "'This is a line chart titled \'Big Tech Stock Prices\' that plots price by date. The corporations include AAPL (Apple), AMZN (Amazon), GOOG (Google), IBM (IBM), and MSFT (Microsoft). The years are plotted on the horizontal x-axis from 2000 to 2010 with an increment of 2 years. The prices are plotted on the vertical y-axis from 0 to 800 with an increment of 200. GOOG has the greatest price over time. MSFT has the lowest price over time. Prices of particular Big Tech corporations seem to fluctuate but nevertheless increase over time. Years 2008-2009 are exceptions as we can see an extreme drop in prices of all given corporations. The big drop in prices was caused by financial crisis of 2007-2008. The crisis culminated with the bankruptcy of Lehman Brothers on September 15, 2008 and an international banking crisis.'\n\n"
    # The part of the prompt describing axis ticks and data
    data_prompt = ""
    if starter_desc != None:
        data_prompt += f"You already know the following information about this figure and its data: '{starter_desc}'.\n\n"
    data_prompt += " Describe this figure."
    # The part of the prompt describing which details to include
    if desc_level == 1:
        info_prompt = "Only include information about the chart type, colors, sizes, textures, title, axis ranges, and labels."
    elif desc_level == 2:
        info_prompt = "Include information about the chart type, colors, sizes, textures, title, axis ranges, and labels. If possible, describe statistics, extrema, outliers, correlations, and point-wise comparisons between variables."
    elif desc_level == 3:
        info_prompt = "Include information about the chart type, colors, sizes, textures, title, axis ranges, and labels. If possible, describe statistics, extrema, outliers, correlations, point-wise comparisons, and trends for each plotted variable."
    elif desc_level == 4:
        info_prompt = "Include information about the chart type, colors, sizes, textures, title, axis ranges, and labels. If possible, Describe statistics, extrema, outliers, correlations, point-wise comparisons, and trends for each plotted variable. If possible, briefly explain domain-specific insights, current events, and socio-political context that explain the data."
    else:
        raise ValueError(f"Unsupported desc_level: {desc_level}")
    # Combine the prompt parts
    full_prompt = f"{base_prompt.strip()} {data_prompt.strip()} {info_prompt.strip()}"
    full_prompt += f"Be concise and limit you response to {max_tokens} tokens."
    return full_prompt


# TODO: add an error for not providing the api key, also automatically pull from the env (print a log statement)
#       add instructions for creating the env
# TODO: Add the role: 'You are a helpful assistant you describe figures. Only include statistics based on the given data'
def generate_api_alt_text(api_key, prompt=None, fig=None, desc_level=4, chart_type=None,
                         model="gpt-4-vision-preview", use_azure=False,
                         use_starter_alt_in_prompt=True, include_table=False,
                         max_tokens=225, **kwargs):
    """
    Return AI generated alt text for the current figure and axes.

    Args:
        api_key (str, optional):
            The API key to use when querying models. If None and use_azure is True, uses the
            environment variable AZURE_OPENAI_API_KEY. Otherwise uses the variable OPENAI_API_KEY.
            Defaults to None
        prompt (str, optional):
            The prompt to use when querying the model to generate alt text. If None is given,
            a prompt is chosen based on the given description level.
        fig (matplotlib.figure.Figure, optional):
            The matplotlib figure object containing information about higher level details if
            there are multiple subplots with unique axes. If no fig is provided, it is inferred
            from plt.gcf()
        desc_level (int, optional):
            The description level to use in alt text descriptions based on Lundgard and
            Satyanarayan 2021. Currently supported description levels are:

            1. axis and encoding descriptions
            2. level 1 plus statistics about the chart's data
            3. level 2 plus trends in the data and relationships between variables
            4. level 3 plus broader context which explains the data

            Defaults to 4. Note that this parameter only changes the prompt and is
            not guarenteed to result in the desired level of description.
        chart_type (str, optional):
            The type of chart to describe. If no chart type is given, it is inferred automatically.
        model (str, optional):
            The model to use when generating alt text. Defaults to gpt-4-vision-preview. Must
            be a model which can handle image inputs.
        use_azure (bool, optional):
            Whether to use azure openai instead of openai. Defaults to False.
        use_starter_alt_in_prompt (bool, optional):
            Whether to use heuristic-generated alt text for the current figure in the prompt.
        include_table (bool, optional):
            Whether to include a markdown table with the chart's data in both starter and
            VLM-generated alt texts. Defaults to False.
        max_tokens (int, optional):
            The maximum number of tokens in the VLM-generated response. Defaults to 225.

    Returns:
        str: The VLM-generated alt text for the matplotlib figure. If there are
        multiple axis objects in axs, returns the suptitle followed by alt text for each
        axis and their number.
    """
    if api_key == None:
        env_var = "OPENAI_API_KEY"
        if use_azure:
            env_var = "AZURE_OPENAI_API_KEY"
        api_key = os.environ.get(env_var)
        print(f"Using the api key from the environment variable {env_var}")
        if api_key == None:
            raise ValueError(f"No api_key provided or variable for {env_var}")
    # Infer fig and raise a value error if there are no axes (fig is blank)
    if not fig:
        fig = plt.gcf()
    if len(fig.get_axes()) == 0:
        raise ValueError("Given axes are blank")
    # Get the base64 image from the canvas
    fig.canvas.draw()
    pil_img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    base64_img = pillow_image_to_base64_string(pil_img)
    # Optionally include starter alt text from a template and/or figure data in the prompt
    starter_desc = None
    data_md_table = ""
    if (use_starter_alt_in_prompt and (prompt in [None, ""])) or include_table:
        chart_desc = generate_alt_text(chart_type=chart_type, desc_level=desc_level,
                                       include_table=include_table, **kwargs)
        table_start = chart_desc.lower().find("data table:")
        # Extract the data table from the chart desc if possible
        if include_table and table_start != -1:
            data_md_table = "\n\n" + chart_desc[table_start:]
            chart_desc = chart_desc[:table_start]
        if use_starter_alt_in_prompt:
            starter_desc = chart_desc

    if prompt in [None, ""]:
        prompt = get_desc_level_prompt(desc_level, starter_desc=starter_desc, max_tokens=max_tokens)
    api_response = get_openai_vision_response(api_key, prompt, base64_img, model=model,
                                              use_azure=use_azure, max_tokens=max_tokens,
                                              return_full_response=False)
    api_response = "This description was generated by a language model. " + api_response
    #api_response = insert_line_breaks(api_response.strip(), max_line_width=max_alt_line_width)
    return api_response + data_md_table


def show_with_api_alt(api_key=None, prompt=None,
                      fig=None, desc_level=4, chart_type=None,
                      model="gpt-4-vision-preview", use_azure=False,
                      use_starter_alt_in_prompt=True, methods=["html"],
                      max_tokens=225, context="", output_file=None,
                      return_alt=False, **kwargs):
    """
    Return and surface AI generated alt text for the current matplotlib figure.

    Args:
        api_key (str, optional):
            The API key to use when querying models. If None and use_azure is True, uses the
            environment variable AZURE_OPENAI_API_KEY. Otherwise uses the variable OPENAI_API_KEY.
            Defaults to None
        prompt (str, optional):
            The prompt to use when querying the model to generate alt text. If None is given,
            a prompt is chosen based on the given description level.
        fig (matplotlib.figure.Figure, optional):
            The matplotlib figure object containing information about higher level details if
            there are multiple subplots with unique axes. If no fig is provided, it is inferred
            from plt.gcf()
        desc_level (int, optional):
            The description level to use in alt text descriptions based on Lundgard and
            Satyanarayan 2021. Currently supported description levels are:

            1. axis and encoding descriptions
            2. level 1 plus statistics about the chart's data
            3. level 2 plus trends in the data and relationships between variables
            4. level 3 plus broader context which explains the data

            Defaults to 4. Note that this parameter only changes the prompt and is
            not guarenteed to result in the desired level of description.
        chart_type (str, optional):
            The type of chart to describe. If no chart type is given, it is inferred automatically.
        model (str, optional):
            The model to use when generating alt text. Defaults to gpt-4-vision-preview. Must
            be a model which can handle image inputs.
        use_azure (bool, optional):
            Whether to use azure openai instead of openai. Defaults to False.
        methods (list[str], optional):
            The methods used to display the generated alt text for screen readers.
            Defaults to ["html"]. Currently supported methods are "html", "markdown", "new_cell",
            "txt_file", "img_file". See add_alt_text for more details about each method.
        use_starter_alt_in_prompt (bool, optional):
            Whether to use heuristic-generated alt text for the current figure in the prompt.
        max_tokens (int, optional):
            The maximum number of tokens in the VLM-generated response. Defaults to 225.
        context (str, optional):
            Extra context which will be appended to automatically generated alt text.
        output_file (str|None, optional):
            The output file name to use for img_file and txt_file display methods. If None
            is given, defaults to "mpa_" plus the title of the last matplotlib chart.
            If None is given and there's no title, defaults to "matplotalt_tmp_" plus a hash.
        return_alt (bool, optional):
            Whether this function should return the generated alt text. Otherwise returns None.
            Defaults to False.
        kwargs (optional):
            Extra config options for the generated alt text, including the number of signifigant
            figures, max color descriptions, whether to include data as a markdown table, etc...

    Returns:
        str|None: If return_alt is True, returns the AI generated alt text for the matplotlib
        figure and axis. If there are multiple axis objects in axs, returns the suptitle followed
        by alt text for each axis and their number. If return_alt is False, returns None.
    """
    if "table" in methods or "md_table" in methods:
        kwargs["include_table"] = True
    ai_alt_text = generate_api_alt_text(api_key, prompt=prompt,
                         fig=fig, desc_level=desc_level, model=model,
                         chart_type=chart_type,
                         use_starter_alt_in_prompt=use_starter_alt_in_prompt,
                         use_azure=use_azure, max_tokens=max_tokens,
                         **kwargs)
    add_alt_text(ai_alt_text + context, methods=methods, output_file=output_file)
    # So returned string is not displayed as cell output by default
    plt.clf()
    if return_alt or len(methods) == 0:
        return ai_alt_text