# matplotalt

matplotalt is a Python library for automatically generating and displaying alt text for matplotlib figures. matplotalt is designed to make figures in IPython notebooks more accessiblte to screen readers, but also works in Python scripts.

## Installation

(Coming soon!) The latest release can be installed from PyPI:

``` pip install matplotalt ```

## Examples

Documentation is available at [matplotalt's read-the-docs page](https://matplotalt.readthedocs.io), including [examples](https://matplotalt.readthedocs.io/en/latest/notebooks/examples.html), [API reference](https://matplotalt.readthedocs.io/en/latest/api.html), and other useful information.

matplotalt's ``generate_alt_text`` function will automatically generate alt text for the most recent matplotlib figure. The desc_level parameter controls how detailed the figure description is from 1 (least detail) to 3 (most) based on [Lundgard and Satyanarayan, 2021](https://arxiv.org/pdf/2110.04406).

```
def sunshine_bars():
    sunshine_hours = [69, 108, 178, 207, 253, 268, 312, 281, 221, 142, 72, 52]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sep", "Oct", "Nov", "Dec"]
    plt.title("Monthly Sunshine in Seattle")
    plt.barh(list(range(12)), sunshine_hours)
    plt.xlabel("Average hours of sunlight")
    plt.ylabel("Month")
    plt.yticks(ticks=list(range(0, 12)), labels=months)

sunshine_bars()
sunshine_alt = generate_alt_text(desc_level=3)
```

Use the ``surface_alt_text`` function to make alt texts visable to screen readers in the notebook environment:

```
surface_alt_text(sunshine_alt, methods=["html", "img_file"])
```

Currently supported methods to display alt text are:

* "html": displays the last figure in html with an alt property containing the given text.
* "markdown": display text in markdown in the current cell output.
* "new_cell": create a new (code) cell after this one containing the markdown magic followed by the given text.
* "txt_file": writes the given text to a text file.
* "img_file": saves the last matplotlib figure with the given text in its alt property.

``show_with_alt`` combines ``generate_alt_text`` and ``surface_alt_text`` functions and is designed to replace calls to ``matplotlib.pyplot.show()``.

```
sunshine_bars()
show_with_alt(desc_level=3, methods=["html", "table"])
```

There are also "API" versions of each function (``show_with_api_alt``, ``generate_api_alt``) which generate alt text using a LLM through OpenAI and Azure APIs.

matplotalt provides the ``alttextify`` command to automatically add alt text to each matplotlib figure in a IPython notebook. For example,

```
alttextify ./examples/examples_no_alt.ipynb examples_with_alt -s html
```

will add alt text to each figure in the examples_no_alt.ipynb notebook through the HTML alt property without changing any of the code cells. ``alttextify`` can also be used to generate alt texts with LLMs:

```
alttextify ./examples/examples_no_alt.ipynb examples_with_llm_alt -k <your API key> -m gpt-4-vision-preview -us
```

## Motivation

Visualizations on the web generated by code often lack alt text. Across 100000 Jupyter notebooks, [Potluri et al., 2023](https://dl.acm.org/doi/pdf/10.1145/3597638.3608417), found that 99.81% of programmatically generated images did not have associated alt text. Of these, the vast majority were created with matplotlib or seaborn, neither of which contain methods for easily generating or displaying image descriptions. This is a critical barrier to perceiving computational notebooks for blind and visually impaired (BVI) users. The goal of this package is to provide an alternative that allows users to automatically generate screen reader visable alt text for matplotlib figures in computational notebooks.