import argparse
import nbformat
import os
import re
from copy import deepcopy
from nbconvert.preprocessors import ExecutePreprocessor

def main():
    # Example usage: alttextify examples.ipynb -l 3 --to_html
    # Example API usage: alttextify examples.ipynb examples_api_alt.ipynb -k {Your API key} -m "gpt-4o" -us
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook_path", help="path of notebook to alttextify")
    parser.add_argument("output_path", nargs='?', default=None, help="path to outputted notebook with figure alt text. If none is given, defaults to the notebook name + '_with_alt'. If --to_html, this is also the name of the outputted html file")
    parser.add_argument('-l', '--desc_level', type=int, default=2, help="The description semantic level from Lundgard & Satyanarayan 2022. Level 1 includes encoding and axis descriptions. Level 2 adds statistics for the underlying data. Level 3 adds trends and variable comparisons.")
    parser.add_argument('-s', '--surface_methods', nargs='+', default=["html"], help="How the alt description should be surfaced in the notebook. Choices are 'html', 'markdown', 'new_cell', 'txt_file', 'img_file', 'md_table'")
    parser.add_argument('-ht', '--to_html', default=False, action='store_true', help="If given, notebook is also exported to html at the given file path.")
    parser.add_argument('-t', '--timeout', type=int, default=600, help="timeout for running each cell")
    parser.add_argument('-c', '--context', type=str, default="", help="text appended to alt text generated for each plot")
    parser.add_argument('-k', '--api_key', type=str, default=None, help="The OpenAI / Azure key to use when querying the LLM. If a key is given, will use show_with_api_alt, else uses show_with_alt")
    parser.add_argument('-m', '--model',   type=str, default="gpt-4-vision-preview", help="The name of the LLM to use to generate alt text.")
    parser.add_argument('-p', '--prompt',  type=str, default=None, help="The prompt given to the LLM. If none is given, it is automatically generated based on the desc_level. Note, adding a prompt through this field will overwrite starter alt text from --use_starter_alt")
    parser.add_argument('-us', '--use_starter_alt', default=False, action='store_true', help="Whether starter heuristic-based alt text should be used in the prompt to the LLM")
    parser.add_argument('-ua', '--use_azure', default=False, action='store_true', help="Whether to load the model from microsoft Azure.")
    args = parser.parse_args()

    if args.output_path == None:
        args.output_path = f"{args.notebook_path.split('.')[0]}_with_alt.ipynb"


    class MatplotaltPreprocessor(ExecutePreprocessor):
        global args
        def __init__(self, *nonkwargs, **kwargs):
            super().__init__(*nonkwargs, **kwargs)
            print(args)
            print(str(args.surface_method))
            self.alt_textify_snippet = "\n\nimport matplotlib.pyplot as plt\nif len(plt.gcf().axes) > 0:\n\t"
            if args.api_key is None:
                self.alt_textify_snippet += f"from matplotalt import show_with_alt\n\tshow_with_alt(methods={str(args.surface_method)}, desc_level={args.desc_level}, context={args.context})"
            else:
                self.alt_textify_snippet += f"from matplotalt import show_with_api_alt\n\tshow_with_api_alt(api_key={args.api_key}, model={args.model}, prompt={args.prompt}, methods={str(args.surface_method)}, desc_level={args.desc_level}, context={args.context}, use_starter_alt_in_prompt={args.use_starter_alt}, use_azure={args.use_azure})"
            self.found_aliases = False
            # import matplotlib.pyplot as plt
            self.pyplot_alias = "plt"
            # import matplotlib as mpl
            self.matplotlib_alias = "matplotlib"


        def preprocess_cell(self, cell, resources, index):
            """
            Override if you want to apply some preprocessing to each cell.
            Must return modified cell and resource dictionary.

            Args:
                cell (NotebookNode cell): Notebook cell being processed
                resources (dict): Additional resources used in the conversion process.
                    Allows preprocessors to pass variables into the Jinja engine.
                index (int): Index of the cell being processed

            Returns:
                cell (NotebookNode cell): Notebook cell after being processed
                resources (dict): Additional resources used in the conversion process.
            """
            self._check_assign_resources(resources)
            unmodified_source = deepcopy(cell["source"])
            # Check for pyplot/matpllib aliases if we haven't found any yet
            if not self.found_aliases:
                pyplot_alias_matches = re.findall(r'(import matplotlib.pyplot as)\s+(.*)', unmodified_source)
                matplotlib_alias_matches = re.findall(r'(import matplotlib as)\s+(.*)', unmodified_source)
                if len(pyplot_alias_matches) > 0:
                    self.pyplot_alias = pyplot_alias_matches[0][1]
                    self.found_aliases = True
                if len(matplotlib_alias_matches) > 0:
                    self.matplotlib_alias = matplotlib_alias_matches[0][1]
                    self.found_aliases = True
            if cell["cell_type"] == "code" and "source" in cell and len(cell["source"]) > 0:
                # Try to replace calls to "plt.show()" with the alttextify snippet\
                cell["source"] = cell["source"].replace(f"{self.pyplot_alias}.show()", self.alt_textify_snippet)
                # If we couldn't find plt.show(), try to find and replace matplotlib.pyplot.show()
                if len(cell["source"]) == len(unmodified_source):
                    cell["source"] = cell["source"].replace(f"{self.matplotlib_alias}.pyplot.show()", self.alt_textify_snippet)
                # If we didn't replace any calls to .show(), just add the snippet at the end of the cell
                if len(cell["source"]) == len(unmodified_source):
                    cell["source"] = cell["source"] + self.alt_textify_snippet
            cell = self.execute_cell(cell, index, store_history=True)
            cell["source"] = unmodified_source
            return cell, self.resources


    ep = MatplotaltPreprocessor(timeout=args.timeout)
    with open(args.notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    ep.preprocess(nb)
    with open(f"{args.output_path}", 'w', encoding="utf-8") as f:
        nbformat.write(nb, f)

    if args.to_html:
        os.system(f"jupyter nbconvert --to html {args.output_path.split('.')[0]}")


if __name__ == "__main__":
    main()