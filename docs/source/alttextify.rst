The alttextify command
======================

| matplotalt includes the ``alttextify`` command to automatically add alt text to each figure in a notebook. For example,
|    ``alttextify ./examples_no_alt.ipynb examples_with_alt -s html img_file``
| will run :func:`matplotalt.show_with_alt` on each cell with a matplotlib figure output and save the notebook with alt texts to ``examples_with_alt.ipynb``

| An api key can be passed to ``alttextify`` to generate alt text using an LLM instead. For instance,
|    ``alttextify ./examples_no_alt.ipynb examples_with_llm_alt -k <your API key> -m gpt-4-vision-preview -us``
| will add alt text to each figure using :func:`matplotalt.show_with_api_alt` and the given model.

Additional notes:

 * Only cell outputs will be changed after running ``alttextify`` on a notebook. All code in cells will stay the same.
 * Alt text will not be added to cells with errors
 * Any cells that take longer than timeout seconds to run will be skipped

usage: alttextify [-h] [-l DESC_LEVEL] [-s SURFACE_METHODS [SURFACE_METHODS ...]] [-ht] [-t TIMEOUT] [-c CONTEXT]
                  [-k API_KEY] [-m MODEL] [-p PROMPT] [-us] [-ua]
                  notebook_path [output_path]

| positional arguments:
|  **notebook_path**         path of notebook to alttextify
|  **output_path**           path to outputted notebook with figure alt text. If none is given, defaults to the notebook name + '_with_alt'. If --to_html, this is also the name of the outputted html file

options:
  -h, --help            show this help message and exit
  -l DESC_LEVEL, --desc_level DESC_LEVEL
                        The description semantic level from Lundgard & Satyanarayan 2022. Level 1 includes encoding
                        and axis descriptions. Level 2 adds statistics for the underlying data. Level 3 adds trends
                        and variable comparisons.
  -s, --surface_methods
                        How the alt description should be surfaced in the notebook. Choices are 'html', 'markdown',
                        'new_cell', 'txt_file', 'img_file', 'md_table'
  -ht, --to_html        If given, notebook is also exported to html at the given file path.
  -t TIMEOUT, --timeout TIMEOUT
                        timeout for running each cell
  -c CONTEXT, --context CONTEXT
                        text appended to alt text generated for each plot
  -k API_KEY, --api_key API_KEY
                        The OpenAI / Azure key to use when querying the LLM. If a key is given, will use
                        show_with_api_alt, else uses show_with_alt
  -m MODEL, --model MODEL
                        The name of the LLM to use to generate alt text.
  -p PROMPT, --prompt PROMPT
                        The prompt given to the LLM. If none is given, it is automatically generated based on the
                        desc_level. Note, adding a prompt through this field will overwrite starter alt text from
                        --use_starter_alt
  -us, --use_starter_alt
                        Whether starter heuristic-based alt text should be used in the prompt to the LLM
  -ua, --use_azure      Whether to load the model from microsoft Azure.
