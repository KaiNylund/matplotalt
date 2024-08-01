.. matplotalt documentation main file, created by
   sphinx-quickstart on Tue Jul  2 10:08:35 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to matplotalt's documentation!
======================================

matplotalt's main function for generating and surfacing alt text is :func:`matplotalt.show_with_alt`, with examples in :ref:`Matplotalt examples`.

To generate alt text using an LLM, matplotalt also provides the :func:`matplotalt.show_with_api_alt` function, with similar examples in :ref:`Matplotalt LLM examples`.

matplotalt includes the ``alttextify`` command to automatically add alt text to each figure in a notebook, documented in :ref:`The alttextify command`.

Finally, other useful functions like :func:`matplotalt.infer_chart_type` are documented in the :ref:`matplotalt API reference`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   notebooks/examples
   notebooks/llm_examples
   alttextify
   api
   matplotalt

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
