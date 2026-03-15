# ruppert_2026_precipeff

This repository contains all data and code required to generate all figures in the study by Ruppert et al. 2026 (submitted), including the main figure set and all Supporting Information (SI) figures.

All code is written in python and organized in a combination of Jupyter notebooks (\*.ipynb) and raw python scripts (\*.py). The Jupyter notebooks in the main directory are the drivers for generating each figure, named corresponding to the figure or figures they generate. The generation of all SI figures is also contained in these notebooks. All python package dependencies are either publicly accessible through conda-forge or are provided directly as python scripts here.

The data (located under data/) is a combination of pickle format (Python object serialization; \*pkl), raw text (\*txt), and CSV. The driver notebooks include any necessary read functions for importing all datasets.

Once all python package dependencies are installed, each notebook can be run to completion, which will generate plots within the notebooks and save \*.png files under figures/.