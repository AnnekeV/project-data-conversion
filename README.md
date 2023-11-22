# Project for converting data from ctd and mooring to netcdf

The goal of this 




## Dependencies

python
in deep_water_simple.yml
install dependencies in a virtual environment
`code here`

```python
conda env create -f deep_water_simple.yml
```






## Usage

You should be able to use python and know how virtual environments work 
`conda activate deep_water`

1. Put CTD data in `data/raw`
2. Run `ctd_monthly.py`

## Project Structure

The project structure distinguishes three kinds of folders:
- read-only (RO): not edited by either code or researcher
- human-writeable (HW): edited by the researcher only.
- project-generated (PG): folders generated when running the code; these folders can be deleted or emptied and will be completely reconstituted as the project is run.


```
.
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── data               <- All project data, ignored by git
│   ├── processed      <- The final, canonical data sets for modeling. (PG)
│   ├── raw            <- The original, immutable data dump. (RO)
│   └── temp           <- Intermediate data that has been transformed. (PG)
├── docs               <- Documentation notebook for users (HW)
│   ├── manuscript     <- Manuscript source, e.g., LaTeX, Markdown, etc. (HW)
│   └── reports        <- Other project reports and notebooks (e.g. Jupyter, .Rmd) (HW)
├── results
│   ├── figures        <- Figures for the manuscript or reports (PG)
│   └── output         <- Other output for the manuscript or reports (PG)
└── src                <- Source code for this project (HW)

```

## Add a citation file
Create a citation file for your repository using [cffinit](https://citation-file-format.github.io/cff-initializer-javascript/#/)

## License

This project is licensed under the terms of the [MIT License](/LICENSE).


## Origin of the project structure (from workshop)
This simple project structure template repository is adapted from the [Good Enough Project](https://github.com/bvreede/good-enough-project) Cookiecutter template by Barbara Vreede (2019).
If you plan to develop a package, check the [template repository for a Python package](https://github.com/UtrechtUniversity/re-python-package).

