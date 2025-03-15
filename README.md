# PackageHierarchyRecovery-2025

This GitHub repository contains the files for the bachelor's thesis "Package hierarchy recovery using word embeddings for flattened remodularized Java systems," conducted by Sem Stammen at Radboud University (2024/2025). The materials provided here enable replication and verification of the study's findings.

## Setup

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pipenv.

```bash
pip install pipenv
```

Move into the `proposedmethod` directory.

```bash
cd proposedmethod
```

Use pipenv to install the project dependencies.

```
pipenv sync
```

## Usage
Use pipenv to run `packageorganizer.py` and `packageanalyzer.py`.

```python
pipenv run py packageorganizer.py
```

```python
pipenv run py packageanalyzer.py
```


## License

[MIT](https://choosealicense.com/licenses/mit/)