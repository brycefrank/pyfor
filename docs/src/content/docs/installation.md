---
title: Installation
description: Installing pyfor and its optional dependencies.
---

pyfor requires Python 3.10 or newer. All of its dependencies are available as binary wheels, so a
virtual environment and `pip` are all that is needed:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install pyfor
```

To work on pyfor itself, install the clone in editable mode instead:

```bash
git clone https://github.com/brycefrank/pyfor.git
cd pyfor
pip install -e ".[test]"
```

The 3D plotting methods (`Cloud.plot3d`) need the optional `plot` dependencies, which are not
installed by default:

```bash
pip install "pyfor[plot]"
```

Following these commands, pyfor should load in an activated Python shell:

```python
import pyfor
```

If you see no errors, you are ready to process.

A conda environment file is also provided for those who prefer conda:

```bash
conda env create -f environment.yml
```
