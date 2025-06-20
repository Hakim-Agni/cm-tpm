**cm-tpm** (Continuous Mixtures of Tractable Probabilistic Models) is a Python package designed for efficient and scalable data imputation using probabilistic circuits. The package provides a flexible and user-friendly API, making it easy to integrate into data preprocessing pipelines. The package is distributed under the MIT licence.

The project was started in 2025 by Hakim Agni, under the supervision of Thomas Krak at Eindhoven University of Technology, as part of a Master's Thesis. It implements the data imputation method described in the paper [Continuous Mixtures of Tractable Probabilistic Models](https://arxiv.org/abs/2209.10584), by Alvaro Correia, Gennaro Gala, Erik Quaeghebeur, Cassio de Campos and Robert Peharz.

Installation
------------

### Dependencies

cm-tpm requires:

- Python (>= 3.10)
- NumPy (>= 1.22.4)
- PyTorch (>= 2.6.0)
- SciPy (>= 1.13.0)
- pandas (>= 2.2.2)


### User installation

The easiest way to install cm-tpm is using ``pip``:
```bash
pip install cm-tpm
```
<!-- or using ``conda``:
```bash
conda install cm-tpm
``` -->

### Optional Dependencies
To install all optional dependencies:
```bash
pip install cm-tpm[all]
```
You can also install them individually:
- Excel file support (``.xlsx``), install
``openpyxl``:
``` 
pip install cm-tpm[excel] 
```
- Parquet and Feather file support (``.parquet``. ``.feather``), install ``pyarrow``:
```
pip install cm-tpm[parquet]
```
- Progress bars (for verbose/debug mode), install ``tqdm``:
```
pip install cm-tpm[tqdm]
```



Development
-----------
### Source code
You can check the latest source code with the command:
```
git clone https://github.com/Hakim-Agni/cm-tpm.git
```

Install development dependencies:
```
pip install -r 'requirements.txt'
```

### Testing
After installation, you can launch the test suite from outside the source directory (this requires ``pytest`` to be installed):
```
pytest tests
```


Help and Support
----------------
### Documentation
The documentation supporting this package can be found in the wiki page on GitHub.

- **Full documentation** : https://github.com/Hakim-Agni/cm-tpm/wiki
- **User Guide** : https://github.com/Hakim-Agni/cm-tpm/wiki/User-Guide
- **API Documentation** : https://github.com/Hakim-Agni/cm-tpm/wiki/API-documentation

### Communication
If you have any questions or feedback, feel free to reach out via:

- **GitHub Discussions** : https://github.com/Hakim-Agni/cm-tpm/discussions
