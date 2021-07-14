# MLP-Tool

This project aims to add a graphical user interface to the scikit-learn [MLPRegressor method](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
and can be used to train simple MLP networks from training and test datasets.
The finished network can be exported as PMML file.
It was developed during my internship semester at the IPM institute from the University of Applied Sciences (HSZG).

## Main features

- Import of Excel files
- Configuration of the most relevant hyperparameters
- Optional per-feature scaling of all input data
- Evaluation via Matplotlib graphs for loss curve and error values
- Comparison between prediction and target data as 3D point cloud
- Statistical evaluation of training and test results
- Weights and Bias values can be viewed as tables and exported to Excel as matrices
- Export of the network as PMML file

## Installation

The Python script can be run directly, but requires a Python installation and all necessary packages.
To deploy the MLP Tool to a wider range on computers, it should be packaged with PyInstaller from a dedicated virtual environment that contains the following packages:

```pyinstaller,  sys,  time,  pyqt5,  pandas,  matplotlib,  seaborn, numpy,  math,  scikit-learn,  nyoka,  openpyxl,  webbrowser,  xlsxwriter```

