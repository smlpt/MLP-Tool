"""
MLP TOOL - training MLP models and exporting them to PMML
Copyright (C) 2021 Hochschule Zittau/Görlitz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses

"""

import sys
import os.path

# Measure elapsed training time
from time import perf_counter

from PyQt5.QtWidgets import (
    QApplication,       # base application      
    QStatusBar,         # status bar in main window
    QMainWindow,        # main window
    QFileDialog,        # dialog for import/saving
    QDialog,            # dialogs (e.g. statistics and weights/bias)
    QLabel,             # text label
    QHBoxLayout,        # horizontal layout
    QVBoxLayout,        # vertical layout
    QTableWidget,       # table
    QTableWidgetItem,   # item in the table
    QAbstractScrollArea,# Needed to resize the table widget
    QSizePolicy,        # Adjust dynamic size of widgets and windows
    QMessageBox,        # Show error messages
    QPushButton         # Generic button type
)
from PyQt5.QtGui import (
    QFont,              # Change font parameters
    QIcon,              # Add window icon
    QPixmap             # Show image in label (for "About" dialog)
)
from PyQt5 import uic   # import UI file
from PyQt5.QtCore import (
    Qt                  # needed for alignment in Boxlayout
)  

# method "read_excel" requires the openpyxl library to be installed
from pandas import DataFrame, read_excel, to_numeric, ExcelWriter
# Print 2D plots
import matplotlib.pyplot as plt
# Provide fancier plot designs
from seaborn import set_style, lineplot
from numpy import array, array2string
from math import sqrt

# Methods for training the network
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

#! Fix for "PMML44 NOT FOUND in pyinstaller": Add nyoka folder as --paths= argument
from nyoka import skl_to_pmml

# Open Documentation in browser
import webbrowser


# Main Function
def main():

    # Create an instance of QtWidgets.QApplication 
    app = QApplication(sys.argv)    

    # Create list for all necessary files
    files = [
        'mlp_tool.ui',
        'MLP_Tool_Icon_V2_64.png',
        'Documentation_DE.html',
        'Documentation_EN.html',
        'dark.qss'
    ]
    # Initialise variable to check if file is missing 
    missing_file = False
    # Check if all files are present
    for file in files:
        if not os.path.exists(file):
            missing_file = True
    # If one or more files are missing:
    if missing_file:
        # Create messagebox and populate it with text
        message = QMessageBox()
        message.setIcon(QMessageBox.Warning)
        message.setWindowTitle("Warning")
        message.setText(
            "One or more program files could not be found,\n"
            "the program might not work as intended.\n"
            "Try downloading the program again.")
        message.setStandardButtons(QMessageBox.Ignore | QMessageBox.Abort)
        # Show messagebox and await user response
        returnvalue = message.exec_()
        # Exit program if user clicked "abort"
        if returnvalue == QMessageBox.Abort:
            exit()
        # Continue to main GUI when user clicked "ignore"
        if returnvalue == QMessageBox.Ignore:
            pass

    # Create an instance of the interface class
    window = GUI()    
    window.show()
    # Start the application
    sys.exit(app.exec_())       


# Class containing the statistics dialog
class StatDialog(QDialog):

    def __init__(self, Ytrain, Ytest, predtrain, predtest, errortrain, errortest, style, parent=None):

        super().__init__(parent=parent)

        self.setWindowTitle("Model Statistics")
        self.setStyleSheet(style)
        # Load icon from main GUI
        self.setWindowIcon(GUI().set_icon())

        # The statistics dialog contains two column pairs, for training and for test statistics.
        layout = QHBoxLayout()
        
        # Add widgets with labels for showing the data columns
        layout.addWidget(QLabel(
            "Training Statistics:\n\n"
            f"R2 Score:\n"
            f"Min. Deviation:\n"
            f"Max. Deviation:\n"
            f"Mean Abs. Error:\n"
            f"Mean Square Error:\n"
            f"RMS Error:\n"
        ))
        layout.addWidget(QLabel(
            "\n\n"
            f"{r2_score(Ytrain, predtrain):.4f}\n"
            f"{min(errortrain):.4f}\n"
            f"{max(errortrain):.4f}\n"
            f"{mean_absolute_error(Ytrain, predtrain):.4f}\n"
            f"{mean_squared_error(Ytrain, predtrain):.4f}\n"
            f"{sqrt(mean_squared_error(Ytrain, predtrain)):>.4f}\n"
        ))
        layout.addWidget(QLabel(
            "Test Statistics:\n\n"
            f"R2 Score:\n"
            f"Min. Deviation:\n"
            f"Max. Deviation:\n"
            f"Mean Abs. Error:\n"
            f"Mean Square Error:\n"
            f"RMS Error:\n"
        ))
        layout.addWidget(QLabel(
            "\n\n"
            f"{r2_score(Ytest, predtest):.4f}\n"
            f"{min(errortest):.4f}\n"
            f"{max(errortest):.4f}\n"
            f"{mean_absolute_error(Ytest, predtest):.4f}\n"
            f"{mean_squared_error(Ytest, predtest):.4f}\n"
            f"{sqrt(mean_squared_error(Ytest, predtest)):.4f}\n"
        ))
        # Increase spacing between columns to maintain readability
        layout.setSpacing(30)
        self.setLayout(layout)


# Class containing the weight/bias dialog
class WeightDialog(QDialog):

    def __init__(self, mlp, style, parent=None):
    
        super().__init__(parent=parent)
        self.setWindowTitle("Weights and Bias")
        self.setStyleSheet(style)
        # Load icon from main GUI
        self.setWindowIcon(GUI().set_icon())
        # Store mlp variable as attribute
        self.mlp = mlp
        # Initialize main layout
        layout = QHBoxLayout()

        # Array structure:
        # Dimension 1: Layer
        # Dimension 2: Neuron
        # Dimension 3: Weight

        # Create 2D array with all neuron connections and corresponding weights
        # Initialize empty array
        weight_data = []
        # Loop through network layers
        for L in range(len(mlp.coefs_)):
            # Loop through layer neurons
            for N in range(len(mlp.coefs_[L])):
                # Loop through neuron connections
                for W in range(len(mlp.coefs_[L][N])):
                    # Add row for input connections
                    if L == 0:
                        weight_data.append([
                            f"Input {N+1}",
                            f"Neuron {L+1}-{W+1}",
                            f"{mlp.coefs_[L][N][W]:.6f}"
                        ])
                    # Add row for output connections
                    elif L==len(mlp.coefs_)-1:
                        weight_data.append([
                            f"Neuron {L}-{N+1}",
                            "Output",
                            f"{mlp.coefs_[L][N][W]:.6f}"
                        ])
                    # Add row for inbetween connections
                    else: 
                        weight_data.append([
                            f"Neuron {L}-{N+1}",
                            f"Neuron {L+1}-{W+1}",
                            f"{mlp.coefs_[L][N][W]:.6f}"
                        ])
        
        # Create weight table and adjust size policies
        weight_table = QTableWidget()
        weight_table.horizontalHeader().setMinimumSectionSize(100)
        weight_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        # Size policy for horizontal and vertical stretching:
        weight_table.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Maximum)
        # Set row count to length of the data table
        weight_table.setRowCount(len(weight_data))
        weight_table.setColumnCount(3)
        weight_table.setHorizontalHeaderLabels(('From', 'To', 'Weight'))
        # Initialize font type and set it to bold (for headings)
        font = QFont()
        font.setBold(True)
        weight_table.horizontalHeader().setFont(font)
        weight_table.setAlternatingRowColors(True)
        
        # Populate weight table with data
        for col in range(3):
            for row in range(len(weight_data)):
                # Create cell item object
                item = QTableWidgetItem(weight_data[row][col])
                # Enable cell selection but disable editing
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                weight_table.setItem(row, col, item)
        
        weight_table.resizeRowsToContents()
        weight_table.resizeColumnsToContents()
        
        # Create 2D array with all neurons and corresponding bias values
        # Initialize empty array
        bias_data = []
        # Loop through network layers
        for L in range(len(mlp.intercepts_)):
            # Loop through layer neurons
            for N in range(len(mlp.intercepts_[L])):
                # Add row for output neurons
                if L == len(mlp.intercepts_)-1:
                    bias_data.append([
                        "Output",
                        f"{mlp.intercepts_[L][N]:.6f}"
                    ])
                # Add rows for the rest of neurons
                else: 
                    bias_data.append([
                        f"Neuron {L+1}-{N+1}",
                        f"{mlp.intercepts_[L][N]:.6f}"
                    ])

        # Create bias table and adjust size policies
        bias_table = QTableWidget()
        bias_table.horizontalHeader().setMinimumSectionSize(100)
        bias_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        # Size policy for horizontal and vertical stretching:
        bias_table.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Maximum)
        # Set row count to length of the data table
        bias_table.setRowCount(len(bias_data))
        bias_table.setColumnCount(2)
        bias_table.setHorizontalHeaderLabels(('Neuron', 'Bias'))
        bias_table.horizontalHeader().setFont(font)
        bias_table.setAlternatingRowColors(True)

        # Populate bias table with data
        for col in range(2):
            for row in range(len(bias_data)):
                # Create cell item object
                item = QTableWidgetItem(bias_data[row][col])
                # Enable cell selection but disable editing
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                bias_table.setItem(row, col, item)

        bias_table.resizeRowsToContents()
        bias_table.resizeColumnsToContents()

        # Add a sublayout as the right column to add the "Save" button
        sublayoutR = QVBoxLayout()
        sublayoutR.addWidget(bias_table, alignment=Qt.AlignTop)
        # Add "Save" button
        SaveMatrixButton = QPushButton("Save Matrices")
        # Connect "SaveMatrix" method to click event
        SaveMatrixButton.clicked.connect(self.save_matrix)
        sublayoutR.addWidget(SaveMatrixButton, alignment=Qt.AlignBottom)
        
        # Add weight table and bias table to layout
        layout.addWidget(weight_table, alignment=Qt.AlignTop)
        layout.addLayout(sublayoutR)

        self.setLayout(layout)
        self.adjustSize()

    # Method for saving the weight/bias tables as excel file
    def save_matrix(self):

        options = QFileDialog.Options()
        address = QFileDialog.getSaveFileName(
            self,"Save Matrix","","Excel Files (*.xlsx)", options=options)

        # Return immediately if address is empty (user clicked "cancel")
        if address[0] == '':
            return

        # Create writer object that contains all excel data before being saved
        writer = ExcelWriter(address[0], engine='xlsxwriter')   

        # Create workbook and worksheet
        workbook = writer.book
        worksheet = workbook.add_worksheet('Matrices')
        writer.sheets['Matrices'] = worksheet

        # Initialize worksheet row
        row = 0

        # Create special formatting for titles
        title_format = workbook.add_format({
            'bold': True,
            'font_size': 12
        })

        # Loop through weight matrices
        for L in range(len(self.mlp.coefs_)):
            # Save the transpose of the current weight matrix in a dataframe
            df = DataFrame(self.mlp.coefs_[L].T)
            # Add matrix titles depending on the layer
            if L == 0:
                worksheet.write(row, 0, f"Input Layer to Layer {L+1}", title_format)
            elif L == len(self.mlp.coefs_)-1:
                worksheet.write(row, 0, f"Layer {L} to Output", title_format)
            else:
                worksheet.write(row, 0, f"Layer {L} to Layer {L+1}", title_format)
            # Add current dataframe to writer object
            df.to_excel(writer, sheet_name='Matrices',
                startrow=row+1, startcol=0, header=False, index=False)
            # Increase row index to leave one blank row inbetween matrices
            row += len(self.mlp.coefs_[L].T) + 2
            
        # Loop through bias matrices
        for B in range(len(self.mlp.intercepts_)):
            # Save the transpose of the current bias array in a dataframe
            df = DataFrame(self.mlp.intercepts_[B].T)
            # Add matrix titles depending on the layer
            if B == len(self.mlp.intercepts_)-1:
                worksheet.write(row, 0, "Bias for Output", title_format)
            else:
                worksheet.write(row, 0, f"Bias for Layer {B+1}", title_format)
            # Add current dataframe to writer object
            df.to_excel(writer, sheet_name='Matrices',
                startrow=row+1, startcol=0, header=False, index=False)
            # Increase row index to leave one blank row inbetween matrices
            row += len(self.mlp.intercepts_[B].T) + 2
            
        # Save writer object to file
        writer.save()
        print(f"Saved matrices to {address[0]}.")


# Class containing the "About" dialog
class AboutDialog(QDialog):

    def __init__(self, style, parent=None):

        super().__init__(parent=parent)

        self.setWindowTitle("About")
        self.setStyleSheet(style)
        # Load icon from main GUI
        self.setWindowIcon(GUI().set_icon())
        # Initialize main layout
        layout = QVBoxLayout()

        # Create image and text widget
        labelpic = QLabel()
        labeltext = QLabel()

        # Align icon in the center
        labelpic.setAlignment(Qt.AlignCenter)
        # Add icon
        labelpic.setPixmap(QPixmap("MLP_Tool_Icon_V2_64.png")) 
        
        # Align text in the center
        labeltext.setAlignment(Qt.AlignCenter)
        # Allow to open the linked URLs
        labeltext.setOpenExternalLinks(True)
        # Add text
        labeltext.setText(
            "<b>MLP Tool</b><br>"
            "Version 1.2<br><br>"
            "This software is used to train simple MLP regression models<br>"
            "and export them as PMML file.<br><br>"
            "Copyright:<br>"
            "2021 Hochschule Zittau/Görlitz<br>"
            "Institut für Prozesstechnik, Prozessautomatisierung und Messtechnik (IPM)<br>"
            "Contact: Daniel Fiß, d.fiss@hszg.de<br><br>"
            "Software created by:<br>"
            "Samuel Pantze<br><br>"
            "License:<br>"
            "<a href='https://www.gnu.org/licenses/gpl-3.0.html'>GPL v3.0</a><br><br>"
            "<a href='https://github.com/RedFalsh/PyQt5_stylesheets'>Dark theme by RedFalsh</a><br><br>"
            "The following libraries have been used:<br>"
            "PyQt5 (GPL v3), PyInstaller (GPL v2), Pandas (BSD 3)<br>"
            "Scikit-Learn (BSD 3), Matplotlib (PSFL), Seaborn (BSD 3)<br>"
            "NumPy (BSD 3), SciPy (BSD 3), Nyoka (Apache License)"
        )
        layout.addWidget(labelpic)
        layout.addWidget(labeltext)
        layout.setSpacing(40)
        # set margins for left, top, right, bottom
        layout.setContentsMargins(20, 40, 20, 40)
        self.setLayout(layout)


# Class containing the main window and functionality
class GUI(QMainWindow):         

    def __init__(self):

        super(GUI, self).__init__()     
        uic.loadUi('mlp_tool.ui', self)
        # Add statusbar object and set it as window statusbar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        # Save the default theme to revert back to it later
        self.DefaultStyle = self.styleSheet()
        # Add window icon from file
        self.setWindowIcon(self.set_icon())
        
        # Connectors for user input events, e.g. pressing buttons
        self.ButtonLoadTrain.clicked.connect(self.opentraindata)
        self.ButtonLoadTest.clicked.connect(self.opentestdata)
        self.actionOpenTrain.triggered.connect(self.opentraindata)
        self.actionOpenTest.triggered.connect(self.opentestdata)
        self.actionClose.triggered.connect(self.close)
        self.actionTheme.triggered.connect(self.set_style)
        self.actionDocEN.triggered.connect(self.open_docsEN)
        self.actionDocDE.triggered.connect(self.open_docsDE)
        self.actionAbout.triggered.connect(self.show_about)
        self.ButtonShowDataset.clicked.connect(self.show_dataset)
        self.ButtonTrain.clicked.connect(self.train)
        self.ButtonLossCurve.clicked.connect(self.loss_curve)
        self.ButtonTest.clicked.connect(self.plot_results)
        self.ButtonTestError.clicked.connect(self.plot_deviation)
        self.ComboSolver.activated.connect(self.check_solver)
        self.ButtonStats.clicked.connect(self.show_stats)
        self.ButtonWeights.clicked.connect(self.show_weights)
        self.ButtonSave.clicked.connect(self.save_model)
        self.ButtonClose.clicked.connect(self.close)


    # Exit function
    def close(self):        
        
        sys.exit() 
    

    # Switch between Light and Dark theme (in Menu -> View)
    def set_style(self):
        if self.actionTheme.text() == "Dark Theme":
            self.setStyleSheet(open("dark.qss").read())
            self.actionTheme.setText("Light Theme")
            # Show dar mode warning in statusbar
            self.statusBar.showMessage("Please do not take screenshots intended for printing in dark mode!", 8000) 
        else:
            self.setStyleSheet(self.DefaultStyle)
            self.actionTheme.setText("Dark Theme")

    
    # Load window icon from file
    def set_icon(self):
        return QIcon("MLP_Tool_Icon_V2_64.png")


    # Open english documentation in a new browser tab
    def open_docsEN(self):
        
        webbrowser.open_new_tab("Documentation_EN.html")


    # Open german documentation in a new browser tab
    def open_docsDE(self):
    
        webbrowser.open_new_tab("Documentation_DE.html")


    # Writes configuration data from line inputs and comboboxes into attributes
    def get_config(self):        

        # Dictionary to convert the combobox strings to activation parameters
        ActivationDict = {             
            "ReLU": "relu",
            "Tanh": "tanh",
            "Linear": "identity",
            "Sigmoid": "logistic"
        }
        # Create a list of input fields. Used to check for missing inputs
        InputList = (
            self.LineLayers,
            self.LineMaxEps,
            self.LineTol,
            self.LineSeed,
            self.LineL2,
            self.LineMomentum,
            self.LineBatch,
            self.LineLearnRate
        )
        # Raise a warning when one of the input fields is empty and abort the method 
        for line in InputList:
            if line.text() == '':
                raise ValueError("Configuration must not be empty.")

        # Raise error if writing to the correct data type fails
        try:
            self.hiddenlayers = eval(self.LineLayers.text())
            self.solver = self.ComboSolver.currentText()
            # Access the dictionary
            self.activation = ActivationDict[self.ComboActivation.currentText()]  
            self.maxEp = int(self.LineMaxEps.text())
            self.tol = float(self.LineTol.text())
            self.seed = int(self.LineSeed.text())
            self.L2 = float(self.LineL2.text())
            self.momentum = float(self.LineMomentum.text())
            # Try to evaluate batch value as integer
            try:
                self.batch = eval(self.LineBatch.text())
            # If batch does not contain an int, write it as a string
            except:
                self.batch = self.LineBatch.text()
            self.learnrate = float(self.LineLearnRate.text())
        except:
            raise TypeError("Input data of wrong file type.")


    # Change configuration fields based on the selected solver
    # Method is called when the solver combobox changes
    def check_solver(self):              
        
        # Get current solver configuration
        self.solver = self.ComboSolver.currentText()

        # Disable loss curve, learning rate and batch size for LBFGS
        if self.solver == "lbfgs":      
            self.ButtonLossCurve.setEnabled(False)
            self.LineLearnRate.setEnabled(False)
            self.LineBatch.setEnabled(False)
        # Only re-enable loss curve if a model has already been trained
        elif hasattr(self, "mlp"):      
            self.ButtonLossCurve.setEnabled(True)
            self.LineLearnRate.setEnabled(True)
            self.LineBatch.setEnabled(True)
        else:
            self.LineLearnRate.setEnabled(True)
            self.LineBatch.setEnabled(True)
        
        # Enable momentum for SGD
        if self.solver == "sgd":       
            self.LineMomentum.setEnabled(True)
        else:
            self.LineMomentum.setEnabled(False)


    # Method for scaling the data to range 0-1
    def scaling(self, data):

        # create scaling estimator object
        self.minmaxscaler = MinMaxScaler()

        # Calculate min and max of the training data and apply scaling:
        data = DataFrame(self.minmaxscaler.fit_transform(data))
        print(
            f"--------------------\n"
            f"Applied scaling with factors: {array2string(1/self.minmaxscaler.scale_, precision=5, separator=', ')}\n"
            f"and minimum values: {array2string(self.minmaxscaler.data_min_, precision=5, separator=', ')}\n"
            f"To scale back: multiply the target column with the factor, then add the minimum value."
        )
        return data


    # Open a dataset file, check its integrity and return address and data
    def inputdatadialog(self):

        options = QFileDialog.Options()
        address = QFileDialog.getOpenFileName(
                            self,"Select data", "","Excel Files (*.xlsx)", options=options)

        # Return with basic error if address is empty (user clicked "cancel")
        # Needed to catch the exception in dependent methods
        if address[0] == '':
            raise BaseException

        # Write data from excel file to local variable
        try:      
            data = read_excel(address[0], header=None, engine='openpyxl')
        except:
            QMessageBox.critical(self, "Error", "Failed to open file.")
            raise ImportError("Failed to open file")
        # Return error message when file is empty
        if data.empty:
            QMessageBox.warning(self, "Error", "Data file cannot be empty!")
            raise ImportError("Data file cannot be empty")
        # Return error message when file has <2 columns
        if len(data.columns) < 2:
            QMessageBox.warning(self, "Error",
                "Data needs at least one column for input and output!")
            raise ImportError("Data needs at least one column for input and output")
        # Only keep rows which are of type float or int
        # Drop the header row if it exists:
        data = data[data[0].apply(lambda x: isinstance(x, (float, int)))]
        # If the header row was dropped, the DataFrame type is an object. Convert to float:
        data = data.apply(to_numeric)
        # Drop any columns with NaN values (can happen if empty cells contain formatting)
        data.dropna(axis=1, inplace=True)

        # Loading a new dataset always reverts to an unscaled state
        self.is_scaled = False

        return address, data
        

    # Open the training data and save its feature count in an attribute
    def opentraindata(self):      

        # Open dialog, write data to attribute
        try:
            address, self.DFtrain_import = self.inputdatadialog()
        except ImportError:
            print("Failed to open training data.")
            return
        # Abort method if user cancelled the file dialog
        except BaseException:
            return

        # Write address to line
        self.LineTrain.setEnabled(True)
        self.LineTrain.setText(address[0])
        # Save feature count (-1 to subtract the output column)
        self.features = self.DFtrain_import.shape[1] - 1
        print(f"Loaded {self.DFtrain_import.shape[0]} training data points from {address[0]}:")
        # Print a data preview in console view
        print(self.DFtrain_import.head(), "\n")
        self.statusBar.showMessage(f"Loaded training data from {address[0]}", 3000)
        # Enable Train button when both datasets are loaded
        if  hasattr(self, 'DFtest_import'):
            self.ButtonTrain.setEnabled(True)
            # Enable "Show Dataset" button if feature count is 2
            if self.features == 2:
                self.ButtonShowDataset.setEnabled(True) 
            else:
                self.ButtonShowDataset.setEnabled(False)


    # Open the test data
    def opentestdata(self):       

        # Open dialog, write data to attribute
        try:
            address, self.DFtest_import = self.inputdatadialog()
        except ImportError:
            print("Failed to open training data.")
            return
        except BaseException:
            return

        # Write address to line   
        self.LineTest.setEnabled(True)
        self.LineTest.setText(address[0])
        print(f"Loaded {self.DFtest_import.shape[0]} test data points from {address[0]}:")
        # Print a data preview in console view
        print(self.DFtest_import.head(), "\n")
        self.statusBar.showMessage(f"Loaded test data from {address[0]}", 3000)
        # Enable Train button when both datasets are loaded
        if hasattr(self, 'DFtrain_import'):
            self.ButtonTrain.setEnabled(True)
            # Enable "Show Dataset" button if feature count is 2
            if self.features == 2:
                self.ButtonShowDataset.setEnabled(True)
            else:
                self.ButtonShowDataset.setEnabled(False)

    
    # Plot input dataset
    def show_dataset(self):

        # Enable matplotlib interactive mode
        plt.ion()
        set_style("whitegrid")
        # Set projection type
        ax = plt.axes(projection='3d')
        ax.scatter(self.DFtrain_import[0], self.DFtrain_import[1],
            self.DFtrain_import[2], color="darkblue")
        ax.scatter(self.DFtest_import[0], self.DFtest_import[1],
            self.DFtest_import[2], color="cornflowerblue")     
        plt.legend(['Train Target', 'Test Target'])
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Y')
        ax.set_proj_type('ortho')
        plt.title("Dataset")
        plt.tight_layout(pad=1)


    # Method for training the model
    # Calls the scaling method if scaling is enabled and checks for data integrity before training.
    def train(self):

        # Get configuration parameters and check if configuration is empty or of incorrect type
        # Abort training method when an error is encountered
        try:
            self.get_config()
        except ValueError:
            QMessageBox.warning(self, "Warning", "Configuration must not be empty.")
            return
        except TypeError:
            QMessageBox.warning(self, "Warning", "Wrong data type in configuration.")
            return

        # Stop method if datasets don't have matching columns
        if self.DFtrain_import.shape[1] != self.DFtest_import.shape[1]:
            QMessageBox.warning(self, "Warning", "Mismatch in Training and Test data column count.")
            return

        # Apply scaling if specified
        if not self.is_scaled and self.CheckScaling.isChecked():
            self.DFtrain = self.scaling(self.DFtrain_import)
            # Apply scaler to the test data (the scaler is only fitted to training data)
            self.DFtest = DataFrame(self.minmaxscaler.transform(self.DFtest_import))
            self.is_scaled = True
        elif not self.CheckScaling.isChecked():
            self.DFtrain = self.DFtrain_import
            self.DFtest = self.DFtest_import
            self.is_scaled = False
            
        # Write train and test dataframes into separate variables.
        # Scikit-Learn takes numpy arrays as input, but excel import returns a pandas DataFrame
        self.Xtrain = self.DFtrain.drop(self.DFtrain.columns[self.features], axis=1).to_numpy()
        self.Ytrain = self.DFtrain[self.features]
        self.Xtest = self.DFtest.drop(self.DFtest.columns[self.features], axis=1).to_numpy()
        self.Ytest = self.DFtest[self.features]

        print("--------------------")

        # Log scaling parameters
        if self.is_scaled:
            print(
                f"Scale factor: {array2string(1/self.minmaxscaler.scale_[self.features], precision=5)} "
                f"and minimum value: {array2string(self.minmaxscaler.data_min_[self.features], precision=5)}\n")
        
        # Log training parameters
        print(
            "Started training with:\n"
            f"Layers: {self.hiddenlayers}\n"
            f"Activation: {self.activation}\n"
            # f"Solver: {self.solver}\n"
            f"Max. Epochs: {self.maxEp}\n"
            f"Stopping Tolerance: {self.tol}"
            # f"Random Seed: {self.Seed}"
        )

        # Create MLP model with the given configuration
        try:
            self.mlp = MLPRegressor(        
                hidden_layer_sizes = self.hiddenlayers,
                solver = self.solver,
                activation = self.activation,
                max_iter = self.maxEp,
                tol = self.tol,
                random_state = self.seed,
                alpha = self.L2,
                momentum = self.momentum,
                batch_size = self.batch,
                # not learn_rate, which only takes ‘constant’, ‘invscaling’, ‘adaptive’
                learning_rate_init = self.learnrate
            )
        except:
            # Return with an error if model initialization failed
            QMessageBox.critical(self, "Error",
                                "Failed to initialize model."
                                "\nCheck configuration parameter.")
            print("Failed to initialize model.")
            return

        # Fit the model to the training data
        try:
            # Track time to show how much time was needed for training
            timestart = perf_counter()
            self.mlp.fit(self.Xtrain, self.Ytrain)    
            timestop = perf_counter()
        except:
            # Return with an error if model fitting failed
            QMessageBox.critical(self, "Error",
                                "Failed to fit the model to the data."
                                "\nCheck dataset consistency.")
            print("Failed to fit model. Check dataset consistency.")
            return

        print(
            "--------------------\n"
            f"Finished training after {self.mlp.n_iter_:.0f} epochs and {timestop-timestart:.1f} seconds.\n"
            f"Training score: {self.mlp.score(self.Xtrain, self.Ytrain):.4f}"
            f", Test score: {self.mlp.score(self.Xtest, self.Ytest):.4f}"
        )
        self.statusBar.showMessage(
            f"Training score: {self.mlp.score(self.Xtrain, self.Ytrain):.4f}, "
            f"Test score: {self.mlp.score(self.Xtest, self.Ytest):.4f}", 8000
        )

        # Enable buttons that require a trained model
        if self.solver != "lbfgs":
            self.ButtonLossCurve.setEnabled(True)
        if self.features == 2:
            self.ButtonTest.setEnabled(True)
        self.ButtonTestError.setEnabled(True)
        self.ButtonSave.setEnabled(True)
        self.ButtonStats.setEnabled(True)
        self.ButtonWeights.setEnabled(True)


    # Print loss curve
    def loss_curve(self):        
        
        # Enable matplotlib interactive mode
        plt.ion()
        set_style("whitegrid")
        DataFrame(self.mlp.loss_curve_).plot()
        plt.title("Loss Curve")
        plt.legend(["Loss"])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.tight_layout(pad=1)


    # Predict values for training and test data
    def predict(self):

        self.predtest = self.mlp.predict(self.Xtest)
        self.predtrain = self.mlp.predict(self.Xtrain)


    # Predict values for training and test data and print as point cloud
    # This method is only activated for datasets with 2D input
    def plot_results(self):     

        # Get prediction values
        self.predict()        

        # Enable matplotlib interactive mode
        plt.ion()
        set_style("whitegrid")
        # Set projection type
        ax = plt.axes(projection='3d')
        ax.scatter(self.DFtrain[0], self.DFtrain[1], self.DFtrain[2], color="darkblue")
        ax.scatter(self.DFtest[0], self.DFtest[1], self.DFtest[2], color="cornflowerblue")
        ax.scatter(self.DFtrain[0], self.DFtrain[1], self.predtrain, color="darkred")
        ax.scatter(self.DFtest[0], self.DFtest[1], self.predtest, color="indianred")      
        plt.legend(['Train Target', 'Test Target', 'Train Prediction', 'Test Prediction'])
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Y')
        ax.set_proj_type('ortho')
        plt.title("Model Test")
        plt.tight_layout(pad=1)
        
    
    # Calculate the deviation of prediction from target
    def calculate_deviation(self):

        # Get prdiction values
        self.predict()

        self.errortrain = self.predtrain - array(self.DFtrain[self.features])
        self.errortest = self.predtest - array(self.DFtest[self.features])
        

    # Plot prediction deviation as lineplot (Error curve)
    def plot_deviation(self):        

        # Get deviation values
        self.calculate_deviation()

        plt.ion()
        set_style("whitegrid")
        lineplot(x=range(len(self.errortrain)), y=self.errortrain)
        lineplot(x=range(len(self.errortest)), y=self.errortest)
        plt.legend(['Train Error', 'Test Error'])
        plt.xlabel("Data Points")
        plt.ylabel("Error")
        plt.title("Error Curve")
        
        plt.tight_layout(pad=1)
        plt.show()


    # Open statistics dialog
    def show_stats(self):

        # Get deviation data
        self.calculate_deviation()
        # Create statistics dialog instance and pass data
        dialog = StatDialog(
            self.Ytrain, self.Ytest,
            self.predtrain, self.predtest,
            self.errortrain, self.errortest,
            self.styleSheet()
        )
        dialog.exec_()


    # Open weights/bias dialog
    def show_weights(self):

        # Create instance of class WeightDialog
        dialog = WeightDialog(self.mlp, self.styleSheet())
        dialog.exec_()


    # Open About dialog
    def show_about(self):

        # Create instance of About dialog and pass the style
        dialog = AboutDialog(self.styleSheet())
        dialog.exec_()


    # Open file dialog and save model as PMML file
    def save_model(self):      

        options = QFileDialog.Options()
        address = QFileDialog.getSaveFileName(self,"Save model","","PMML Files (*.pmml)", options=options)

        # Add scaler to the pipeline if scaling is enabled
        if self.is_scaled:
            pipe = Pipeline([
                ("scaler", self.minmaxscaler),
                ("model", self.mlp)
                ])
        else:
            pipe = Pipeline([
                    ("model", self.mlp)
                    ])
        
        # Create string array for column names
        col_names = []
        for i in range(self.features):
            col_names.append(f'X{i+1}')

        # Write pipeline to PMML file
        if address[0] != '':
            try:
                skl_to_pmml(pipeline=pipe, col_names=col_names, target_name='Y',  pmml_f_name=address[0])
                print(f"Saved MLP model to {address[0]}")
            except:
                print("\nError while saving model.\n"
                "If only one hidden layer exists, try adding another layer with 1 neuron.")


# Calling main loop to start the program
if __name__ == '__main__':      
    main()

