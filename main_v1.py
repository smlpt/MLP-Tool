import sys
# import os

# Measure elapsed training time
from time import perf_counter

# VS Code error "no  name [...] in module PyQt5.QtWidgets" is just a linting error,
# it does not prevent running the script.
# Reason: Pylint does not load any C extensions by default.
from PyQt5.QtWidgets import (
    QApplication,       # base application      
    QStatusBar,         # status bar in main window
    QMainWindow,        # main window
    QWidget,            # general widget class
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
    QStyleFactory       # Access different system themes
)
from PyQt5.QtGui import QFont, QIcon   # change font parameters, add window icon
from PyQt5 import uic   # import UI file
from PyQt5.QtCore import (
    Qt,                 # needed for alignment in Boxlayout
    QObject,            # generic PyQt object
    QThread,            # needed for worker thread for training (to maintain responsive UI)
    pyqtSignal          # emit signals from worker thread to main thread
    )  

# method "read_excel" requires the openpyxl library to be installed
from pandas import DataFrame, read_excel, to_numeric
# Print 3D plots
from mpl_toolkits.mplot3d import Axes3D
# Print 2D plots
import matplotlib.pyplot as plt
# Provide fancier plot designs
from seaborn import set_style, lineplot
from numpy import array
from math import sqrt

# Methods for training the network
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# FIX FOR PMML44 NOT FOUND in pyinstaller: Add nyoka folder as --paths= argument
from nyoka import skl_to_pmml, PMML44

# Open Documentation in browser
import webbrowser

# import tensorflow as tf
# from tensorflow import keras
#import tensorflow_model_optimization as tfmot

# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Main Function
def main():         

    app = QApplication(sys.argv)    # Create an instance of QtWidgets.QApplication
    window = GUI()    # Create an instance of the interface class
    window.show()
    sys.exit(app.exec_())       # Start the application


# Class containing the statistics dialog
class StatDialog(QDialog):

    def __init__(self, Ytrain, Ytest, predtrain, predtest, errortrain, errortest, style, parent=None):

        super().__init__(parent=parent)

        self.setWindowTitle("Model Statistics")
        self.setStyleSheet(style)
        # Load icon from main GUI
        self.setWindowIcon(GUI().SetIcon())

        # The statistics dialog contains two column pairs, for training and test statistics.
        layout = QHBoxLayout()
        
        # Add widgets with labels that print the required data columns
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
        self.setWindowIcon(GUI().SetIcon())
        # Initialize main layout
        layout = QHBoxLayout()

        # Array structure:
        # Dim. 1: Layer
        # Dim. 2: Neuron
        # Dim. 3: Weight

        # Create 2D array with all neuron connections and corresponding weights
        weight_data = []
        for L in range(len(mlp.coefs_)):
            for N in range(len(mlp.coefs_[L])):
                for W in range(len(mlp.coefs_[L][N])):
                    if L == 0:
                        weight_data.append([
                            f"Input {N+1}",
                            f"Neuron {L+1}-{W+1}",
                            f"{mlp.coefs_[L][N][W]:.6f}"
                        ])
                    elif L==len(mlp.coefs_)-1:
                        weight_data.append([
                            f"Neuron {L}-{N+1}",
                            "Output",
                            f"{mlp.coefs_[L][N][W]:.6f}"
                        ])
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
        weight_table.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Maximum)
        weight_table.setRowCount(len(weight_data))
        weight_table.setColumnCount(3)
        weight_table.setHorizontalHeaderLabels(('From', 'To', 'Weight'))
        font = QFont()
        font.setBold(True)
        weight_table.horizontalHeader().setFont(font)
        weight_table.setAlternatingRowColors(True)
        
        # Populate weight table with data
        for col in range(3):
            for row in range(len(weight_data)):
                item = QTableWidgetItem(weight_data[row][col])
                # Enable selecting cell but disable editing
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                weight_table.setItem(row, col, item)
        
        weight_table.resizeRowsToContents()
        weight_table.resizeColumnsToContents()
        
        # Create 2D array with all neurons and corresponding bias values
        bias_data = []
        for L in range(len(mlp.intercepts_)):
            for N in range(len(mlp.intercepts_[L])):
                if L == len(mlp.intercepts_)-1:
                    bias_data.append([
                        "Output",
                        f"{mlp.intercepts_[L][N]:.6f}"
                    ])
                else: 
                    bias_data.append([
                        f"Neuron {L+1}-{N+1}",
                        f"{mlp.intercepts_[L][N]:.6f}"
                    ])

        # Create bias table and adjust size policies
        bias_table = QTableWidget()
        bias_table.horizontalHeader().setMinimumSectionSize(100)
        bias_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        bias_table.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Maximum)
        bias_table.setRowCount(len(bias_data))
        bias_table.setColumnCount(2)
        bias_table.setHorizontalHeaderLabels(('Neuron', 'Bias'))
        bias_table.horizontalHeader().setFont(font)
        bias_table.setAlternatingRowColors(True)

        # Populate bias table with data
        for col in range(2):
            for row in range(len(bias_data)):
                item = QTableWidgetItem(bias_data[row][col])
                # Enable selecting cell but disable editing
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                bias_table.setItem(row, col, item)

        bias_table.resizeRowsToContents()
        bias_table.resizeColumnsToContents()

        # Add weight table and bias table to layout
        layout.addWidget(weight_table, alignment=Qt.AlignTop)
        layout.addWidget(bias_table, alignment=Qt.AlignTop)

        self.setLayout(layout)
        self.adjustSize()


# Class containing the main functionality
class GUI(QMainWindow):         

    def __init__(self):

        # Call the inherited classes __init__ method
        super(GUI, self).__init__()     
        uic.loadUi('mlp_tool_v1.ui', self)
        # Add statusbar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        # Save the default theme to revert back to it later
        self.DefaultStyle = self.styleSheet()
        # Add window icon from file
        self.setWindowIcon(self.SetIcon())
        
        # Connectors for user input, e.g. pressing buttons
        self.ButtonLoadTrain.clicked.connect(self.OpenTrainData)
        self.ButtonLoadTest.clicked.connect(self.OpenTestData)
        self.actionClose.triggered.connect(self.close)
        self.actionTheme.triggered.connect(self.SetStyle)
        self.actionDocumentation.triggered.connect(self.OpenDocs)
        self.ButtonTrain.clicked.connect(self.Train)
        self.ButtonLossCurve.clicked.connect(self.LossCurve)
        self.ButtonTest.clicked.connect(self.PlotResults)
        self.ButtonTestError.clicked.connect(self.PlotDeviation)
        self.ComboSolver.activated.connect(self.checkSolver)
        self.ButtonStats.clicked.connect(self.ShowStats)
        self.ButtonWeights.clicked.connect(self.ShowWeights)
        self.ButtonSave.clicked.connect(self.SaveModel)
        self.ButtonClose.clicked.connect(self.close)

        # Is set to True after scaling method is applied, prevents scaling twice
        self.is_scaled = False
        

    # Exit function
    def close(self):        
        
        sys.exit() 
    

    # Switch between Light and Dark theme (Menu -> View)
    def SetStyle(self):
        if self.actionTheme.text() == "Dark Theme":
            self.setStyleSheet(open("dark.css").read())
            self.actionTheme.setText("Light Theme")
        else:
            self.setStyleSheet(self.DefaultStyle)
            self.actionTheme.setText("Dark Theme")

    
    # Load window icon from file
    def SetIcon(self):
        return QIcon("MLP_Tool_Icon_64.png")


    def OpenDocs(self):
        # Open Documentation file in new tab
        webbrowser.open_new_tab("Documentation.html")


    # Writes data from line inputs and comboboxes into attributes
    def getConfig(self):        

        # Dictionary to convert the combobox strings to activation parameters
        ActDict = {             
            "ReLu": "relu",
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
        # Show a warning when one of the input fields is empty and abort the method 
        for Line in InputList:
            if Line.text() == '':
                raise ValueError("Configuration must not be empty.")
        # Raise error if writing to the correct data type fails
        try:
            self.HiddenLayers = eval(self.LineLayers.text())
            self.Solver = self.ComboSolver.currentText()
            self.Activation = ActDict[self.ComboActivation.currentText()]   # access the dictionary
            self.MaxEp = int(self.LineMaxEps.text())
            self.Tol = float(self.LineTol.text())
            self.Seed = int(self.LineSeed.text())
            self.L2 = float(self.LineL2.text())
            self.Momentum = float(self.LineMomentum.text())
            self.Batch = self.LineBatch.text()
            self.LearnRate = float(self.LineLearnRate.text())
        except:
            raise TypeError("Input data of wrong file type.")


    # Change UI based on selected solver
    def checkSolver(self):              
        
        # Get config from UI
        self.Solver = self.ComboSolver.currentText()

        # Disable loss curve, learning rate and batch size for LBFGS
        if self.Solver == "lbfgs":      
            self.ButtonLossCurve.setEnabled(False)
            self.LineLearnRate.setEnabled(False)
            self.LineBatch.setEnabled(False)
        # Only re-enable loss curve if a model has already been trained
        elif hasattr(self, "mlp"):      
            self.ButtonLossCurve.setEnabled(True)
        else:
            self.LineLearnRate.setEnabled(True)
            self.LineBatch.setEnabled(True)
        
        # Enable momentum for SGD
        if self.Solver == "sgd":       
            self.LineMomentum.setEnabled(True)
        else:
            self.LineMomentum.setEnabled(False)


    # Scales the data to range 0-1
    def scaling(self, data):

        # create scaling estimator object
        self.minmaxscaler = MinMaxScaler()

        # Calculate min and max of the training data and apply scaling:
        data = DataFrame(self.minmaxscaler.fit_transform(data))
        print(
            f"--------------------\n"
            f"Applied scaling with factors: {self.minmaxscaler.scale_} "
            f"and minimum values: {self.minmaxscaler.data_min_}\n"
            f"Descale the target column by dividing through the factor and adding the minimum value."
        )
        return data


    # Open data file, check integrity and return address and data
    def InputDataDialog(self):

        options = QFileDialog.Options()
        Address = QFileDialog.getOpenFileName(
                            self,"Select data", "","Excel Files (*.xlsx)", options=options)
        # Write data from excel file to local variable
        if Address[0] != '':
            try:      
                data = read_excel(Address[0], header=None, engine='openpyxl')
            except:
                QMessageBox.critical(self, "Error", "Failed to open file.")
                return
            # Return error message when file is empty
            if data.empty == True:
                QMessageBox.warning(self, "Error", "Data file cannot be empty!")
                raise ImportError("Data file cannot be empty")
            # Return error message when file has <2 columns
            if len(data.columns) < 2:
                QMessageBox.warning(self, "Error", "Data needs at least one column for input and output!")
                raise ImportError("Data needs at least one column for input and output!")
            # Only keep rows which are of type float or int
            # Used to drop the header row if it exists
            data = data[data[0].apply(lambda x: isinstance(x, (float, int)))]
            # If header row was dropped, DataFrame is object instead of float. Convert to float:
            data = data.apply(to_numeric)
            return Address, data
        # Return with basic error if address is empty (user clicked "cancel")
        else:
            raise BaseException


    # Open training data and save feature count in attribute
    def OpenTrainData(self):      

        # Open dialog, write data to attribute
        try:
            Address, self.DFtrain_import = self.InputDataDialog()
        except ImportError:
            print("Failed to open training data.")
            return
        # Abort method if user cancelled the file dialog
        except BaseException:
            return

        self.LineTrain.setEnabled(True)
        self.LineTrain.setText(Address[0])
        # Save feature count
        self.Features = self.DFtrain_import.shape[1] - 1
        print(f"Loaded {self.DFtrain_import.shape[0]} training data points from {Address[0]}:")
        print(self.DFtrain_import.head())
        self.statusBar.showMessage(f"Loaded training data from {Address[0]}", 3000)
        # Enable Train button when both datasets are loaded
        if  hasattr(self, 'DFtest_import'):
            self.ButtonTrain.setEnabled(True)    


    # Open test data
    def OpenTestData(self):       

        # Open dialog, write data to attribute
        try:
            Address, self.DFtest_import = self.InputDataDialog()
        except ImportError:
            print("Failed to open training data.")
            return
        except BaseException:
            return
            
        self.LineTest.setEnabled(True)
        self.LineTest.setText(Address[0])
        print(f"Loaded {self.DFtest_import.shape[0]} test data points from {Address[0]}:")
        print(self.DFtest_import.head())
        self.statusBar.showMessage(f"Loaded test data from {Address[0]}", 3000)
        # Enable Train button when both datasets are loaded
        if hasattr(self, 'DFtrain_import'):
            self.ButtonTrain.setEnabled(True) 


    # Method for training the model.
    # Calls the scaling method if scaling is enabled and checks for data integrity before training.
    def Train(self):

        # Load hyperparameters into attributes
        # and check if configuration is empty or of incorrect type
        try:
            self.getConfig()
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

        # Enable buttons that require a trained model
        if self.Solver != "lbfgs":
            self.ButtonLossCurve.setEnabled(True)
        if self.Features == 2:
            self.ButtonTest.setEnabled(True)
        self.ButtonTestError.setEnabled(True)
        self.ButtonSave.setEnabled(True)
        self.ButtonStats.setEnabled(True)
        self.ButtonWeights.setEnabled(True)

        # Apply scaling if specified
        if self.is_scaled == False and self.CheckScaling.isChecked():
            self.DFtrain = self.scaling(self.DFtrain_import)
            self.DFtest = DataFrame(self.minmaxscaler.transform(self.DFtest_import))
            self.is_scaled = True
        elif self.CheckScaling.isChecked() == False:
            self.DFtrain = self.DFtrain_import
            self.DFtest = self.DFtest_import
            self.is_scaled = False
            
        # Write train and test dataframes into separate variables.
        # Scikit-Learn takes numpy arrays as input, but excel import is of type pandas DataFrame.
        self.Xtrain = self.DFtrain.drop(self.DFtrain.columns[self.Features], axis=1).to_numpy()
        self.Ytrain = self.DFtrain[self.Features]
        self.Xtest = self.DFtest.drop(self.DFtest.columns[self.Features], axis=1).to_numpy()
        self.Ytest = self.DFtest[self.Features]

        print(
            "--------------------\n"
            "Started training with:\n"
            f"Layers: {self.HiddenLayers}\n"
            # f"Activation: {self.Activation}\n"
            # f"Solver: {self.Solver}\n"
            f"Max. Epochs: {self.MaxEp}\n"
            f"Stopping Tolerance: {self.Tol}\n"
            # f"Random Seed: {self.Seed}"
        )

        # Create MLP model with specified hyperparameters
        try:
            self.mlp = MLPRegressor(        
                hidden_layer_sizes = self.HiddenLayers,
                solver = self.Solver,
                activation = self.Activation,
                max_iter = self.MaxEp,
                tol = self.Tol,
                random_state = self.Seed,
                alpha = self.L2,
                momentum = self.Momentum,
                batch_size=self.Batch,
                # not learn_rate, which only takes ‘constant’, ‘invscaling’, ‘adaptive’
                learning_rate_init=self.LearnRate
            )
        except:
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
            f"Training score: {self.mlp.score(self.Xtrain, self.Ytrain):.4f}"
            f", Test score: {self.mlp.score(self.Xtest, self.Ytest):.4f}", 8000
        )


    # Print loss curve
    def LossCurve(self):        
        
        plt.ion()
        set_style("whitegrid")
        DataFrame(self.mlp.loss_curve_).plot()
        plt.title("Loss Curve")
        plt.legend(["Loss"])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.tight_layout(pad=1)


    # Predict values for training and test data
    def Predict(self):

        self.predtest = self.mlp.predict(self.Xtest)
        self.predtrain = self.mlp.predict(self.Xtrain)


    # Predict values for training and test data and print as point cloud
    # This method is only activated for datasets with 2D input
    def PlotResults(self):     

        # Get prediction values
        self.Predict()        

        plt.ion()
        set_style("whitegrid")
        ax = plt.axes(projection='3d')
        ax.scatter(self.DFtrain[0], self.DFtrain[1], self.DFtrain[2], color="darkblue")
        ax.scatter(self.DFtest[0], self.DFtest[1], self.DFtest[2], color="royalblue")
        ax.scatter(self.DFtrain[0], self.DFtrain[1], self.predtrain, color="darkred")
        ax.scatter(self.DFtest[0], self.DFtest[1], self.predtest, color="indianred")      
        plt.legend(['Train Target', 'Test Target', 'Train Prediction', 'Test Prediction'])
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Y')
        ax.set_proj_type('ortho')
        plt.title("Error Curve")
        plt.tight_layout(pad=1)
        
    
    # Calculate the deviation of prediction from target
    def CalculateDeviation(self):

        # Get prdiction values
        self.Predict()

        self.errortrain = self.predtrain - array(self.DFtrain[self.Features])
        self.errortest = self.predtest - array(self.DFtest[self.Features])
        

    # Calculate the deviation of prediction from target and plot the results as lineplot
    def PlotDeviation(self):        

        self.CalculateDeviation()

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
    def ShowStats(self):

        self.CalculateDeviation()

        Dialog = StatDialog(
            self.Ytrain, self.Ytest,
            self.predtrain, self.predtest,
            self.errortrain, self.errortest,
            self.styleSheet()
        )
        Dialog.exec_()


    # Open weights/bias dialog
    def ShowWeights(self):

        # Create instance of class WeightDialog
        Dialog = WeightDialog(self.mlp, self.styleSheet())
        Dialog.exec_()


    # Opens file dialog and saves model as PMML file
    def SaveModel(self):            
        options = QFileDialog.Options()
        fileName = QFileDialog.getSaveFileName(self,"Save model","","PMML Files (*.pmml)", options=options)

        # Add scaler to the pipeline if scaling is enabled
        if self.is_scaled == True:
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
        for i in range(self.Features):
            col_names.append(f'X{i+1}')

        # Write pipeline to PMML file
        if fileName[0] != '':
            skl_to_pmml(pipeline=pipe, col_names=col_names, target_name='Y',  pmml_f_name=fileName[0])
            print(f"Saved MLP model to {fileName[0]}")



# Calling main loop to start the program
if __name__ == '__main__':      
    main()

