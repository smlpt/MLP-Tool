# Documentation MLP Tool

This tool provides a graphical interface to the scikit-learn Python library and can be used to train simple neural networks of the class **Multilayer Perceptron** (MLP) for regression problems.

## Theoretical background

### Structure

Excerpt from [Wikipedia](https://en.wikipedia.org/wiki/Multilayer_perceptron):

> An MLP consists of at least three [layers](https://en.wikipedia.org/wiki/Layer_(deep_learning) "Layer (deep learning)") of nodes: an input [layer](https://en.wikipedia.org/wiki/Layer_(deep_learning) "Layer (deep learning)"), a hidden [layer](https://en.wikipedia.org/wiki/Layer_(deep_learning) "Layer (deep learning)") and an output [layer](https://en.wikipedia.org/wiki/Layer_(deep_learning) "Layer (deep learning)"). Except for the input nodes, each node is a neuron that uses a nonlinear [activation function](https://en.wikipedia.org/wiki/Activation_function "Activation function"). [...] It can distinguish data that is not [linearly separable](https://en.wikipedia.org/wiki/Linear_separability "Linear separability"). [...]
> 
> Since MLPs are fully connected, each node in one layer connects with a certain weight $\omega_{ij}$ to every node in the following layer.

### Activation functions

The output $f(x)$ of each neuron is determined by mapping its weighted inputs $x$ to the activation function. To model nonlinear behaviour, activation functions like Tangens Hyperbolicus or the Sigmoid function are necessary.

Typical activation functions are:

#### Linear

The linear function maps the inputs directly to the neurons output.

$$
f(x)=x
$$

#### Tangens Hyperbolicus

This logistic activation function ranges from -1 to 1.

$$
f(x)=\text{tanh}(x)
$$

#### Sigmoid

This logistic activation function ranges from 0 to 1.

$$
f(x)=\frac{1}{1+e^{-1}}
$$

#### ReLu

The rectified linear unit function equals the linear function for values $x>0$ and is equal to 0 for values $x<0$.

$$
f(x)=\text{max$(0,x)$}
$$

This function is often applied in deep neural networks to counteract the vanishing gradient problem that prevents models from learning effectively.

### Solver

MLPs use an algorithm called [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) to compute the gradient of the loss function with respect to each weight by the chain rule. These gradients are then available for the optimizer (or solver) algorithms to update the weights after each training epoch. The available optimizers differ in terms of the approach to finding the global minimum in the parameter hyperspace in an efficient way.

## Feature Overview

### File Input

The training and test data are loaded in as Excel files (.xlsx). Training data are used to train the model with the given configuration. The model is then applied to the test data to calculate its accuracy.

The first columns contain the input data, the last column is always reserved for the output data. Rows containing strings are automatically dropped during the input process. Files containing less than two columns are rejected, as well as files with mismatching column counts.

A data preview is printed in the console after loading a dataset.

### Scaler

If the dataset is not within the range of 0-1 yet, checking this option will scale all data to the minimum and maximum values of the training data and output the factor and the minimum value to the console. These values can then be used in other software to descale the model output back to the original range.  

### Configuration

The following parameters can be configured:
(see [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) for more details)

- **Hidden layers and neuron count per layer:**
  The number of neurons per layer is separated by a comma. E.g. "4,2" will create a network with two hidden layers, containing 4 and 2 neurons.

- **Activation function:**
  The mathematical function used to determine the neuron output based on its input. Available functions are:
  
  - ReLu: the rectified linear unit function, $f(x)=max(0, x)$
  - TanH: Tangens Hyperbolicus, $f(x)=tanh(x)$
  - Linear: $f(x)=x$
  - Sigmoid: logistic sigmoid function, $\frac{1}{1+e^{-x}}$

- **Solver:**
  The learning algorithm used to improve the connection weights after each epoch.
  Available solvers are:
  
  - [Adam](https://arxiv.org/abs/1412.6980): Adaptive Moment Estimation. Stores exponentially decaying average of past squared gradients and the gradients to adaptively compute learning rates. [🡥](https://ruder.io/optimizing-gradient-descent/)
  
  - SGD: Classic gradient descent algorithm with configurable batch size.
  
  - L-BFGS: Limited-memory implementation of the [Broyden–Fletcher–Goldfarb–Shanno algorithm](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm), a quasi-Newtonian method that uses an estimate of the inverse [Hessian matrix](https://en.wikipedia.org/wiki/Hessian_matrix) (second-order derivatives) to update the weights. Converges very fast for smaller datasets. This solver does not provide a learning curve.

- **Maximum number of epochs:**
  Training will stop after reaching this number of epochs or after reaching the specified tolerance.

- **Tolerance:**
  When training is not improving by this delta value for at least 10 epochs, the training is considered to have finished.

- **Random state:**
  The seed value for random number generation, used for weights and bias initialization as well as batch sampling. Setting an integer here results in reproducible results.

- **L2 penalty:**
  The regularization parameter for ridge regression that is used to prevent large weights and over-fitting by adding squared magnitude of coefficients as penalty to the loss function. Large L2 parameter values will lead to under-fitting. [🡥](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c?gi=273b9364d0a7)

- **Momentum:**
  Adds a momentum term to the gradient descent by adding adding an exponentially weighed average, which causes the weight updates to accelerate towards a global parameter minimum and prevents getting stuck in local minima. 0.9 is a typical value.

- **Batch size:**
  Size of minibatches for the stochastic solvers (SGD and Adam). "auto" chooses the minimum from 200 or the total number of samples. A size of 1 equals a stochastic gradient descent with weight updates after each presented data sample. A batch size that includes all samples equals the batch gradient descent algorithm, where weights are updated after presenting all samples. Using minibatches (between 1 and all samples) is usually the most robust approach.

- **Learning rate:**
  Constant learning rate for stochastic solvers that controls the step size in updating the weights.

### Plotting and statistics

After the model training has finished, a number of evaluation options become available:

#### Test Model

Plots a point cloud of the real and the predicted values in 3D space. This option is only available for two-dimensional input data.

#### Loss Curve

Plots a 2D graph that shows the history of losses over the training period.

#### Error Curve

Plots a 2D graph of the difference between target data and predictions. Training and test data are plotted separately.

#### Statistics

Calculates a number of relevant statistical properties, separated by training and test data. This includes the R^2^ score, minimum and maximum error values, absolute mean square error, RMS error, etc.

> The **R^2^ score**, also called the [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination), "[...] represents the proportion of variance (of y) that has been explained by the independent variables in the model. [...] It's therefore a measure of how well unseen samples are likely to be predicted by the model, through the portion of explained variance."
> The R^2^ score implementation in scikit-learn is defined as:

> $$
> R^2(y,\hat y_i) = 1 - \frac{\sum_{i=1}^{n}(y_i-\hat y_i)^2}{\sum_{i=1}^{n}(y_i-\bar y)^2}
> $$

> where $\bar y=\frac 1 n \sum_{i=1}^{n}y_i$ and $\sum_{i=1}^{n}(y_i-\hat y_i)^2=\sum_{i=1}^{n}\epsilon_i^2$. 
> 
> $\hat y_i$ is the predicted value of the $i$-th sample and $y_i$ the corresponding true value for a total of $n$ samples. [🡥](https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score)

The R^2^ score ranges from 0 to 1, but can also be negative when the score is calculated from data that haven't been used to train the network model.

The same R^2^ score calculation applies to the trendline feature in Microsoft Excel. 

#### Weights & Bias

Print a table of all weights in the network model and a table of bias values per neuron.

### Saving

Opens a saving dialog to export the trained model as PMML file for later use in other software.

## Limitations

Neural networks exist in a large variety of types and application scenarios. This tool only provides functionality for one specific class of model, the Multilayer Perceptron. These are also often used for classification problems, but this tool only works for regression problems.

The input data can have an arbitrary amount of dimensions, the output is always one-dimensional however, because it is the most used application scenario. Because this tool relies on the MLPRegressor method from the scikit-learn library, implementing more than one output is currently not possible.

Due to the relatively small datasets and the internal serial processing, GPU acceleration is not  available for this tool (or scikit-learn in general). Fast CPU single core clock speeds help decreasing the overall training time.

Model optimization functionality is highly limited due to the used MLPRegressor method. There is no pruning or dropout available. However tweaking the model parameters should still wield satisfactory results in the majority of cases.

This tool offers no data pre-processing features apart from scaling the values from 0 to 1. The user is responsible for cleaning the dataset and checking for outliers or implausible data points. 

## Troubleshooting

Most usage errors should be caught by the tool itself and explain the issue via a warning message. These include:

- Mismatch in column and row count in training and test datasets or not enough columns

- Empty or corrupt data files

- Empty parameter configuration fields

- Parameter configurations of the wrong data type

If **model initialization** fails, this is most likely caused by the parameter configuration. Please check the parameter configuration for correct data types. Use a period as the decimal separator, not a comma.

If **model fitting** fails, this is most likely caused by an issue inside the dataset.
