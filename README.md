# Linear Regression from Scratch

## Overview
This projects shows a custom implementation of the linear regression algorithm using gradient descent. You can checkou tmy repo and use the class in one of your projects. IN the repo is also a step by step comparison with the scikit-learn implementation.

## Project Structure
- `source/linear_regression.py`: implementation of `LinearRegressionGD`
- `notebooks/linear_regression.ipynb`: step by step walkthrough
- `LICENSE`: project license


## How to Use
- Run the Jupyter notebook for a full walkthrough
- Inspect and modify the Python class in `source/linear_regression.py`
- Try changing loss functions, learning rate, or number of epochs
- Compare results to scikit-learn and visualize the outcomes

## Features
- Supports multiple loss functions: MSE, MAE, Huber
- Synthetic data generation with known parameters and noise
- Visualizes data, true relationship, and model fits
- Plots training loss curve for convergence analysis
- Compares custom model to scikit-learn's `LinearRegression`
- Quantitative evaluation: MSE, MAE, R² metrics
- Residual analysis for both models
- Easy to extend: try more features, regularization, or multivariate regression

---

## Math Derivations

This project implements **linear regression** trained with **gradient descent** and supports three loss functions:  
 - **MSE**, **MAE**, and **Huber**.  

Below is the mathematical foundation behind each component.


### Linear Regression Model

For a single input feature \( x \in \mathbb{R} \), the model predicts:

\[
\hat{y} = wx + b
\]

Where:

- \( w \): weight (slope)  
- \( b \): bias (intercept)

The prediction error for sample \( i \) is:

\[
e_i = y_i - \hat{y}_i = y_i - (wx_i + b)
\]


### Loss Functions

The model learns parameters \(w\) and \(b\) by minimizing a chosen loss function over the dataset.

#### Mean Squared Error (MSE)

\[
J_{\text{MSE}}(w,b) = \frac{1}{m} \sum_{i=1}^m (y_i - \hat{y}_i)^2
\]


**Gradient Derivation**

Compute the partial derivatives with respect to \( w \) and \( b \).

***Gradient \( w \)***

\[
\frac{\partial J}{\partial w}
= \frac{1}{m} \sum 2 e_i \frac{\partial e_i}{\partial w}
\]

Since:

\[
\frac{\partial e_i}{\partial w} = -x_i
\]

We obtain:

\[
\boxed{
\frac{\partial J}{\partial w}
= -\frac{2}{m} \sum_{i=1}^m x_i (y_i - \hat{y}_i)
}
\]

**Gradient \( b \)**

\[
\frac{\partial e_i}{\partial b} = -1
\]

\[
\boxed{
\frac{\partial J}{\partial b}
= -\frac{2}{m} \sum_{i=1}^m (y_i - \hat{y}_i)
}
\]

---

#### Mean Absolute Error (MAE)

**Definition**

\[
J_{\text{MAE}}(w,b) = \frac{1}{m} \sum_{i=1}^m |y_i - \hat{y}_i|
\]

MAE is **not differentiable** at 0, so we use the ubgradient:

\[
\frac{d}{de}|e| =
\begin{cases}
+1 & e > 0 \\
-1 & e < 0 \\
\text{anything in } [-1,1] & e = 0
\end{cases}
\]

Compactly using the sign function:

\[
\text{sign}(e_i)
\]


**Subgradient Derivatives**

\[
\boxed{
\frac{\partial J}{\partial w}
= -\frac{1}{m}\sum_{i=1}^m x_i \,\text{sign}(y_i - \hat{y}_i)
}
\]

\[
\boxed{
\frac{\partial J}{\partial b}
= -\frac{1}{m}\sum_{i=1}^m \text{sign}(y_i - \hat{y}_i)
}
\]


#### Huber Loss

The Huber loss is a robust loss function that behaves like:

- **MSE** when the error is small  
- **MAE** when the error is large

**Definition**

For error \( e = y - \hat{y} \) and threshold \( \delta \):

\[
L_\delta(e) =
\begin{cases}
\frac{1}{2} e^2 & \text{if } |e| \le \delta \\
\delta |e| - \frac{1}{2}\delta^2 & \text{if } |e| > \delta
\end{cases}
\]

The mean Huber loss:

\[
J_{\text{Huber}} = \frac{1}{m} \sum_{i=1}^m L_\delta(e_i)
\]

---

**Gradient Derivation**

We compute the derivative of the loss w.r.t prediction:
**Case 1 — Quadratic region (|e| ≤ δ)**

\[
L = \frac{1}{2}e^2
\]
\[
\frac{dL}{d\hat{y}} = -e
\]

**Case 2 — Linear region (|e| > δ)**

\[
L = \delta |e| - \frac{1}{2}\delta^2
\]
\[
\frac{dL}{d\hat{y}} = -\delta \,\text{sign}(e)
\]

---

**Final Gradients**

\[
\boxed{
\frac{\partial J}{\partial w} =
\frac{1}{m} \sum_{i=1}^m
x_i
\begin{cases}
-e_i & |e_i| \le \delta \\
-\delta\,\text{sign}(e_i) & |e_i| > \delta
\end{cases}
}
\]

\[
\boxed{
\frac{\partial J}{\partial b} =
\frac{1}{m} \sum_{i=1}^m
\begin{cases}
-e_i & |e_i| \le \delta \\
-\delta\,\text{sign}(e_i) & |e_i| > \delta
\end{cases}
}
\]

---

### Gradient Descent Update Rule

For every epoch, parameters are updated using:

\[
w \leftarrow w - \alpha \frac{\partial J}{\partial w}
\]

\[
b \leftarrow b - \alpha \frac{\partial J}{\partial b}
\]

Where:

- \( \alpha \): learning rate  
- gradients depend on the selected loss function

This iterative optimization continues for a predefined number of epochs.