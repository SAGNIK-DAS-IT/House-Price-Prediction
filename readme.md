# 🏡 Boston House Price Prediction: A Linear Regression Implementation

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Data_Processing-green.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)

## 📌 Project Objective
This repository contains a from-scratch implementation of a **Multiple Linear Regression** model. Rather than utilizing pre-built machine learning libraries like `scikit-learn`, this project builds the underlying algorithms directly using Python and NumPy to predict the median value of owner-occupied homes in Boston. 

By implementing the mathematical foundations manually, this project demonstrates a deep understanding of cost functions, gradient descent optimization, and linear algebra solutions.

---

## 🧮 Mathematical Foundation

This project solves the linear regression problem using two distinct methodologies to verify accuracy and compare performance:

### 1. Batch Gradient Descent
An iterative optimization algorithm used to find the model parameters ($\theta$) that minimize the cost function.
* **Hypothesis Function:** $h_\theta(x) = \theta^T x$
* **Cost Function (Mean Squared Error):** $$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$
* **Update Rule:** $$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$$
* **Hyperparameters:** Learning rate ($\alpha$) = 0.001, Convergence threshold = 0.001.

### 2. The Normal Equation
An analytical approach that solves for the optimal parameters in a single computation, bypassing the need for feature scaling or iterative updates. 
* **Equation:**
  $$\theta = (X^T X)^{-1} X^T y$$
*(Note: The implementation utilizes the Moore-Penrose pseudo-inverse to ensure stability even if the matrix is non-invertible).*

---

## 📊 Dataset & Preprocessing

The model is trained on the classic **Boston Housing Dataset** (`boston.csv`).
* **Total Samples:** 506 (450 Training | 56 Testing)
* **Features:** 12 variables (Crime rate, zoning, room count, tax rates, etc. The `CHAS` dummy variable was excluded).

**Feature Scaling:**
To ensure smooth and rapid convergence during Gradient Descent, standard scaling (Z-score normalization) was applied to the feature matrix:
$$x_{scaled} = \frac{x - \mu}{\sigma}$$

---

## 🚀 Results & Performance

Both the iterative and analytical models were evaluated against the unseen testing dataset using the half Mean Squared Error metric.

| Model Approach | Test Cost (Half MSE) |
| :--- | :--- |
| **Batch Gradient Descent** | ~4.97 |
| **Normal Equation** | ~6.17 |

*The results indicate that the iterative gradient descent approach (with the chosen hyperparameters and standard scaling) generalized slightly better to the test set than the direct analytical solution.*

---

## 💻 Local Setup & Usage

To run this notebook locally and experiment with the learning rate or feature set:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/SAGNIK-DAS-IT/House-Price-Prediction.git](https://github.com/SAGNIK-DAS-IT/House-Price-Prediction.git)
   cd House-Price-Prediction