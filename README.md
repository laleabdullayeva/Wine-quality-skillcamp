# Wine-quality-skillcamp
# Wine Quality Prediction

This repository contains a machine learning pipeline for predicting wine quality based on physicochemical test data. The project involves end-to-end data preprocessing and the implementation of custom Linear and Logistic Regression models using Gradient Descent from scratch.

## Project Pipeline & Contributions
The notebook workflow is divided into modular steps to ensure clean and scalable data processing:

* **Data Cleaning & Missing Values (Lala):** Loading the `WineQuality.csv` dataset, identifying and dropping duplicated records, and binarizing the target variable (`quality` >= 7 is classified as 1, otherwise 0).
* **Outlier Detection & Treatment (Sona):** Utilizing the Interquartile Range (IQR) method to identify and remove outliers from both the training and testing sets, supported by boxplot visualizations.
* **Feature Scaling (Abalfaz):** Applying `MinMaxScaler` and `StandardScaler` to bring all physicochemical features onto a comparable numerical scale for stable gradient descent.
* **Categorical Encoding (Vusala):** Verifying data types to handle categorical columns (the dataset consists entirely of numerical float and integer types, ensuring structural integrity).

## Models Implemented
Rather than relying solely on pre-built libraries, the algorithms in this project are constructed from scratch using NumPy to demonstrate the underlying math of Gradient Descent.

### 1. Linear Regression
* A custom linear model trained over 2000 iterations with a learning rate of 0.01.
* **Mean Squared Error (MSE):** 0.4367
* **R-squared (R2 Score):** 0.2903
* **Exact Match Accuracy:** 52.14%

### 2. Logistic Regression
* A custom classification model using the Sigmoid function, optimized to distinguish between high-quality and standard-quality wines.
* **Accuracy:** 57.26%
* **Precision:** 57.26%
* **Recall:** 100.00%
* **F1-Score:** 72.83%

## Technologies Used
* **Python 3**
* **Pandas:** Data manipulation, cleaning, and analysis.
* **NumPy:** Matrix operations and custom machine learning algorithms.
* **Scikit-Learn:** Train/test splitting, scaling utilities, and evaluation metrics.
* **Matplotlib & Seaborn:** Visualizing regressions, boxplots, and confusion matrices.

## How to Run
1. Clone this repository to your local machine.
2. Ensure the `WineQuality.csv` file is placed in the correct working directory.
3. Open the Jupyter Notebook and execute the cells sequentially to observe the data transformations and model training processes.
