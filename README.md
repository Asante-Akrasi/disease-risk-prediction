Overview

This project applies machine learning techniques to predict the likelihood of diabetes using clinical data. The goal is to explore how data-driven models can support early detection and decision-making in healthcare systems.


Dataset

* Pima Indians Diabetes Dataset
* 768 patient records with features including:
    * Glucose level
    * BMI
    * Age
    * Blood pressure
    * Insulin levels


Models Used

* Logistic Regression
    * Used as a baseline model due to its interpretability
* Random Forest
    * Used to capture non-linear relationships and improve prediction performance

Methodology

* Data cleaning (replacing zero values in selected medical features with median values)
* Exploratory Data Analysis (EDA):
    * Correlation heatmap
    * Feature distribution plots
    * Outcome distribution analysis
* Data preprocessing:
    * Feature scaling using StandardScaler
    * Stratified train-test split (80/20)
* Model training and evaluation
* Comparison of model performance

Model Evaluation

The models were evaluated using the following metrics:

* Accuracy
* Precision
* Recall

Performance results are printed directly when the script is executed.

Key Insights

* Glucose shows the strongest correlation with diabetes outcome (~0.49)
* BMI and age demonstrate moderate influence on predictions
* Most other features have weak individual correlations
* The dataset is slightly imbalanced, which may affect model performance

Visualization

The project includes:
* Correlation heatmap
* Feature distribution plots (e.g., glucose distribution)
* Outcome distribution (class balance)
* Confusion matrix (for model evaluation)
* Feature importance plot (Random Forest)

Tools & Technologies

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn

Limitations

* Dataset size is relatively small (768 samples)
* Limited demographic diversity
* Class imbalance not explicitly handled
* Evaluation does not include advanced metrics such as ROC-AUC

Future Improvements

* Add advanced evaluation metrics (ROC-AUC, F1-score)
* Perform hyperparameter tuning
* Apply cross-validation
* Address class imbalance using resampling techniques
* Explore more advanced models such as neural networks

Motivation

This project reflects my interest in applying artificial intelligence to healthcare systems, particularly in developing data-driven tools that can support early disease detection and improve decision-making in resource-constrained environments.
  
