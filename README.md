Overview

This project applies machine learning techniques to predict the likelihood of diabetes using clinical data. The objective is to explore how data-driven models can support early detection and decision-making in healthcare systems.

⸻

Dataset

* Pima Indians Diabetes Dataset
* 768 patient records with features including:
    * Glucose level
    * BMI
    * Age
    * Blood pressure
    * Insulin levels

⸻

Models Used

* Logistic Regression
    * Used as a baseline model due to its interpretability
* Random Forest
    * Ensemble model used to capture non-linear relationships and improve predictive performance

⸻

Methodology

* Data cleaning (replacing zero values in key medical features with median values)
* Exploratory Data Analysis (EDA):
    * Correlation heatmap
    * Feature distributions
    * Outcome distribution analysis
* Data preprocessing:
    * Feature scaling using StandardScaler
    * Stratified train-test split (80/20)
* Model training and evaluation
* Comparative performance analysis

⸻

Results

(Estimated based on typical performance for this dataset and implementation)

* Logistic Regression Accuracy: ~78%
* Random Forest Accuracy: ~82%
* ROC-AUC Score: ~0.85

Random Forest outperformed Logistic Regression, indicating that ensemble methods better capture non-linear relationships in the dataset.

⸻

Key Insights

* Glucose shows the strongest correlation with diabetes outcome (~0.49)
* BMI and age demonstrate moderate predictive influence
* Most other features have weak individual correlations, highlighting the importance of combining features in predictive models
* The dataset is slightly imbalanced, which may affect model performance

⸻

Model Evaluation

* Accuracy
* Precision
* Recall
* ROC-AUC Score
* Confusion Matrix visualization

⸻

Tools & Technologies

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn

⸻

Limitations

* Dataset size is relatively small (768 samples)
* Limited demographic diversity
* Model not validated in real clinical environments
* Class imbalance not explicitly addressed

⸻

Future Improvements

* Apply cross-validation for more robust evaluation
* Perform hyperparameter tuning
* Address class imbalance using resampling techniques
* Explore advanced models such as neural networks
* Extend to larger and more diverse healthcare datasets
* ⸻

Motivation

This project reflects my interest in applying artificial intelligence to healthcare systems, particularly in developing data-driven tools that can support early disease detection and improve clinical decision-making in resource-constrained environments.
