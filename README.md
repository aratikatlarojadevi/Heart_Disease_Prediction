# Heart Disease Prediction  🩺

An end-to-end Machine Learning web application designed to predict the likelihood of heart disease in patients using standard medical data. Built with Python, this project leverages multiple classification algorithms to analyze medical indicators and delivers automated predictions through a polished, interactive Streamlit user interface.

🚀 Features

1.Exploratory Data Analysis (EDA): Deep analysis and visualization of the heart disease dataset using Plotly (including sunburst charts, distributions, and violin plots) to identify trends, class imbalances, and correlations.

2.Data Cleaning & Imputation:** Handles data inconsistencies, maps categorical text variables into structured numeric data, and utilizes **K-Nearest Neighbors (KNN) Imputation** to handle missing fields (like zero values in Cholesterol).

3.Multi-Model ML Architecture:** Trains, optimizes, and evaluates multiple supervised machine learning models:
  * Decision Trees
  * Logistic Regression
  * Random Forest Classifier
  * Support Vector Machine (SVM)
  * GridSearchCV for hyperparameter tuning.
   
4.Interactive Streamlit Web Dashboard: Features a clean, dark-themed dashboard divided into three functional tabs:
Predict: A manual data-entry interface with sliders and dropdowns for individual patient logs to receive live predictions from all trained models simultaneously.
Bulk Predict:** An automated batch processor where users can upload a single `.csv` file, view the raw predictions table, and download the evaluated sheet instantly.
Model Information: An evaluation metric interface showcasing a live performance bar chart comparing accuracies across all trained classifiers.



📊 Dataset Features

The predictive models ingest 11 core clinical attributes to determine the target variable (`HeartDisease`):

| Feature Name | Description | Value Range / Classes |

| Age | Age of the patient | Years |
| Sex | Gender of the patient | `Male`, `Female` |
| ChestPainType | Type of chest pain | `Typical Angina`, `Atypical Angina`, `Non-Anginal Pain`, `Asymptomatic` |
| RestingBP | Resting Blood Pressure | mm Hg |
| Cholesterol | Serum Cholesterol | mm/dl |
| FastingBS | Fasting Blood Sugar level | `<= 120 mg/dl`, `> 120 mg/dl` |
| RestingECG | Resting electrocardiogram results | `Normal`, `ST-T Wave Abnormality`, `Left Ventricular Hypertrophy` |
| MaxHR | Maximum Heart Rate achieved | Beats per minute (BPM) |
| ExerciseAngina | Exercise-induced chest pain | `Yes`, `No` |
| Oldpeak | ST depression induced by exercise | Numeric value measured in depression |
| ST_Slope | The slope of the peak exercise ST segment | `Upsloping`, `Flat`, `Downsloping` |


 🛠️ Tech Stack

* Language: Python
* Data Libraries: Pandas, NumPy
* Visualization:Plotly Express
* Machine Learning: Scikit-Learn
* Model Deployment:Streamlit
* Model Serialization: Pickle (`.pkl`)



