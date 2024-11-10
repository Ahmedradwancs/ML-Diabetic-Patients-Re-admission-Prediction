# Diabetic Patients' Re-admission Prediction

## Course Information
This project is part of the **DA380A HT24 Machine Learning** course at Kristianstad University.

## Project Overview
This project aims to predict hospital readmissions for diabetic patients within 30 days post-discharge. Using a dataset of around 100,000 hospital records, we developed a machine learning pipeline to identify high-risk patients and optimize healthcare resource allocation.

### Key Objectives
1. **Predictive Modeling**: Identify diabetic patients at high risk of readmission.
2. **Feature Analysis**: Determine factors contributing to readmissions to help inform healthcare strategies.

## Dataset
The dataset is derived from the **Diabetes 130-US Hospitals** dataset, which includes patient records from 130 hospitals in the United States between 1999 and 2008. It contains:
- **50 features**: Patient demographics, hospital admission details, and medical information, including medications and lab results.
- **Target Variable**: `readmitted` - indicates whether a patient was readmitted within 30 days, after 30 days, or not readmitted.

Data Source: [UCI Machine Learning Repository - Diabetes 130-US hospitals dataset](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)

## Methodology
### 1. Data Exploration and Preprocessing
   - **Clustering**: K-means and DBSCAN were used to explore potential patterns but showed minimal clustering due to the complex nature of the data.
   - **Handling Missing Values**: Imputed missing values for `payer_code` and `race`, and removed features with excessive missing data (e.g., `weight`).
   - **Outlier Management**: Retained outliers in features like `Time in Hospital` due to their high predictive importance.
   - **Feature Engineering**: Created `age_group` and used feature selection techniques to retain top predictors (e.g., `number_inpatient`, `number_diagnoses`).
   - **Encoding**: Categorical features were encoded using one-hot encoding, and binary encoding for features with two values.
   - **Balancing**: Applied SMOTE to address class imbalance.
   - **Scaling**: Standardized numerical features for consistency.

### 2. Model Selection
   - **Models**: We implemented Logistic Regression, Random Forest, and Gradient Boosting (XGBoost) along with AdaBoost and MLP.
   - **Model Improvement**: Hyperparameter tuning (GridSearchCV) was performed for optimal performance. SMOTE was used to improve recall for minority classes.

### 3. Evaluation
   - **Metrics**: Models were evaluated using accuracy, precision, recall, F1-score, and AUC-ROC.
   - **Best Performing Model**: The Gradient Boosting model achieved the highest accuracy, followed closely by the Random Forest model.

## Results
- **Gradient Boosting (XGBoost)**: Showed the highest predictive accuracy and balanced precision-recall performance.
- **Balanced Random Forest**: Achieved good recall, suitable for applications prioritizing detection of high-risk patients.
- **Key Features**: Important predictors included `number_inpatient`, `number_diagnoses`, and `number_emergency`.

## Discussion
### Challenges
- **Class Imbalance**: The dataset had a small proportion of readmissions, which was addressed through SMOTE.
- **Data Quality**: Extensive preprocessing was needed due to missing values and high-cardinality features.

### Lessons Learned
- Robust preprocessing and feature engineering significantly impact model performance.
- Handling imbalanced datasets is crucial in healthcare prediction tasks to ensure accuracy across all classes.

### Future Work
- **Deep Learning**: Explore models such as LSTM for sequential data analysis.
- **Data Enrichment**: Integrate additional data sources to improve prediction accuracy.

## How to Run the Project
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter Notebook to preprocess the data, train models, and evaluate results.
4. For deployment, we integrated a FastAPI-based web interface allowing users to select models and upload data for predictions.

## Dependencies
- Python
- scikit-learn
- XGBoost
- FastAPI
- Pandas, Numpy

## Contributors
- [Ahmed Radwan](https://github.com/Ahmedradwancs)
- [Sam El Saati](https://github.com/sams258)
