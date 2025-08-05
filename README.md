# Customer Churn Prediction

This project aims to predict **customer churn** in the telecom industry using demographic and usage data. It implements machine learning models to identify patterns in customer behavior that may indicate a likelihood of churn.

## 📊 Dataset

The dataset combines two sources:
- `telecom_demographics.csv` – Customer demographics (e.g., gender, state, city, telecom partner)
- `telecom_usage.csv` – Customer usage metrics and churn label

These datasets are merged on `customer_id` to form the final dataset.

## ⚙️ Workflow

1. **Data Exploration** – Inspect missing values and unique categories.
2. **Preprocessing**:
   - Handle missing values
   - Convert categorical columns to category types
   - One-hot encode categorical variables
   - Standardize features
3. **Model Training & Evaluation**:
   - Logistic Regression
   - Random Forest Classifier
   - Models are evaluated using confusion matrix and classification report

## 🧠 Algorithms Used

- `LogisticRegression` – A simple linear model for binary classification.
- `RandomForestClassifier` – An ensemble learning method that improves prediction accuracy.

## 📈 Results

Both models were trained and tested. Their performance was compared using accuracy, precision, recall, and F1-score.

## 🛠️ Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## 🧾 License

This project is for educational purposes.
