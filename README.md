# 🚢 Titanic Survival Prediction

This project explores the Titanic passenger dataset to predict survival outcomes using machine learning. The workflow includes data cleaning, exploratory data analysis (EDA), feature engineering, model training, and generating predictions for Kaggle submission.

---

## 📁 Data

- `train.csv` — Training data (891 passengers)
- `test.csv` — Test data for Kaggle submission (418 passengers)

---

## 🔍 Project Workflow

### 1. Data Preprocessing
- Dropped irrelevant columns: `Cabin`, `Ticket`, `PassengerId`
- Imputed missing values:
  - `Age` with median
  - `Embarked` with mode
  - `Fare` in test data with median
- Feature Engineering:
  - Created `FamilySize` (`SibSp` + `Parch` + 1)
  - Extracted `Title` from `Name` and grouped rare titles
  - One-hot encoded categorical features

### 2. Exploratory Data Analysis (EDA)
- Analyzed survival distributions:
  - **Gender**: 74% of females survived vs. 19% of males
  - **Class**: 63% in 1st class, 47% in 2nd, and 24% in 3rd
- Visual tools:
  - Bar plots
  - Heatmaps of feature correlations

### 3. Model Training & Evaluation
- **Baseline Model**: Logistic Regression — ~78% accuracy
- **Final Model**: Random Forest Classifier — ~82% accuracy
- Metrics:
  - Accuracy
  - Classification report (precision, recall, F1-score)
  - Confusion matrix

### 4. Kaggle Submission
- Predictions made using final model on test set
- Exported to `submission.csv`
- Achieved Kaggle Score: *0.75358*

---

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 📊 Results Summary

| Model               | Accuracy |
|--------------------|--------|
| Logistic Regression| ~78%   |
| Random Forest      | ~82%   |
| Kaggle Score       | ~75%    |

---

## 🧠 Key Learnings

- Strong impact of **gender**, **class**, and **fare paid** on survival
- `Fare` was a top feature in the Random Forest model, indicating socioeconomic status significantly influenced survival outcomes
- Effective feature engineering (like `Title` and `FamilySize`) helped boost model performance
- Random Forests provided a more flexible and accurate model than Logistic Regression by capturing non-linear relationships

---

## 📎 License

This project is licensed under the **MIT License**.
