# What sorts of people were more likely to survive?”
# Using passenger data (ie name, age, gender, socio-economic class, etc)

### Imports ###
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


### Pandas display options ###
pd.set_option('display.max_rows', None)      # Show all rows
pd.set_option('display.max_columns', None)   # Show all columns
pd.set_option('display.width', None)         # Auto-detect terminal width
pd.set_option('display.max_colwidth', None)  # Show full column content


### Load data ###
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")


### Data Cleaning ###
# Drop Cabin column due to many missing values
train_df = train_df.drop(columns=['Cabin'])
test_df = test_df.drop(columns=['Cabin'])

# Fill missing Embarked with most frequent value
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# Fill missing Age with median
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())

# Fill missing Age and Fare in test set
test_df['Age'] = test_df['Age'].fillna(train_df['Age'].median())
test_df['Fare'] = test_df['Fare'].fillna(train_df['Fare'].median())


### Exploratory Data Analysis ###
# Visualize survival rate by Sex
sns.barplot(data=train_df, x='Sex', y='Survived')
# plt.show()

# Visualize survival rate by Passenger Class
sns.barplot(data=train_df, x='Pclass', y='Survived')
# plt.show()

# Correlation heatmap for numerical features
df_numeric = train_df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(10,6))
sns.heatmap(data=df_numeric.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap (Numerical)")
# plt.show()


### Feature Engineering ###
# Encode Sex: male=0, female=1
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode Embarked, drop first to avoid dummy trap
train_df = pd.get_dummies(train_df, columns=['Embarked'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Embarked'], drop_first=True)

# Ensure test set has same Embarked columns as train set
for col in ['Embarked_Q', 'Embarked_S']:
    if col not in test_df.columns:
        test_df[col] = 0

# Combine SibSp and Parch into FamilySize
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

# Extract Title from Name
train_df['Title'] = train_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
test_df['Title'] = test_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

# Group rare titles together
rare_titles = ['Dr','Rev','Major','Col','Countess','Capt','Sir','Lady','Don','Jonkheer']

train_df['Title'] = train_df['Title'].replace(rare_titles, 'Rare')
test_df['Title'] = test_df['Title'].replace(rare_titles, 'Rare')

# Standardize French titles
train_df['Title'] = train_df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
test_df['Title'] = test_df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

# Map titles to numeric
title_mapping = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
train_df['Title'] = train_df['Title'].map(title_mapping)
test_df['Title'] = test_df['Title'].map(title_mapping)
test_df['Title'] = test_df['Title'].fillna(4)  # fill missing titles with Rare


### Drop unneeded columns ###
train_df = train_df.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket'])
test_df = test_df.drop(columns=['Name', 'SibSp', 'Parch', 'Ticket'])


### Model Training - Logistic Regression ###
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Accuracy and reports can be printed if needed
# print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))


### Feature Importance - Logistic Regression ###
coeffs = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0],
    'Impact': np.abs(model.coef_[0])
}).sort_values(by='Impact', ascending=False)
# print(coeffs)


### Model Training - Random Forest ###
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
# print(classification_report(y_test, y_pred_rf))


### Feature Importance - Random Forest ###
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
# print(importances)


### Kaggle Submission Preparation ###
feature_columns = ["Pclass", "Sex", "Age", "Fare",
                   "Embarked_Q", "Embarked_S",
                   "FamilySize", "Title"]

X_kaggle = test_df[feature_columns]

preds = rf_model.predict(X_kaggle)

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": preds
})

# submission.to_csv("submission.csv", index=False)