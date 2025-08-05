# Exploratory-Data-Analysis-EDA-
Exploratory Data Analysis on the Titanic dataset using Python. Includes data cleaning, summary statistics, visualizations, correlation analysis, and insights to prepare the data for machine learning tasks.

## 🎯 Objective

The goal of this task is to explore the **Titanic dataset** using Python libraries to identify patterns, trends, and insights that help understand the structure and features of the data. EDA is an essential first step in any machine learning pipeline.

---

## 📂 Dataset Used

- **Name:** Titanic - Machine Learning from Disaster  
- **Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data)  
- **Description:** Contains data on 891 passengers including age, gender, class, fare, cabin, and survival status.

---

## 🛠 Tools & Libraries Used

- Python  
- Pandas  
- Matplotlib  
- Seaborn  
- Plotly  
- Statsmodels

---

## 🔍 EDA Steps Performed

1. **Data Loading & Overview**
   - Loaded the Titanic dataset using Pandas
   - Displayed the first few rows for basic structure

2. **Summary Statistics**
   - Used `describe()` to find mean, median, std, etc.
   - Identified data types and null values

3. **Data Cleaning**
   - Detected missing values in `Age`, `Cabin`, and `Embarked`
   - Skewness detection using `.skew()`

4. **Visualizations**
   - **Histograms** for numerical features
   - **Boxplots** to detect outliers
   - **Correlation Matrix** using Seaborn Heatmap
   - **Pairplots** for multivariate insights
   - **Interactive Plotly Histogram** colored by survival

5. **Multicollinearity Check**
   - Used Variance Inflation Factor (VIF) to detect multicollinearity between numeric features

---

## 📈 Key Insights

- Female passengers had a significantly higher survival rate.
- 3rd class passengers were the majority but had the lowest survival rate.
- Fare and Age are skewed and need scaling before ML modeling.
- `Pclass`, `Sex`, and `Fare` are the most relevant features for survival prediction.

---

## ✅ Conclusion

This EDA helped uncover critical insights from the Titanic dataset and prepared it for further machine learning applications. Understanding the data through visualizations and statistics is a crucial step before model building.

-
