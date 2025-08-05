# Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Load the Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Display first few rows
print("First 5 rows:")
print(df.head())

# 1. Summary statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# 2. Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Data info
print("\nData Info:")
print(df.info())

# 4. Histograms for numerical features
df.hist(figsize=(12, 10), color='skyblue', edgecolor='black')
plt.suptitle('Histograms of Numeric Features')
plt.show()

# 5. Boxplots to detect outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Age distribution across Pclass')
plt.show()

# 6. Correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 7. Pairplot
sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']].dropna(), hue='Survived')
plt.suptitle('Pairplot of Features Colored by Survival', y=1.02)
plt.show()

# 8. Detecting skewness
print("\nSkewness of Numeric Features:")
print(df[['Age', 'Fare']].skew())

# 9. Plotly interactive chart
fig = px.histogram(df, x="Age", color="Survived", nbins=30, title="Age Distribution by Survival")
fig.show()

# 10. Variance Inflation Factor (VIF)
X = df[['Age', 'Fare', 'Pclass']].dropna()
X = add_constant(X)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVariance Inflation Factor (VIF):")
print(vif_data)
