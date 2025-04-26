# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
try:
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris['data']
    df['target'] = iris['target']
except Exception as e:
    print("Error loading dataset:", e)

# Explore Dataset
print(df.head())
print(df.info())
print(df.isnull().sum())

# Clean Dataset (no missing values in iris, but shown here)
df = df.dropna()

# Basic Data Analysis
print(df.describe())

# Grouping
grouped = df.groupby('target').mean()
print(grouped)

# Data Visualization

# 1. Line Chart (using petal length over sample index)
plt.figure(figsize=(8,5))
plt.plot(df.index, df['petal length (cm)'], label='Petal Length')
plt.title('Petal Length over Samples')
plt.xlabel('Sample Index')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.grid(True)
plt.show()

# 2. Bar Chart (average petal length per species)
plt.figure(figsize=(8,5))
grouped['petal length (cm)'].plot(kind='bar')
plt.title('Average Petal Length per Species')
plt.xlabel('Species (Target)')
plt.ylabel('Average Petal Length (cm)')
plt.grid(axis='y')
plt.show()

# 3. Histogram (distribution of sepal length)
plt.figure(figsize=(8,5))
plt.hist(df['sepal length (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 4. Scatter Plot (sepal length vs petal length)
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='target', palette='deep')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.grid(True)
plt.show()
