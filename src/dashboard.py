import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/churn.csv")

# Create Age Groups
df['AgeGroup'] = pd.cut(df['Age'],
                       bins=[18,30,45,60,75,100],
                       labels=["<30","30-45","45-60","60-75","75+"])

# Create subplot layout (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

fig.suptitle("Customer Churn Analysis Dashboard", fontsize=16)

# -------------------------------
# 1. Gender Distribution (Pie)
# -------------------------------
gender_counts = df['Gender'].value_counts()
axes[0,0].pie(gender_counts, labels=gender_counts.index,
              autopct='%1.1f%%')
axes[0,0].set_title("Gender Distribution")

# -------------------------------
# 2. Age Distribution (Histogram)
# -------------------------------
sns.histplot(df['Age'], bins=20, ax=axes[0,1])
axes[0,1].set_title("Age Distribution")

# -------------------------------
# 3. Age Group Distribution
# -------------------------------
sns.countplot(x='AgeGroup', data=df, ax=axes[0,2])
axes[0,2].set_title("Age Group Distribution")

# -------------------------------
# 4. Churn Distribution
# -------------------------------
sns.countplot(x='Exited', data=df, ax=axes[1,0])
axes[1,0].set_title("Churn Distribution")

# -------------------------------
# 5. Gender vs Churn
# -------------------------------
sns.countplot(x='Gender', hue='Exited', data=df, ax=axes[1,1])
axes[1,1].set_title("Churn by Gender")

# -------------------------------
# 6. Age vs Churn
# -------------------------------
sns.histplot(data=df, x='Age', hue='Exited', bins=20, ax=axes[1,2])
axes[1,2].set_title("Age Distribution by Churn")

plt.tight_layout()
plt.show()