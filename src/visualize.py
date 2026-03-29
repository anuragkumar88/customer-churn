import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_graphs():
    df = pd.read_csv("data/churn.csv")

    sns.countplot(x='Exited', data=df)
    plt.title("Churn Distribution")
    plt.show()

    sns.boxplot(x='Exited', y='Age', data=df)
    plt.title("Age vs Churn")
    plt.show()

    sns.countplot(x='Geography', hue='Exited', data=df)
    plt.title("Churn by Geography")
    plt.show()