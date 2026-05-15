import matplotlib.pyplot as plt
import seaborn as sns

def plot_age_distribution(df):

    plt.figure(figsize=(8,5))

    sns.histplot(df["Age"], kde=True)

    plt.title("Age Distribution")

    plt.show()