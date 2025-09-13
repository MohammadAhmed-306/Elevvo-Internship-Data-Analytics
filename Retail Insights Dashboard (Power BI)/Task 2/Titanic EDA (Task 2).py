# ========================================
# 1. Import Libraries
# ========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white")  # clean style, no background grids

# ========================================
# 2. Load Dataset
# ========================================
df = pd.read_csv("train.csv")

# ========================================
# 3. Handle Missing Values
# ========================================
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Age"] = df["Age"].fillna(df["Age"].median())

df["HasCabin"] = df["Cabin"].notna().astype(int)
df["Cabin"] = df["Cabin"].fillna("Unknown")

# ========================================
# 4. Feature Engineering
# ========================================
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

df["Name"] = df["Name"].astype(str)
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")[0].str.strip()
rare_titles = df["Title"].value_counts()[df["Title"].value_counts() < 10].index
df["Title"] = df["Title"].replace(rare_titles, "Rare")

bins = [0, 12, 18, 30, 45, 60, 80]
labels = ["Child", "Teen", "YoungAdult", "Adult", "MidAge", "Senior"]
df["AgeBand"] = pd.cut(df["Age"], bins=bins, labels=labels)

# ========================================
# 5. Chart Pair 1 – Sex & Pclass (Bars)
# ========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

surv_by_sex = df.groupby("Sex")["Survived"].mean() * 100
sns.barplot(x=surv_by_sex.index, y=surv_by_sex.values,
            ax=axes[0], color="steelblue", edgecolor="black")
axes[0].set_title("Survival Rate by Sex (%)")
axes[0].set_ylabel("Survival Rate (%)")
axes[0].grid(False)

surv_by_pclass = df.groupby("Pclass")["Survived"].mean() * 100
sns.barplot(x=surv_by_pclass.index.astype(str), y=surv_by_pclass.values,
            ax=axes[1], color="orange", edgecolor="black")
axes[1].set_title("Survival Rate by Passenger Class (%)")
axes[1].set_ylabel("Survival Rate (%)")
axes[1].grid(False)

plt.tight_layout()
plt.show()

# ========================================
# 6. Chart Pair 2 – Embarked (Pie) & Title (Horizontal Bar)
# ========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

surv_by_embarked = df.groupby("Embarked")["Survived"].mean() * 100
axes[0].pie(surv_by_embarked.values, labels=surv_by_embarked.index,
            autopct="%1.1f%%", startangle=90,
            colors=["skyblue", "lightgreen", "salmon"])
axes[0].set_title("Survival Rate by Embarkation Port")

surv_by_title = df.groupby("Title")["Survived"].mean() * 100
sns.barplot(y=surv_by_title.index, x=surv_by_title.values,
            ax=axes[1], palette="muted", edgecolor="black")
axes[1].set_title("Survival Rate by Title (%)")
axes[1].set_xlabel("Survival Rate (%)")
axes[1].grid(False)

plt.tight_layout()
plt.show()

# ========================================
# 7. Chart Pair 3 – Age Band (Line) & Family Size (Area)
# ========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

surv_by_ageband = df.groupby("AgeBand")["Survived"].mean() * 100
axes[0].plot(surv_by_ageband.index.astype(str), surv_by_ageband.values,
             marker="o", color="purple", linewidth=2)
axes[0].set_title("Survival Rate by Age Band")
axes[0].set_ylabel("Survival Rate (%)")
axes[0].grid(False)

surv_by_famsize = df.groupby("FamilySize")["Survived"].mean() * 100
axes[1].fill_between(surv_by_famsize.index, surv_by_famsize.values,
                     color="lightcoral", alpha=0.6)
axes[1].plot(surv_by_famsize.index, surv_by_famsize.values,
             marker="o", color="red")
axes[1].set_title("Survival Rate by Family Size")
axes[1].set_xlabel("Family Size")
axes[1].set_ylabel("Survival Rate (%)")
axes[1].grid(False)

plt.tight_layout()
plt.show()

# ========================================
# 8. Chart Pair 4 – Fare (Boxplot) & Correlation (Heatmap)
# ========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.boxplot(x="Survived", y="Fare", data=df, ax=axes[0])
axes[0].set_title("Fare Distribution by Survival")
axes[0].grid(False)

corr = df[["Survived", "Age", "Fare", "FamilySize", "IsAlone", "HasCabin"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=axes[1], cbar=True)
axes[1].set_title("Correlation Heatmap")

plt.tight_layout()
plt.show()

# ========================================
# 9. Save Cleaned Dataset
# ========================================
df.to_csv("titanic_cleaned.csv", index=False)
print("\nCleaned dataset saved as titanic_cleaned.csv")
