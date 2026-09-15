# ============================================================
# CUSTOMER SEGMENTATION PROJECT
# Complete Data Analysis + Feature Engineering + K-Means
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import joblib


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("customer_segmentation.csv")

print("Dataset loaded successfully!")


# ============================================================
# 3. BASIC DATA EXPLORATION
# ============================================================

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== MISSING VALUES ==========")
print(df.isna().sum())

print("\nTotal missing values:", df.isna().sum().sum())


# ============================================================
# 4. REMOVE MISSING VALUES
# ============================================================

df.dropna(inplace=True)

print("\nMissing values after cleaning:")
print(df.isna().sum().sum())


# ============================================================
# 5. STATISTICAL SUMMARY
# ============================================================

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe())


# ============================================================
# 6. CATEGORICAL DATA ANALYSIS
# ============================================================

print("\n========== EDUCATION ==========")
print(df["Education"].value_counts())

print("\n========== MARITAL STATUS ==========")
print(df["Marital_Status"].value_counts())


# ============================================================
# 7. CONVERT CUSTOMER DATE
# ============================================================

df["Dt_Customer"] = pd.to_datetime(
    df["Dt_Customer"],
    dayfirst=True
)

print("\nDate conversion completed.")


# ============================================================
# 8. FEATURE ENGINEERING — AGE
# ============================================================

df["Age"] = 2025 - df["Year_Birth"]

print("\n========== AGE ==========")
print(df["Age"].head())


# ============================================================
# 9. FEATURE ENGINEERING — TOTAL CHILDREN
# ============================================================

df["Total_children"] = (
    df["Kidhome"] +
    df["Teenhome"]
)

print("\n========== TOTAL CHILDREN ==========")
print(df["Total_children"].head())


# ============================================================
# 10. FEATURE ENGINEERING — TOTAL SPENDING
# ============================================================

spend_cols = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

df["Total_Spending"] = df[spend_cols].sum(axis=1)

print("\n========== TOTAL SPENDING ==========")
print(df["Total_Spending"].head())


# ============================================================
# 11. FEATURE ENGINEERING — CUSTOMER SINCE
# ============================================================

df["Customer_Since"] = (
    pd.Timestamp("2025-01-01") -
    df["Dt_Customer"]
).dt.days

print("\n========== CUSTOMER SINCE ==========")
print(df["Customer_Since"].head())


# ============================================================
# 12. VISUALIZATION — AGE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Age"],
    bins=30,
    kde=True
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")

plt.show()


# ============================================================
# 13. VISUALIZATION — INCOME DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Income"],
    bins=30,
    kde=True
)

plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Number of Customers")

plt.show()


# ============================================================
# 14. VISUALIZATION — TOTAL SPENDING
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Total_Spending"],
    bins=30,
    kde=True
)

plt.title("Total Spending Distribution")
plt.xlabel("Total Spending")
plt.ylabel("Number of Customers")

plt.show()


# ============================================================
# 15. INCOME BY EDUCATION
# ============================================================

plt.figure(figsize=(9, 5))

sns.boxplot(
    x="Education",
    y="Income",
    data=df
)

plt.xticks(rotation=45)

plt.title("Income Distribution by Education Level")
plt.xlabel("Education")
plt.ylabel("Income")

plt.show()


# ============================================================
# 16. SPENDING BY MARITAL STATUS
# ============================================================

plt.figure(figsize=(10, 5))

sns.boxplot(
    x="Marital_Status",
    y="Total_Spending",
    data=df
)

plt.xticks(rotation=45)

plt.title("Total Spending Distribution by Marital Status")
plt.xlabel("Marital Status")
plt.ylabel("Total Spending")

plt.show()


# ============================================================
# 17. CORRELATION ANALYSIS
# ============================================================

corr_features = [
    "Income",
    "Age",
    "Recency",
    "Total_Spending",
    "NumWebPurchases",
    "NumStorePurchases"
]

corr = df[corr_features].corr()

print("\n========== CORRELATION MATRIX ==========")
print(corr)


# ============================================================
# 18. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(9, 6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")

plt.show()


# ============================================================
# 19. INCOME BY EDUCATION AND MARITAL STATUS
# ============================================================

pivot_income = df.pivot_table(
    values="Income",
    index="Education",
    columns="Marital_Status",
    aggfunc="mean"
)

print("\n========== AVERAGE INCOME ==========")
print(pivot_income)


# ============================================================
# 20. PIVOT TABLE HEATMAP
# ============================================================

plt.figure(figsize=(10, 6))

sns.heatmap(
    pivot_income,
    annot=True,
    fmt=".0f",
    cmap="YlGnBu"
)

plt.title(
    "Average Income by Education and Marital Status"
)

plt.show()


# ============================================================
# 21. AVERAGE SPENDING BY EDUCATION
# ============================================================

group1 = (
    df.groupby("Education")["Total_Spending"]
    .mean()
    .sort_values(ascending=False)
)

print("\n========== AVERAGE SPENDING BY EDUCATION ==========")
print(group1)


# ============================================================
# 22. SPENDING BY EDUCATION BAR CHART
# ============================================================

plt.figure(figsize=(9, 5))

group1.plot(
    kind="bar"
)

plt.title(
    "Average Total Spending by Education Level"
)

plt.ylabel("Average Total Spending")
plt.xlabel("Education")

plt.xticks(rotation=45)

plt.show()


# ============================================================
# 23. CAMPAIGN ACCEPTANCE
# ============================================================

campaign_cols = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "Response"
]

df["AcceptedAny"] = (
    df[campaign_cols]
    .sum(axis=1)
)

df["AcceptedAny"] = (
    df["AcceptedAny"]
    .apply(lambda x: 1 if x > 0 else 0)
)

print("\n========== ACCEPTED ANY CAMPAIGN ==========")
print(df["AcceptedAny"].value_counts())


# ============================================================
# 24. CAMPAIGN ACCEPTANCE BY MARITAL STATUS
# ============================================================

group2 = (
    df.groupby("Marital_Status")["AcceptedAny"]
    .mean()
    .sort_values(ascending=False)
)

print("\n========== ACCEPTANCE RATE BY MARITAL STATUS ==========")
print(group2)


# ============================================================
# 25. ACCEPTANCE RATE CHART
# ============================================================

plt.figure(figsize=(10, 5))

group2.plot(
    kind="bar"
)

plt.title(
    "Average Acceptance Rate by Marital Status"
)

plt.ylabel("Acceptance Rate")
plt.xlabel("Marital Status")

plt.xticks(rotation=45)

plt.show()


# ============================================================
# 26. AGE GROUP
# ============================================================

age_bins = [
    18,
    30,
    40,
    50,
    60,
    70,
    90
]

age_labels = [
    "18-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70+"
]

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=age_bins,
    labels=age_labels
)

print("\n========== AGE GROUP ==========")
print(df["AgeGroup"].value_counts())


# ============================================================
# 27. AVERAGE INCOME BY AGE GROUP
# ============================================================

group3 = (
    df.groupby(
        "AgeGroup",
        observed=False
    )["Income"]
    .mean()
)

print("\n========== AVERAGE INCOME BY AGE GROUP ==========")
print(group3)


# ============================================================
# 28. AGE GROUP INCOME CHART
# ============================================================

plt.figure(figsize=(9, 5))

group3.plot(
    kind="barh"
)

plt.title(
    "Average Income by Age Group"
)

plt.xlabel("Average Income")
plt.ylabel("Age Group")

plt.show()


# ============================================================
# 29. FINAL FEATURES FOR MACHINE LEARNING
# ============================================================

features = [
    "Age",
    "Income",
    "Total_Spending",
    "NumWebPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Recency"
]

print("\n========== FEATURES USED FOR MODEL ==========")

for feature in features:
    print(feature)


# ============================================================
# 30. CREATE X
# ============================================================

X = df[features].copy()

print("\n========== MODEL INPUT ==========")
print(X.head())


# ============================================================
# 31. CHECK MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES IN MODEL FEATURES ==========")
print(X.isna().sum())


# ============================================================
# 32. FEATURE SCALING
# ============================================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n========== SCALED DATA ==========")
print(X_scaled[:5])


# ============================================================
# 33. ELBOW METHOD
# ============================================================

from sklearn.cluster import KMeans

wcss = []

for i in range(2, 10):

    kmeans_temp = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans_temp.fit(X_scaled)

    wcss.append(
        kmeans_temp.inertia_
    )


# ============================================================
# 34. ELBOW GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 10),
    wcss,
    marker="o"
)

plt.title(
    "Elbow Method for Optimal Number of Clusters"
)

plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.show()


# ============================================================
# 35. FINAL K-MEANS MODEL
# ============================================================

kmeans = KMeans(
    n_clusters=6,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(
    X_scaled
)

print("\nK-Means clustering completed!")


# ============================================================
# 36. CLUSTER SUMMARY
# ============================================================

cluster_summary = (
    df.groupby("Cluster")[features]
    .mean()
    .round(2)
)

print("\n==============================================")
print("         CUSTOMER SEGMENT CHARACTERISTICS")
print("==============================================")

print(cluster_summary)


# ============================================================
# 37. CUSTOMER COUNT PER CLUSTER
# ============================================================

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

print("\n==============================================")
print("             CUSTOMER COUNT")
print("==============================================")

print(cluster_counts)


# ============================================================
# 38. SEGMENT DISTRIBUTION
# ============================================================

cluster_percentage = (
    df["Cluster"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

print("\n==============================================")
print("          SEGMENT DISTRIBUTION (%)")
print("==============================================")

print(cluster_percentage)


# ============================================================
# 39. CLUSTER VISUALIZATION USING PCA
# ============================================================

from sklearn.decomposition import PCA

pca = PCA(
    n_components=2
)

pca_data = pca.fit_transform(
    X_scaled
)

df["PCA1"] = pca_data[:, 0]
df["PCA2"] = pca_data[:, 1]


# ============================================================
# 40. PCA SCATTER PLOT
# ============================================================

plt.figure(figsize=(10, 7))

sns.scatterplot(
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    data=df,
    palette="Set1",
    s=60
)

plt.title(
    "Customer Segmentation using K-Means"
)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.legend(
    title="Cluster"
)

plt.show()


# ============================================================
# 41. SAVE K-MEANS MODEL
# ============================================================

joblib.dump(
    kmeans,
    "kmeans_model.pkl"
)


# ============================================================
# 42. SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    "scaler.pkl"
)


# ============================================================
# 43. SAVE CLUSTER SUMMARY
# ============================================================

cluster_summary.to_csv(
    "cluster_summary.csv"
)


# ============================================================
# 44. FINAL OUTPUT
# ============================================================

print("\n==============================================")
print("             MODEL TRAINING COMPLETE")
print("==============================================")

print("\nFiles created:")

print("1. kmeans_model.pkl")
print("2. scaler.pkl")
print("3. cluster_summary.csv")

print("\nNumber of clusters:", kmeans.n_clusters)

print("\nFeatures used:")

for feature in features:
    print(" -", feature)

print("\nModel training completed successfully!")