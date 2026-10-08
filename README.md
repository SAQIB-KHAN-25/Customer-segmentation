# 🎯 Customer Segmentation using K-Means Clustering

An end-to-end **Machine Learning project** that uses **K-Means Clustering** to segment customers based on their demographic information, purchasing behavior, online activity, income, spending patterns, and recency.

The project includes **data cleaning, exploratory data analysis, feature engineering, visualization, unsupervised machine learning, model persistence, and an interactive Streamlit application** for predicting customer segments.

---

## 📌 Project Overview

Understanding different types of customers is important for businesses to create targeted marketing strategies.

Instead of treating every customer the same, this project uses **K-Means Clustering** to group customers with similar characteristics.

The model analyzes:

- 👤 Customer Age
- 💰 Income
- 🛍️ Total Spending
- 🌐 Web Purchases
- 🏪 Store Purchases
- 👀 Web Visits
- 📅 Recency

The final model divides customers into **6 different segments**.

The trained model is integrated into a **Streamlit web application**, where users can enter customer details and receive:

- Predicted customer segment
- Customer segment description
- Recommended marketing strategy
- Distance from each cluster

---

# 🚀 Features

### 📊 Exploratory Data Analysis

The project performs analysis on:

- Dataset structure
- Missing values
- Statistical summaries
- Education distribution
- Marital status distribution
- Age distribution
- Income distribution
- Spending distribution
- Correlation between variables

### 🛠️ Feature Engineering

New features are created from the original dataset:

```text
Age
Total_Spending
Total_children
Customer_Since
AcceptedAny
AgeGroup
