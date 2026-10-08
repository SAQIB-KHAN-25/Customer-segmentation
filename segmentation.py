import streamlit as st
import pandas as pd
import joblib

# ============================================================
# LOAD TRAINED MODEL AND SCALER
# ============================================================

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🎯",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎯 Customer Segmentation App")

st.write(
    "Enter customer details to predict the customer segment "
    "using the trained K-Means clustering model."
)


# ============================================================
# CUSTOMER INPUT
# ============================================================

st.subheader("Enter Customer Details")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

income = st.number_input(
    "Income",
    min_value=0,
    max_value=200000,
    value=50000
)

total_spending = st.number_input(
    "Total Spending (Sum of purchases)",
    min_value=0,
    max_value=5000,
    value=1000
)

num_web_purchases = st.number_input(
    "Number of Web Purchases",
    min_value=0,
    max_value=100,
    value=10
)

num_store_purchases = st.number_input(
    "Number of Store Purchases",
    min_value=0,
    max_value=100,
    value=10
)

num_web_visits = st.number_input(
    "Number of Web Visits",
    min_value=0,
    max_value=50,
    value=3
)

recency = st.number_input(
    "Recency (Days since last purchase)",
    min_value=0,
    max_value=365,
    value=30
)


# ============================================================
# SEGMENT INFORMATION
# ============================================================

segment_info = {

    0: {
        "name": "Active Valuable Customers",
        "description": (
            "Customers with good spending and strong purchase activity "
            "who have purchased relatively recently."
        ),
        "recommendation": (
            "Use loyalty rewards, personalized offers, "
            "and cross-selling campaigns."
        )
    },

    1: {
        "name": "High-Income Premium Customers",
        "description": (
            "High-income customers with the highest average spending "
            "among the identified customer groups."
        ),
        "recommendation": (
            "Offer premium products, VIP memberships, "
            "exclusive offers, and loyalty rewards."
        )
    },

    2: {
        "name": "Low-Value Customers",
        "description": (
            "Customers with relatively low spending and low purchase "
            "activity, with a higher average age."
        ),
        "recommendation": (
            "Use targeted discounts, affordable product bundles, "
            "and simple re-engagement campaigns."
        )
    },

    3: {
        "name": "At-Risk Valuable Customers",
        "description": (
            "Customers with good spending and purchase activity "
            "but a long period since their last purchase."
        ),
        "recommendation": (
            "Launch win-back campaigns, personalized discounts, "
            "and limited-time offers."
        )
    },

    4: {
        "name": "Low-Value Active Customers",
        "description": (
            "Customers with lower income and spending but relatively "
            "recent engagement."
        ),
        "recommendation": (
            "Use affordable promotions, bundles, and "
            "upselling strategies."
        )
    },

    5: {
        "name": "Inactive Low-Value Customers",
        "description": (
            "Customers with low income and spending and a long period "
            "since their last purchase."
        ),
        "recommendation": (
            "Use reactivation campaigns and targeted incentives "
            "to encourage another purchase."
        )
    }
}


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("🎯 Predict Segment", use_container_width=True):

    # --------------------------------------------------------
    # Create input DataFrame
    # IMPORTANT: Column names MUST match training data
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "Age": [age],
        "Income": [income],
        "Total_Spending": [total_spending],
        "NumWebPurchases": [num_web_purchases],
        "NumStorePurchases": [num_store_purchases],
        "NumWebVisitsMonth": [num_web_visits],
        "Recency": [recency]
    })

    # --------------------------------------------------------
    # Scale input
    # --------------------------------------------------------

    input_scaled = scaler.transform(input_data)

    # --------------------------------------------------------
    # Predict cluster
    # --------------------------------------------------------

    cluster = int(kmeans.predict(input_scaled)[0])

    # --------------------------------------------------------
    # Get segment information
    # --------------------------------------------------------

    info = segment_info[cluster]

    # ========================================================
    # DISPLAY INPUT DATA
    # ========================================================

    st.subheader("📋 Customer Input")

    st.dataframe(
        input_data,
        use_container_width=True
    )

    # ========================================================
    # DISPLAY PREDICTION
    # ========================================================

    st.success(
        f"🎯 Predicted Customer Segment: {cluster}"
    )

    st.header(
        f"Segment {cluster} — {info['name']}"
    )

    # ========================================================
    # CUSTOMER PROFILE
    # ========================================================

    st.subheader("👤 Customer Profile")

    st.write(info["description"])

    # ========================================================
    # BUSINESS RECOMMENDATION
    # ========================================================

    st.subheader("💡 Recommended Business Strategy")

    st.info(info["recommendation"])

    # ========================================================
    # DISTANCE FROM EACH CLUSTER
    # ========================================================

    distances = kmeans.transform(input_scaled)[0]

    distance_df = pd.DataFrame({
        "Segment": range(len(distances)),
        "Distance": distances
    })

    distance_df = distance_df.sort_values(
        "Distance"
    ).reset_index(drop=True)

    st.subheader("📊 Distance from Customer to Each Segment")

    st.dataframe(
        distance_df,
        use_container_width=True
    )

    # ========================================================
    # CUSTOMER SUMMARY
    # ========================================================

    st.subheader("📌 Customer Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Income",
            f"{income:,.0f}"
        )

    with col2:
        st.metric(
            "Total Spending",
            f"{total_spending:,.0f}"
        )

    with col3:
        st.metric(
            "Recency",
            f"{recency} days"
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            "Web Purchases",
            num_web_purchases
        )

    with col5:
        st.metric(
            "Store Purchases",
            num_store_purchases
        )

    with col6:
        st.metric(
            "Web Visits",
            num_web_visits
        )