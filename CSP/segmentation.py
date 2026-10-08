
import os
import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🎯",
    layout="centered"
)


# ============================================================
# FIND PROJECT DIRECTORY
# ============================================================

# Get the folder where segmentation.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# FILE PATHS
# ============================================================

KMEANS_PATH = os.path.join(
    BASE_DIR,
    "kmeans_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "scaler.pkl"
)

SUMMARY_PATH = os.path.join(
    BASE_DIR,
    "cluster_summary.csv"
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

missing_files = []

if not os.path.exists(KMEANS_PATH):
    missing_files.append("kmeans_model.pkl")

if not os.path.exists(SCALER_PATH):
    missing_files.append("scaler.pkl")

if not os.path.exists(SUMMARY_PATH):
    missing_files.append("cluster_summary.csv")


if missing_files:

    st.error(
        "The following required file(s) were not found:"
    )

    for file in missing_files:
        st.write(f"❌ {file}")

    st.warning(
        "Make sure these files are present in the same folder "
        "as segmentation.py."
    )

    st.info(
        f"Expected folder:\n\n{BASE_DIR}"
    )

    st.stop()


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

try:

    kmeans = joblib.load(
        KMEANS_PATH
    )

except Exception as e:

    st.error(
        f"Error loading kmeans_model.pkl: {e}"
    )

    st.stop()


# ============================================================
# LOAD SCALER
# ============================================================

try:

    scaler = joblib.load(
        SCALER_PATH
    )

except Exception as e:

    st.error(
        f"Error loading scaler.pkl: {e}"
    )

    st.stop()


# ============================================================
# LOAD CLUSTER SUMMARY
# ============================================================

try:

    cluster_summary = pd.read_csv(
        SUMMARY_PATH
    )

except Exception as e:

    st.error(
        f"Error loading cluster_summary.csv: {e}"
    )

    st.stop()


# ============================================================
# CHECK CLUSTER COLUMN
# ============================================================

if "Cluster" not in cluster_summary.columns:

    st.error(
        "cluster_summary.csv must contain a 'Cluster' column."
    )

    st.stop()


cluster_summary["Cluster"] = (
    cluster_summary["Cluster"].astype(int)
)


# ============================================================
# VERIFY MODEL CLUSTERS
# ============================================================

model_clusters = set(
    range(kmeans.n_clusters)
)


summary_clusters = set(
    cluster_summary["Cluster"]
)


if model_clusters != summary_clusters:

    st.error(
        "Model and cluster summary contain different "
        "cluster IDs."
    )

    st.write(
        "Clusters in KMeans model:",
        sorted(model_clusters)
    )

    st.write(
        "Clusters in cluster_summary.csv:",
        sorted(summary_clusters)
    )

    st.warning(
        "Please regenerate the KMeans model and "
        "cluster_summary.csv together."
    )

    st.stop()


# ============================================================
# SEGMENT INFORMATION
# ============================================================

segment_info = {

    0: {
        "name": "Inactive Low-Value Customers",

        "description": (
            "Customers with low income and very low spending, "
            "low purchase activity, and the highest average "
            "recency among the identified customer groups."
        ),

        "recommendation": (
            "Use reactivation campaigns, targeted discounts, "
            "and personalized incentives to encourage purchases."
        )
    },


    1: {
        "name": "Active Valuable Customers",

        "description": (
            "Customers with relatively high income and spending, "
            "strong web and store purchase activity, and moderate "
            "recency."
        ),

        "recommendation": (
            "Use loyalty rewards, personalized offers, "
            "cross-selling, and customer retention campaigns."
        )
    },


    2: {
        "name": "High-Income Premium Customers",

        "description": (
            "High-income customers with high spending and strong "
            "store purchase activity."
        ),

        "recommendation": (
            "Offer premium products, VIP memberships, "
            "exclusive offers, and loyalty rewards."
        )
    },


    3: {
        "name": "Low-Value Active Customers",

        "description": (
            "Customers with low income and spending but relatively "
            "recent purchasing activity compared with the other "
            "low-value customer groups."
        ),

        "recommendation": (
            "Use affordable promotions, product bundles, "
            "and upselling strategies to increase customer value."
        )
    },


    4: {
        "name": "Highest-Value Premium Customers",

        "description": (
            "The highest-income and highest-spending customer group "
            "with strong store purchasing activity."
        ),

        "recommendation": (
            "Provide VIP treatment, premium products, exclusive "
            "benefits, personalized recommendations, and "
            "high-value loyalty rewards."
        )
    },


    5: {
        "name": "Low-Value Inactive Customers",

        "description": (
            "Customers with relatively low income and spending, "
            "low purchase activity, and relatively high recency."
        ),

        "recommendation": (
            "Use targeted re-engagement campaigns, discounts, "
            "and affordable product bundles."
        )
    }
}


# ============================================================
# VERIFY SEGMENT INFORMATION
# ============================================================

if set(segment_info.keys()) != model_clusters:

    st.error(
        "Segment information does not match the model "
        "cluster IDs."
    )

    st.write(
        "Model clusters:",
        sorted(model_clusters)
    )

    st.write(
        "Segment information:",
        sorted(segment_info.keys())
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title(
    "🎯 Customer Segmentation App"
)


st.write(
    "Enter customer details to predict the customer segment "
    "using the trained K-Means clustering model."
)


# ============================================================
# CUSTOMER INPUT
# ============================================================

st.subheader(
    "Enter Customer Details"
)


# ------------------------------------------------------------
# AGE
# ------------------------------------------------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35,
    step=1
)


# ------------------------------------------------------------
# INCOME
# ------------------------------------------------------------

income = st.number_input(
    "Income",
    min_value=0,
    max_value=200000,
    value=50000,
    step=1000
)


# ------------------------------------------------------------
# TOTAL SPENDING
# ------------------------------------------------------------

total_spending = st.number_input(
    "Total Spending (Sum of purchases)",
    min_value=0,
    max_value=5000,
    value=1000,
    step=50
)


# ------------------------------------------------------------
# WEB PURCHASES
# ------------------------------------------------------------

num_web_purchases = st.number_input(
    "Number of Web Purchases",
    min_value=0,
    max_value=100,
    value=10,
    step=1
)


# ------------------------------------------------------------
# STORE PURCHASES
# ------------------------------------------------------------

num_store_purchases = st.number_input(
    "Number of Store Purchases",
    min_value=0,
    max_value=100,
    value=10,
    step=1
)


# ------------------------------------------------------------
# WEB VISITS
# ------------------------------------------------------------

num_web_visits = st.number_input(
    "Number of Web Visits per Month",
    min_value=0,
    max_value=50,
    value=3,
    step=1
)


# ------------------------------------------------------------
# RECENCY
# ------------------------------------------------------------

recency = st.number_input(
    "Recency (Days since last purchase)",
    min_value=0,
    max_value=365,
    value=30,
    step=1
)


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "🎯 Predict Segment",
    use_container_width=True
):

    # ========================================================
    # CREATE INPUT DATAFRAME
    # ========================================================

    input_data = pd.DataFrame({

        "Age": [age],

        "Income": [income],

        "Total_Spending": [total_spending],

        "NumWebPurchases": [num_web_purchases],

        "NumStorePurchases": [num_store_purchases],

        "NumWebVisitsMonth": [num_web_visits],

        "Recency": [recency]

    })


    # ========================================================
    # VERIFY SCALER FEATURES
    # ========================================================

    expected_features = [
        "Age",
        "Income",
        "Total_Spending",
        "NumWebPurchases",
        "NumStorePurchases",
        "NumWebVisitsMonth",
        "Recency"
    ]


    # Check whether the scaler has feature names
    if hasattr(
        scaler,
        "feature_names_in_"
    ):

        scaler_features = list(
            scaler.feature_names_in_
        )

        if scaler_features != expected_features:

            st.error(
                "The scaler was trained using different "
                "features or a different feature order."
            )

            st.write(
                "Expected features:",
                expected_features
            )

            st.write(
                "Scaler features:",
                scaler_features
            )

            st.stop()


    # ========================================================
    # SCALE INPUT
    # ========================================================

    try:

        input_scaled = scaler.transform(
            input_data
        )

    except Exception as e:

        st.error(
            f"Error while scaling input data: {e}"
        )

        st.stop()


    # ========================================================
    # PREDICT CLUSTER
    # ========================================================

    try:

        cluster = int(
            kmeans.predict(
                input_scaled
            )[0]
        )

    except Exception as e:

        st.error(
            f"Error while predicting customer segment: {e}"
        )

        st.stop()


    # ========================================================
    # GET SEGMENT INFORMATION
    # ========================================================

    if cluster not in segment_info:

        st.error(
            f"No segment information available for cluster {cluster}."
        )

        st.stop()


    info = segment_info[cluster]


    # ========================================================
    # DISPLAY CUSTOMER INPUT
    # ========================================================

    st.subheader(
        "📋 Customer Input"
    )


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

    st.subheader(
        "👤 Customer Profile"
    )


    st.write(
        info["description"]
    )


    # ========================================================
    # BUSINESS RECOMMENDATION
    # ========================================================

    st.subheader(
        "💡 Recommended Business Strategy"
    )


    st.info(
        info["recommendation"]
    )


    # ========================================================
    # DISTANCE FROM EACH CLUSTER
    # ========================================================

    try:

        distances = kmeans.transform(
            input_scaled
        )[0]

    except Exception as e:

        st.error(
            f"Unable to calculate cluster distances: {e}"
        )

        st.stop()


    distance_df = pd.DataFrame({

        "Segment": range(
            len(distances)
        ),

        "Distance": distances

    })


    # Sort by closest distance
    distance_df = (
        distance_df
        .sort_values(
            "Distance"
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # DISPLAY DISTANCES
    # ========================================================

    st.subheader(
        "📊 Distance from Customer to Each Segment"
    )


    st.dataframe(
        distance_df,
        use_container_width=True
    )


    # ========================================================
    # CUSTOMER SUMMARY
    # ========================================================

    st.subheader(
        "📌 Customer Summary"
    )


    # --------------------------------------------------------
    # FIRST ROW
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # SECOND ROW
    # --------------------------------------------------------

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


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Customer Segmentation using K-Means Clustering"
)
