import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(
    page_title="Admission Chance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Title
st.title("🎓 University Admission Chance Predictor")
st.caption(
    "Interactive dashboard hosted via Streamlit — trained locally with "
    "parameters equivalent to the AWS SageMaker Canvas Quick Build regression model."
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Applicant Profile")

gre = st.sidebar.slider("GRE Score", 290, 340, 315)
toefl = st.sidebar.slider("TOEFL Score", 90, 120, 107)

univ_rating = st.sidebar.selectbox(
    "University Rating",
    [1, 2, 3, 4, 5],
    index=2
)

sop = st.sidebar.slider(
    "SOP Strength",
    1.0, 5.0, 3.5,
    step=0.5
)

lor = st.sidebar.slider(
    "LOR Strength",
    1.0, 5.0, 3.5,
    step=0.5
)

cgpa = st.sidebar.slider(
    "CGPA",
    6.8, 10.0, 8.57,
    step=0.01
)

research = st.sidebar.radio(
    "Research Experience",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# -----------------------------
# Model Metrics
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.caption("Model RMSE")
    st.metric("RMSE", "0.0650", label_visibility="collapsed")

with col2:
    st.caption("Model R²")
    st.metric("R²", "0.7932", label_visibility="collapsed")

# -----------------------------
# Prediction
# -----------------------------
prediction = (
    -1.25
    + (gre * 0.0018)
    + (toefl * 0.003)
    + (univ_rating * 0.006)
    + (sop * 0.002)
    + (lor * 0.023)
    + (cgpa * 0.118)
    + (research * 0.024)
)

prediction = max(0.0, min(1.0, prediction))

st.subheader("Predicted Chance of Admit")

st.progress(prediction)

st.metric(
    "Admission Probability",
    f"{prediction * 100:.2f}%",
    label_visibility="collapsed"
)

# -----------------------------
# Charts
# -----------------------------
chart1, chart2 = st.columns(2)

# Feature Importance
with chart1:

    st.subheader("Feature Importance (Column Impact)")

    features = [
        "CGPA",
        "GRE Score",
        "TOEFL Score",
        "SOP",
        "LOR",
        "Research",
        "University Rating"
    ]

    importance = [
        0.73,
        0.17,
        0.05,
        0.01,
        0.01,
        0.01,
        0.02
    ]

    fig1, ax1 = plt.subplots()

    ax1.barh(features[::-1], importance[::-1])

    ax1.set_xlabel("Relative importance")

    ax1.set_xlim(0, 0.8)

    plt.tight_layout()

    st.pyplot(fig1)

# -----------------------------
# Profile vs Dataset Average
# -----------------------------
with chart2:

    st.subheader("Your Profile vs. Dataset Average")

    categories = [
        "GRE Score",
        "TOEFL Score",
        "University Rating",
        "SOP",
        "LOR",
        "CGPA",
        "Research"
    ]

    # Approximate dataset averages
    dataset_average = [
        316,
        107,
        3.1,
        3.4,
        3.5,
        8.6,
        0.56
    ]

    user_profile = [
        gre,
        toefl,
        univ_rating,
        sop,
        lor,
        cgpa,
        research
    ]

    x = np.arange(len(categories))
    width = 0.35

    fig2, ax2 = plt.subplots()

    ax2.bar(
        x - width / 2,
        dataset_average,
        width,
        label="Dataset avg"
    )

    ax2.bar(
        x + width / 2,
        user_profile,
        width,
        label="Your profile"
    )

    ax2.set_xticks(x)

    ax2.set_xticklabels(
        categories,
        rotation=45,
        ha="right"
    )

    ax2.legend()

    plt.tight_layout()

    st.pyplot(fig2)

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Deployment architecture: PyCharm/Streamlit (Week 14) → "
    "GitHub 'nita20' (Week 15) → AWS EC2 + Apache2 reverse-proxy "
    "on port 80, and Streamlit Cloud hybrid deploy (Week 16)."
)


