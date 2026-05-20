# Advanced Professional Streamlit App for Heart Disease Prediction

Create a file named:

```python
app.py
```

Paste this FULL professional code:

```python
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)

# Load Model and Scaler
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ----------------------------
# Custom CSS Styling
# ----------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }

    .title {
        font-size: 42px;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        padding: 10px;
    }

    .subtitle {
        text-align: center;
        color: #BBBBBB;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .card {
        background-color: #161B22;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 0px 15px rgba(255,75,75,0.2);
        margin-bottom: 20px;
    }

    .prediction-high {
        background-color: #ff4b4b;
        color: white;
        padding: 20px;
        border-radius: 12px;
        font-size: 22px;
        text-align: center;
        font-weight: bold;
    }

    .prediction-low {
        background-color: #00C853;
        color: white;
        padding: 20px;
        border-radius: 12px;
        font-size: 22px;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="title">❤️ Heart Disease Prediction System</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Advanced Machine Learning Web Application using XGBoost & Streamlit</div>',
    unsafe_allow_html=True
)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("📌 About Project")
st.sidebar.info(
    """
    This ML application predicts the likelihood of heart disease using clinical parameters.

    ### Technologies Used
    - Python
    - Streamlit
    - XGBoost
    - Scikit-learn
    - Pandas
    - NumPy

    ### Features
    ✅ Real-time Prediction
    ✅ Interactive Dashboard
    ✅ Professional UI
    ✅ Machine Learning Analysis
    """
)

# ----------------------------
# Layout Columns
# ----------------------------
col1, col2 = st.columns(2)

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🧑 Patient Information")

    age = st.slider("Age", 20, 100, 40)

    sex = st.selectbox(
        "Sex",
        [0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [0, 1, 2, 3]
    )

    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)

    chol = st.slider("Cholesterol", 100, 600, 200)

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1]
    )

    restecg = st.selectbox(
        "Rest ECG",
        [0, 1, 2]
    )

    st.markdown('</div>', unsafe_allow_html=True)

with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Medical Parameters")

    thalach = st.slider("Maximum Heart Rate", 60, 220, 150)

    exang = st.selectbox(
        "Exercise Induced Angina",
        [0, 1]
    )

    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0)

    slope = st.selectbox(
        "Slope",
        [0, 1, 2]
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        [0, 1, 2, 3, 4]
    )

    thal = st.selectbox(
        "Thalassemia",
        [0, 1, 2, 3]
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Prediction Section
# ----------------------------
if st.button("🔍 Predict Heart Disease Risk"):

    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    # Scale Input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    # Probability
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")

    if prediction[0] == 1:

        st.markdown(
            f'<div class="prediction-high">⚠️ High Risk of Heart Disease<br><br>Risk Score: {probability:.2%}</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f'<div class="prediction-low">✅ Low Risk of Heart Disease<br><br>Risk Score: {probability:.2%}</div>',
            unsafe_allow_html=True
        )

    # Risk Meter
    st.subheader("📈 Prediction Confidence")

    chart_data = pd.DataFrame({
        'Category': ['Low Risk', 'High Risk'],
        'Probability': [1 - probability, probability]
    })

    fig, ax = plt.subplots(figsize=(5, 3))

    ax.bar(chart_data['Category'], chart_data['Probability'])

    ax.set_ylabel('Probability')

    ax.set_title('Risk Analysis')

    st.pyplot(fig)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")

st.markdown(
    """
    <center>
    Developed with ❤️ using Machine Learning & Streamlit<br>
    Internship Project - CodeAlpha
    </center>
    """,
    unsafe_allow_html=True
)
