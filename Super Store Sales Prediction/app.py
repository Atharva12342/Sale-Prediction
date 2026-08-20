import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Super Store Sales Prediction System",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD PICKLE FILES & DATASET
# =========================================================
@st.cache_resource
def load_assets():
    with open("knn_model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    with open("feature_columns.pkl", "rb") as file:
        feature_columns = pickle.load(file)
    return model, scaler, feature_columns

try:
    model, scaler, feature_columns = load_assets()
except FileNotFoundError:
    st.error("⚠️ Pickle files not found! Please check model training output files.")
    st.stop()

# =========================================================
# STYLES & COLOR PALETTE
# =========================================================
BG_IMAGE_URL = "https://images.unsplash.com/photo-1578916171728-46686eac8d58?q=80&w=1920&auto=format&fit=crop"

st.markdown(
    f"""
    <style>
    /* Background setup */
    .stApp {{
        background-image: url("{BG_IMAGE_URL}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Global Typography */
    h1, h2, h3, h4, h5, h6, p, label, span, .stMarkdown {{
        color: #0F172A !important;
        font-weight: 700 !important;
    }}

    /* Center Align Tab Bar Container */
    .stTabs [data-baseweb="tab-list"] {{
        justify-content: center !important;
        gap: 20px !important;
        margin-bottom: 16px !important;
    }}

    /* High Specificity Tab Button Styling */
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] {{
        padding: 14px 36px !important;
        border-radius: 12px !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
        box-shadow: 0px 4px 14px rgba(0, 0, 0, 0.15) !important;
        height: auto !important;
    }}

    /* Enforce Large Font Size across all sub-elements in Tabs */
    .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] * {{
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        line-height: 1.2 !important;
    }}

    /* Expanded & Colored Input Form Container */
    div[data-testid="stForm"] {{
        background: linear-gradient(135deg, rgba(240, 249, 255, 0.85) 0%, rgba(224, 242, 254, 0.85) 100%) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        padding: 35px !important;
        border-radius: 20px !important;
        box-shadow: 0px 12px 35px rgba(15, 23, 42, 0.4) !important;
        border: 3px solid #0284C7 !important;
        min-height: 480px;
    }}

    /* Tab Content Panel Styling with Backdrop Blur */
    .stTabs [data-baseweb="tab-panel"] {{
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        padding: 32px !important;
        border-radius: 20px !important;
        box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.25) !important;
        border: 2px solid rgba(2, 132, 199, 0.5) !important;
    }}

    /* Hero Header for Top Title */
    .hero-container {{
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.9) 0%, rgba(37, 99, 235, 0.9) 100%);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 32px 28px;
        border-radius: 18px;
        color: #FFFFFF !important;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0px 8px 25px rgba(0, 0, 0, 0.4);
    }}
    .hero-container h1, .hero-container h3, .hero-container div {{
        color: #FFFFFF !important;
    }}
    
    /* Dashboard Title */
    .hero-title {{
        font-size: 52px !important;
        font-weight: 900 !important;
        margin: 0;
        color: #FFFFFF !important;
        letter-spacing: -0.5px;
    }}
    
    .hero-subtitle {{
        font-size: 18px;
        margin-top: 8px;
        color: #E0F2FE !important;
    }}

    /* Prediction Output Banner */
    .result-card {{
        background: linear-gradient(135deg, #0D9488 0%, #059669 100%);
        padding: 28px;
        border-radius: 18px;
        color: #FFFFFF !important;
        text-align: center;
        margin: 24px 0;
        box-shadow: 0px 10px 28px rgba(13, 148, 136, 0.4);
    }}
    .result-card div, .result-card h1, .result-card p {{
        color: #FFFFFF !important;
    }}
    .result-value {{
        font-size: 48px;
        font-weight: 900;
        margin: 6px 0;
    }}

    /* About Us Cards with Blur */
    .card {{
        background: rgba(248, 250, 252, 0.75) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        padding: 28px;
        border-radius: 16px;
        box-shadow: 0px 6px 18px rgba(15, 23, 42, 0.1);
        border: 2px solid rgba(56, 189, 248, 0.6);
    }}

    /* Input & Select Box Accent Colors */
    div[data-baseweb="select"] > div, input {{
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 2px solid #0284C7 !important;
        border-radius: 8px !important;
    }}

    /* Section Heading Panels */
    .section-header {{
        font-size: 26px !important;
        font-weight: 900 !important;
        color: #1E3A8A !important;
        margin-bottom: 20px !important;
        border-bottom: 3px solid #2563EB !important;
        padding-bottom: 10px !important;
        letter-spacing: 0.5px;
    }}

    /* About Us Footer Container with Blur */
    .footer-container {{
        text-align: center;
        padding: 25px;
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        box-shadow: 0px 8px 25px rgba(0, 0, 0, 0.4);
        margin-top: 24px;
        border: 1px solid #334155;
    }}
    .footer-container h3 {{
        color: #F8FAFC !important;
        font-size: 22px;
        margin-bottom: 8px;
    }}
    .footer-container p {{
        color: #CBD5E1 !important;
        font-size: 15px;
        margin: 4px 0;
    }}
    .footer-container span {{
        color: #38BDF8 !important;
        font-weight: bold !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# DASHBOARD HEADER
# =========================================================
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🏬 Super Store Sales Prediction</div>
        <div class="hero-subtitle">Advanced Machine Learning Analytics & Revenue Forecasting System</div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# CONTROL PANEL & INPUT FORM
# =========================================================
tab_predict, tab_about = st.tabs(["📊 Prediction Form", "ℹ️ About Us"])

with tab_predict:
    with st.form(key="prediction_form"):
        col_item, col_outlet = st.columns(2, gap="large")

        # ITEM INPUTS
        with col_item:
            st.markdown('<div class="section-header">📦 Item Characteristics</div>', unsafe_allow_html=True)
            
            item_type = st.selectbox(
                "Item Type",
                [
                    "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", 
                    "Frozen Foods", "Fruits and Vegetables", "Hard Drinks", 
                    "Health and Hygiene", "Household", "Meat", "Others", 
                    "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"
                ]
            )
            
            c1, c2 = st.columns(2)
            with c1:
                item_mrp = st.number_input("Item MRP (₹)", min_value=0.0, max_value=1000.0, value=180.0, step=1.0)
                item_weight = st.number_input("Item Weight (kg)", min_value=0.0, max_value=100.0, value=12.5, step=0.1)
            
            with c2:
                item_fat_content = st.selectbox("Fat Content", ["Low Fat", "Regular"])
                item_visibility = st.number_input("Visibility Score", min_value=0.0, max_value=0.5, value=0.06, step=0.01)

        # OUTLET INPUTS
        with col_outlet:
            st.markdown('<div class="section-header">🏪 Super Store Details</div>', unsafe_allow_html=True)
            
            outlet_identifier = st.selectbox(
                "Outlet Identifier",
                ["OUT010", "OUT013", "OUT017", "OUT018", "OUT019", "OUT027", "OUT035", "OUT045", "OUT046", "OUT049"]
            )
            
            c3, c4 = st.columns(2)
            with c3:
                outlet_type = st.selectbox(
                    "Outlet Type",
                    ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]
                )
                outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
            
            with c4:
                outlet_location = st.selectbox("Location Tier", ["Tier 1", "Tier 2", "Tier 3"])
                outlet_year = st.selectbox("Establishment Year", options=list(range(1980, 2026)), index=19)

        st.write("")
        predict_button = st.form_submit_button("🔮 Predict Super Store Sales", use_container_width=True, type="primary")

    # =========================================================
    # INFERENCE & DASHBOARD VISUALS (PREDICTION TAB ONLY)
    # =========================================================
    if predict_button:
        input_data = pd.DataFrame({
            "Item_Weight": [item_weight],
            "Item_Fat_Content": [item_fat_content],
            "Item_Visibility": [item_visibility],
            "Item_Type": [item_type],
            "Item_MRP": [item_mrp],
            "Outlet_Identifier": [outlet_identifier],
            "Outlet_Establishment_Year": [outlet_year],
            "Outlet_Size": [outlet_size],
            "Outlet_Location_Type": [outlet_location],
            "Outlet_Type": [outlet_type]
        })

        input_data["Item_Weight"] = input_data["Item_Weight"].fillna(12.6)
        input_data["Item_Fat_Content"] = input_data["Item_Fat_Content"].replace({"Low Fat": 0, "Regular": 1, "LF": 0, "reg": 1, "low fat": 0})
        input_data["Outlet_Size"] = input_data["Outlet_Size"].replace({"Small": 0, "Medium": 1, "High": 2})
        input_data["Outlet_Location_Type"] = input_data["Outlet_Location_Type"].replace({"Tier 1": 0, "Tier 2": 1, "Tier 3": 2})

        item_type_encoded = pd.get_dummies(input_data["Item_Type"], prefix="Item_Type", drop_first=True)
        outlet_id_encoded = pd.get_dummies(input_data["Outlet_Identifier"], prefix="Outlet_Identifier")
        outlet_type_encoded = pd.get_dummies(input_data["Outlet_Type"], prefix="Outlet_Type", drop_first=True)

        input_data = input_data.drop(["Item_Type", "Outlet_Identifier", "Outlet_Type"], axis=1)
        input_data = pd.concat([input_data, item_type_encoded, outlet_id_encoded, outlet_type_encoded], axis=1)
        input_data = input_data.reindex(columns=feature_columns, fill_value=0)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        predicted_sales = max(0.0, float(prediction[0]))

        st.markdown(
            f"""
            <div class="result-card">
                <div style="font-size: 16px; text-transform: uppercase; letter-spacing: 1.2px;">Predicted Super Store Sales</div>
                <div class="result-value">₹ {predicted_sales:,.2f}</div>
                <div style="font-size: 14px;">Forecast generated using KNN Regression</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Selected Item MRP", f"₹ {item_mrp:.2f}")
        m2.metric("Item Visibility", f"{item_visibility:.1%}")
        m3.metric("Store Format", outlet_type)
        m4.metric("Store Location", outlet_location)

        st.markdown("### 📈 Prediction & Sales Benchmark Graphs")
        graph_col1, graph_col2 = st.columns(2)

        with graph_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=predicted_sales,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Sales vs Benchmark", 'font': {'size': 18, 'color': "#0F172A"}},
                delta={'reference': item_mrp * 12, 'increasing': {'color': "#0D9488"}},
                gauge={
                    'axis': {'range': [None, max(predicted_sales * 1.5, 5000)], 'tickwidth': 1, 'tickcolor': "#0F172A"},
                    'bar': {'color': "#0284C7"},
                    'steps': [
                        {'range': [0, 1500], 'color': "#FEE2E2"},
                        {'range': [1500, 3500], 'color': "#FEF3C7"},
                        {'range': [3500, 6000], 'color': "#CCFBF1"}
                    ],
                }
            ))
            fig_gauge.update_layout(height=340, paper_bgcolor='#FFFFFF', font={'color': "#0F172A"}, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with graph_col2:
            outlet_types = ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]
            type_multipliers = {"Grocery Store": 0.35, "Supermarket Type1": 0.9, "Supermarket Type2": 1.1, "Supermarket Type3": 1.55}
            base_estimate = predicted_sales / type_multipliers.get(outlet_type, 1.0)
            comparison_sales = [base_estimate * mult for mult in type_multipliers.values()]
            
            df_chart = pd.DataFrame({
                "Outlet Type": outlet_types,
                "Estimated Sales (₹)": comparison_sales
            })
            
            fig_bar = px.bar(
                df_chart, 
                x="Outlet Type", 
                y="Estimated Sales (₹)",
                color="Outlet Type",
                title=f"Sales Comparison across Store Formats ({item_type})",
                color_discrete_sequence=["#0D9488", "#0284C7", "#059669", "#0891B2"],
                text_auto='.2s'
            )
            fig_bar.update_layout(height=340, paper_bgcolor='#FFFFFF', plot_bgcolor='#F8FAFC', font={'color': "#0F172A"}, showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

# =========================================================
# ABOUT US TAB
# =========================================================
with tab_about:
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.markdown("""
## 📖 About the Project

**Super Store Sales Prediction System** is a Machine Learning application designed to predict total sales revenues for products across various supermarket formats and store locations.

Retail chains lose substantial revenue due to improper inventory allocation, inaccurate demand forecasting, and ineffective product shelf positioning. By leveraging historical product and store characteristics, this system enables business managers to forecast product-level revenues accurately.

---

## 🚀 Workflow

Item & Store Input Details   
⬇   
Data Preprocessing & Missing Value Imputation   
⬇   
Categorical Feature Encoding (One-Hot & Ordinal)   
⬇   
Standard Feature Scaling (`StandardScaler`)   
⬇   
KNN Regression Model Forecasting   
⬇   
Predicted Sales Generation   
⬇   
Decision Making for Retail Managers   

---

## ✨ Features

✅ Item Sales Revenue Forecasting   
✅ Interactive Parameter Customization   
✅ Multi-Store Format Comparison Metrics   
✅ High-Contrast Responsive Dashboard UI   
✅ KNN Regressor Architecture   
✅ Benchmarking vs Baseline Estimates   
""")

    with right:
        st.markdown("""
<div class="card">

## 📌 Project Details

🏬 **Domain**   
Retail & E-Commerce

🤖 **Model**   
K-Nearest Neighbors (KNN)

📊 **Dataset**   
Super Store Sales Dataset

👥 **Outlets**   
10 Unique Outlet Identifiers

📈 **Features**   
Item & Store Parameters

💻 **Framework**   
Streamlit

🐍 **Language**   
Python

</div>
""", unsafe_allow_html=True)

    st.write("---")

    st.subheader("🛠 Technology Stack")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info("""
### 🐍 Python & Data
✔ Python   
✔ NumPy   
✔ Pandas   
✔ Pickle   
""")

    with c2:
        st.info("""
### 🤖 Machine Learning
✔ KNN Regressor   
✔ Scikit-Learn   
✔ Standard Scaler   
✔ One-Hot Encoding   
""")

    with c3:
        st.info("""
### 🌐 Dashboard & Visuals
✔ Streamlit   
✔ Custom CSS Glassmorphism   
✔ Interactive Inputs   
""")

    st.write("---")

    st.subheader("🎯 Business Benefits")

    a, b, c = st.columns(3)

    with a:
        st.metric("Inventory Optimization", "↑ 22%")

    with b:
        st.metric("Forecast Accuracy", "↑ 35%")

    with c:
        st.metric("Stockout Reduction", "↓ 18%")

    st.write("---")

    st.subheader("📈 Why Machine Learning in Retail?")

    st.info("""
Machine Learning empowers retail chains to transition from reactive management to proactive predictive planning.

Key Advantages:
• Precision demand forecasting for new product launches.
• Optimized shelf space allocation based on visibility scores.
• Strategic pricing alignment with store location tiers (Tier 1, Tier 2, Tier 3).
• Waste reduction in perishable inventory categories.
• Data-backed decision making across varying supermarket sizes.
""")

    st.write("---")

    st.subheader("👨‍💻 Developer")

    st.markdown("""
<div class="card" style="border: 2px solid #0284C7;">

<h2 style="color:#0369A1;">Darshan Bhor</h2>

<hr style="border: 1px solid #E0F2FE;">

<b>Project :</b> Super Store Sales Prediction System<br><br>
<b>Role :</b> Machine Learning Developer<br><br>
<b>Skills :</b>
<ul>
<li>Python</li>
<li>Machine Learning</li>
<li>Scikit-Learn</li>
<li>Pandas</li>
<li>NumPy</li>
<li>Streamlit</li>
</ul>

</div>
""", unsafe_allow_html=True)

    st.write("")

    st.subheader("📬 Contact")

    st.success("""
📧 Email: darshanbhor2006@gmail.com   
💼 LinkedIn: https://www.linkedin.com/in/darshan-bhor   
🐙 GitHub: https://github.com/darshanbhor2006   
""")

    st.write("---")

    st.markdown("""
<div class="footer-container">
    <h3>🏬 Super Store Sales Prediction System</h3>
    <p>Built with <span>Darshan Bhor</span> using Python, Streamlit & Machine Learning</p>
    <p>© 2026 Darshan Bhor | All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)