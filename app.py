import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
import json
import requests
from PIL import Image
from streamlit_lottie import st_lottie

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Real Estate Intelligence", page_icon="🏢", layout="wide")
st.markdown("""
    <style>
    .stApp {background-color: #f8f9fa;}
    .main-card {background-color: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);}
    h1, h2, h3 {color: #004b87;}
    .metric-container {background-color: #e3f2fd; padding: 10px; border-radius: 8px; border-left: 5px solid #004b87;}
    </style>
    """, unsafe_allow_html=True)

# 2. LOAD RESOURCES
MODEL_FILE = 'final_model.jb'
METRICS_FILE = 'final_metrics.json'
DATA_FILE = 'dataset.csv'

@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_FILE) if os.path.exists(MODEL_FILE) else None
    
    r2 = 0.0
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f: r2 = json.load(f).get('r2_score', 0.0)
    
    df = pd.read_csv(DATA_FILE) if os.path.exists(DATA_FILE) else None
    return model, r2, df

model, r2_score, df_market = load_resources()

# Lottie Animation
def load_lottieurl(url):
    try: return requests.get(url).json()
    except: return None
lottie_house = load_lottieurl("https://assets7.lottiefiles.com/private_files/lf30_p5tali1o.json")

# 3. SIDEBAR
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=160)
    else:
        st.image("https://upload.wikimedia.org/wikipedia/commons/4/46/TH-Deggendorf-Logo.svg", width=150)
    
    st.markdown("### Intelligent Valuation System")
    st.write("Group Project: **DIT Cham Campus (MLDL 2025)**")
    
    if lottie_house: st_lottie(lottie_house, height=150)
    
    st.markdown("---")
    st.info(f"📊 **Model Accuracy (R²):** {r2_score:.2%}")
    st.caption("Powered by XGBoost & Streamlit")

# 4. MAIN LAYOUT
st.title("Real Estate Intelligence Dashboard")

if not model:
    st.error("🚨 Model not found. Please run 'train_final.py' first.")
    st.stop()

# --- TABS FOR ORGANIZED VIEW ---
tab1, tab2, tab3 = st.tabs(["💰 Valuation Tool", "🧠 Model Insights", "📈 Market Analysis"])

# =======================================================
# TAB 1: PREDICTION TOOL (Interactive)
# =======================================================
with tab1:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("🛠️ Property Configurator")
        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
        year_built = st.number_input("Year Built", 1800, 2025, 2005)
        full_bath = st.slider("Full Bathrooms", 1, 5, 2)
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a: gr_liv_area = st.number_input("Living Area (SqFt)", 500, 5000, 1600)
        with col_b: bsmt_area = st.number_input("Basement Area (SqFt)", 0, 3000, 800)
        garage_area = st.number_input("Garage Area (SqFt)", 0, 2000, 480)

    with c2:
        st.subheader("📍 Location Intelligence")
        hood_map = {
            'College Creek': 'CollgCr', 'Old Town': 'OldTown', 'North Ames': 'NAmes', 
            'Somerset': 'Somerst', 'Edwards': 'Edwards', 'Gilbert': 'Gilbert', 
            'Northridge': 'NoRidge', 'Iowa DOT and Rail': 'IDOTRR'
        }
        selected_display = st.selectbox("Select Neighborhood", list(hood_map.keys()))
        selected_id = hood_map[selected_display]
        
        # Map
        coords = {'College Creek': [42.019, -93.690], 'Old Town': [42.030, -93.615], 'North Ames': [42.042, -93.620], 'Somerset': [42.050, -93.640], 'Edwards': [42.020, -93.660], 'Gilbert': [42.110, -93.650], 'Northridge': [42.050, -93.655], 'Iowa DOT and Rail': [42.020, -93.620]}
        st.map(pd.DataFrame({'lat': [coords[selected_display][0]], 'lon': [coords[selected_display][1]]}), zoom=13)

    st.markdown("---")
    
    if st.button("🚀 Analyze Value", type="primary"):
        # Calc Feature
        total_sf = float(gr_liv_area) + float(bsmt_area)
        
        # DataFrame
        input_df = pd.DataFrame([{
            'OverallQual': overall_qual, 'TotalSF': total_sf, 'GarageArea': garage_area,
            'YearBuilt': year_built, 'FullBath': full_bath, 'Neighborhood': selected_id 
        }])
        
        # Predict
        pred = model.predict(input_df)[0]
        
        st.success(f"### 💎 Estimated Market Value: ${pred:,.2f}")
        
        # --- DYNAMIC PLOTS ---
        viz_c1, viz_c2 = st.columns(2)
        
        with viz_c1:
            # 1. Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = pred,
                title = {'text': "Market Position"},
                gauge = {
                    'axis': {'range': [50000, 600000]},
                    'bar': {'color': "#004b87"},
                    'steps': [{'range': [0, 200000], 'color': "#e9ecef"}, {'range': [200000, 400000], 'color': "#dee2e6"}]
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with viz_c2:
            # 2. Radar Chart (Comparison)
            # Normalize values against max in dataset for plotting
            max_vals = {'OverallQual': 10, 'TotalSF': 5000, 'GarageArea': 1500}
            
            categories = ['Quality', 'Total Size', 'Garage']
            user_vals = [overall_qual/10, total_sf/5000, garage_area/1500]
            avg_vals = [6/10, 2500/5000, 500/1500] # Approximate market averages
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=user_vals, theta=categories, fill='toself', name='Target Property'))
            fig_radar.add_trace(go.Scatterpolar(r=avg_vals, theta=categories, fill='toself', name='Market Avg'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=250, margin=dict(t=30, b=10))
            st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# =======================================================
# TAB 2: MODEL INSIGHTS (Static Report Images)
# =======================================================
with tab2:
    st.header("🧠 Understanding the AI Logic")
    st.write("These visualizations explain how the model makes decisions.")
    
    col_i1, col_i2 = st.columns(2)
    
    with col_i1:
        st.subheader("Feature Importance")
        st.caption("Which factors drive the price the most?")
        if os.path.exists("report_images/3_feature_importance.png"):
            st.image("report_images/3_feature_importance.png", use_column_width=True)
        else:
            st.warning("Run 'generate_plots.py' to see this chart.")
            
    with col_i2:
        st.subheader("Correlation Heatmap")
        st.caption("How features relate to each other.")
        if os.path.exists("report_images/1_correlation_heatmap.png"):
            st.image("report_images/1_correlation_heatmap.png", use_column_width=True)
        else:
            st.warning("Run 'generate_plots.py' to see this chart.")

    st.markdown("---")
    st.subheader("Model Performance Evaluation")
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.image("report_images/6_actual_vs_predicted.png") if os.path.exists("report_images/6_actual_vs_predicted.png") else None
    with col_e2:
        st.image("report_images/7_residual_plot.png") if os.path.exists("report_images/7_residual_plot.png") else None


# =======================================================
# TAB 3: MARKET ANALYSIS (Interactive Data)
# =======================================================
with tab3:
    st.header("📈 Interactive Market Data")
    
    if df_market is not None:
        # Interactive Scatter Plot
        st.subheader("Price vs. Size Analysis")
        fig_scatter = px.scatter(
            df_market, x="GrLivArea", y="SalePrice", 
            color="OverallQual", 
            size="GarageArea",
            hover_data=['YearBuilt', 'Neighborhood'],
            title="Explore the Dataset (Color = Quality, Size = Garage)",
            template="plotly_white"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Interactive Box Plot
        st.subheader("Price Variance by Neighborhood")
        fig_box = px.box(df_market, x="Neighborhood", y="SalePrice", color="Neighborhood")
        st.plotly_chart(fig_box, use_container_width=True)
        
    else:
        st.error("Dataset not loaded.")