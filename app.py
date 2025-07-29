import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# --- Core Scientific & ML Libraries ---
# Make sure to install these: pip install scikit-learn pywavelets pyts umap-learn hdbscan
import pywt
from pyts.image import RecurrencePlot
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy, gaussian_kde
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --- Page Configuration ---
st.set_page_config(
    page_title="Bifurcated Stochastic Systems Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧬"
)

# --- App Styling ---
st.markdown("""
<style>
    .stExpander { border: 1px solid #2c3e50; border-radius: 10px; }
    .stExpander>div[data-baseweb="expander"]>div { background-color: #f0f2f6; }
    .stMetric { border-left: 5px solid #1f77b4; padding-left: 15px; border-radius: 5px; background-color: #fafafa; }
    .uploaded-file-info { padding: 10px; border-radius: 5px; background-color: #e8f0fe; border: 1px solid #1e88e5; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# MODULE 1: CORE ANALYTICAL ENGINE (Functions)
# ==============================================================================
@st.cache_data
def run_time_series_analysis(_df, digit_col):
    """Performs Fourier, Wavelet, and Recurrence analysis on a single digit's time series."""
    series = _df[digit_col].values
    
    # Fourier Analysis
    fft_vals = np.fft.fft(series)
    fft_freq = np.fft.fftfreq(len(series))
    fig_fft = px.line(x=fft_freq[1:len(fft_freq)//2], y=np.abs(fft_vals)[1:len(fft_vals)//2], labels={'x': 'Frequency', 'y': 'Amplitude'})
    fig_fft.update_layout(yaxis_title="Power Spectrum")

    # Wavelet Decomposition
    coeffs = pywt.wavedec(series, 'db4', level=4)
    fig_wavelet = make_subplots(rows=len(coeffs), cols=1, subplot_titles=[f'Lvl {i}' for i in range(len(coeffs))])
    for i, c in enumerate(coeffs):
        fig_wavelet.add_trace(go.Scatter(y=c, mode='lines'), row=i+1, col=1)
    fig_wavelet.update_layout(height=400, showlegend=False)

    # Recurrence Plot
    rp = RecurrencePlot(threshold='point', percentage=20)
    X_rp = rp.fit_transform(series.reshape(1, -1))
    fig_rp = px.imshow(X_rp[0])
    
    return {"fft": fig_fft, "wavelet": fig_wavelet, "recurrence": fig_rp}

# ==============================================================================
# MODULE 2: BIFURCATED PREDICTIVE MODELING (Live Analysis)
# ==============================================================================
def create_features_for_model(df, target_cols, lags=3):
    """Create lagged features from the full dataset for prediction."""
    df_feat = df.copy()
    # Use all 6 columns for feature generation to capture cross-system influence
    for lag in range(1, lags + 1):
        df_feat[[f'{col}_lag_{lag}' for col in df.columns]] = df.shift(lag)
    
    # Select only the target columns to be predicted
    df_feat = df_feat[target_cols + [col for col in df_feat.columns if col not in target_cols]]
    return df_feat.dropna()

@st.cache_data
def run_bifurcated_predictive_modeling(_df_full, _df_set, _df_entity, training_size, forecast_horizon):
    """Trains and forecasts two separate models: one for the 5-digit set and one for the 6th digit."""
    
    results = {}
    
    # --- Model for the 5-Digit Set ---
    train_data_set = _df_set.tail(training_size)
    features_df_set = create_features_for_model(_df_full.tail(training_size), _df_set.columns)
    X_set = features_df_set.drop(columns=_df_set.columns)
    y_set = features_df_set[_df_set.columns]
    
    model_set = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=5)
    model_set.fit(X_set, y_set)
    
    # --- Model for the 6th Digit (Entity) ---
    train_data_entity = _df_entity.tail(training_size)
    features_df_entity = create_features_for_model(_df_full.tail(training_size), _df_entity.columns)
    X_entity = features_df_entity.drop(columns=_df_entity.columns)
    y_entity = features_df_entity[_df_entity.columns]

    model_entity = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=5)
    model_entity.fit(X_entity, y_entity)

    # --- Autoregressive Forecasting for Both Models ---
    history = _df_full.tail(training_size).copy()
    set_forecasts, entity_forecasts = [], []

    for _ in range(forecast_horizon):
        current_features_df = create_features_for_model(history, _df_full.columns)
        last_features = current_features_df.iloc[-1:].drop(columns=_df_full.columns)
        
        pred_set = np.round(model_set.predict(last_features)).astype(int).flatten()
        pred_entity = np.round(model_entity.predict(last_features)).astype(int).flatten()
        
        set_forecasts.append(pred_set)
        entity_forecasts.append(pred_entity)
        
        # Update history with the new full prediction
        next_full_pred = np.concatenate([pred_set, pred_entity])
        next_index = history.index[-1] + pd.Timedelta(days=1)
        history = pd.concat([history, pd.DataFrame([next_full_pred], columns=_df_full.columns, index=[next_index])])

    # --- Package Results ---
    results['set'] = {
        'forecast_df': pd.DataFrame(set_forecasts, columns=_df_set.columns, index=pd.date_range(start=_df_full.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)),
        'train_data': train_data_set
    }
    results['entity'] = {
        'forecast_df': pd.DataFrame(entity_forecasts, columns=_df_entity.columns, index=pd.date_range(start=_df_full.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)),
        'train_data': train_data_entity
    }
    return results

# ==============================================================================
# MODULE 3: BIFURCATED DYNAMICS (Functions)
# ==============================================================================
@st.cache_data
def run_set_clustering_analysis(_df_set, min_cluster_size):
    """Performs UMAP reduction and HDBSCAN clustering on the 5-digit set."""
    data_scaled = StandardScaler().fit_transform(_df_set)
    reducer = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(data_scaled)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
    clusterer.fit(embedding)
    plot_df = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])
    plot_df['Cluster'] = clusterer.labels_.astype(str)
    fig = px.scatter(plot_df, x='UMAP_1', y='UMAP_2', color='Cluster', title="Latent Space of 5-Digit Set Dynamics", color_discrete_sequence=px.colors.qualitative.Vivid)
    fig.update_layout(legend_title_text='Behavioral Cluster')
    return fig

@st.cache_data
def run_entity_distribution_analysis(_df_entity):
    """Visualizes the probability distribution of the individual 6th digit."""
    data = _df_entity.iloc[:, 0].values
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min() - 1, data.max() + 1, 100)
    pdf = kde(x_range)
    fig = px.area(x=x_range, y=pdf, title="Probability Density of Position 6 Entity", labels={'x': 'Digit Value', 'y': 'Density'})
    fig.update_layout(yaxis_visible=False)
    return fig

# ==============================================================================
# MAIN APP INTERFACE
# ==============================================================================
st.title("🧬 Bifurcated Stochastic Systems Analytics")
st.markdown("An interactive laboratory analyzing a 6-digit system as two distinct entities: a **5-digit set** and a **final individual digit**.")

# --- Sidebar ---
with st.sidebar:
    st.header("🔬 Analysis Controls")
    data_loaded = 'data_set' in st.session_state and st.session_state.data_set is not None
    
    with st.form(key="analysis_form"):
        st.header("🧠 Predictive Model Tuning")
        training_size = st.slider("Training History Size (Draws)", min_value=50, max_value=5000, value=250, step=50, disabled=not data_loaded)
        forecast_horizon = st.slider("Forecast Horizon (Draws)", 5, 50, 10, disabled=not data_loaded)
        st.header("🌐 Cluster Analysis Tuning")
        cluster_sensitivity = st.slider("Cluster Sensitivity (for 5-Digit Set)", 5, 50, 15, disabled=not data_loaded)
        run_button = st.form_submit_button("🚀 Run Full System Analysis", type="primary", use_container_width=True, disabled=not data_loaded)

# --- MODULE 0: DATA INGESTION ---
st.header("Module 0: Data Ingestion & Validation")
st.download_button("Download Sample CSV", pd.DataFrame(np.random.randint(0, 10, size=(100, 6))).to_csv(index=False, header=False).encode('utf-8'), "sample_data.csv")
uploaded_file = st.file_uploader("Upload your historical data (CSV, 6 numeric columns)", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, header=None, dtype=int, usecols=range(6))
        df.columns = [f'd{i+1}' for i in range(6)]
        
        st.session_state.data_full = df
        st.session_state.data_set = df[['d1', 'd2', 'd3', 'd4', 'd5']]
        st.session_state.data_entity = df[['d6']]
        
        st.success(f"File validated. {len(df)} draws loaded. System is now bifurcated.")
        st.markdown('<div class="uploaded-file-info"><strong>Data Loaded.</strong> Tune parameters and run analysis.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing file: {e}. Please ensure it has 6 numeric columns.")
        st.session_state.data_set = None

# --- Analysis Execution & Display ---
if run_button and 'data_set' in st.session_state:
    with st.spinner("Executing bifurcated analysis... This may take a moment."):
        # Time Series Analysis (pre-run for both)
        st.session_state.analysis_set = run_time_series_analysis(st.session_state.data_set, 'd1')
        st.session_state.analysis_entity = run_time_series_analysis(st.session_state.data_entity, 'd6')
        # Predictive Modeling
        st.session_state.predictive_results = run_bifurcated_predictive_modeling(st.session_state.data_full, st.session_state.data_set, st.session_state.data_entity, training_size, forecast_horizon)
        # Dynamics Analysis
        st.session_state.clustering_results = run_set_clustering_analysis(st.session_state.data_set, cluster_sensitivity)
        st.session_state.distribution_results = run_entity_distribution_analysis(st.session_state.data_entity)
    st.toast("Analysis Complete!", icon="✅")
    st.session_state.analysis_run = True

if 'analysis_run' in st.session_state and st.session_state.analysis_run:
    # --- MODULE 1: Analytical Engine ---
    with st.expander("MODULE 1 — Core Analytical Engine: Time Evolution Modeling", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Analysis of the 5-Digit Set")
            selected_digit_set = st.selectbox("Select Digit from Set for Analysis", options=st.session_state.data_set.columns)
            if selected_digit_set != st.session_state.get('_last_digit_set', None):
                st.session_state.analysis_set = run_time_series_analysis(st.session_state.data_set, selected_digit_set)
                st.session_state._last_digit_set = selected_digit_set
            tab1, tab2, tab3 = st.tabs(["Fourier", "Wavelet", "Recurrence"])
            with tab1: st.plotly_chart(st.session_state.analysis_set['fft'], use_container_width=True)
            with tab2: st.plotly_chart(st.session_state.analysis_set['wavelet'], use_container_width=True)
            with tab3: st.plotly_chart(st.session_state.analysis_set['recurrence'], use_container_width=True)
        with col2:
            st.subheader("Analysis of the Position 6 Entity")
            res = st.session_state.analysis_entity
            tab4, tab5, tab6 = st.tabs(["Fourier", "Wavelet", "Recurrence"])
            with tab4: st.plotly_chart(res['fft'], use_container_width=True)
            with tab5: st.plotly_chart(res['wavelet'], use_container_width=True)
            with tab6: st.plotly_chart(res['recurrence'], use_container_width=True)
        
        with st.expander("🔬 Methodology & Significance"):
            st.markdown("""
            This module performs a **differential diagnosis** on the time-series properties of the two system components.
            - **Fourier Analysis**: Compares the dominant frequencies. Does the 6th digit have a simpler or more complex cyclical structure than the digits in the set?
            - **Wavelet Decomposition**: Compares transient behavior. Does the 6th digit exhibit volatility at different times or scales compared to the set?
            - **Recurrence Plot**: Compares determinism. Is the 6th digit more chaotic (random speckles) or more predictable (strong diagonals) than the digits within the set?
            """)

    # --- MODULE 2: Predictive Modeling ---
    with st.expander("MODULE 2 — Predictive Modeling & Stability Convergence", expanded=True):
        pred_res = st.session_state.predictive_results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Forecast for 5-Digit Set")
            fig = go.Figure()
            for col in pred_res['set']['train_data'].columns:
                fig.add_trace(go.Scatter(x=pred_res['set']['train_data'].index, y=pred_res['set']['train_data'][col], mode='lines', name=f'Hist {col}'))
                fig.add_trace(go.Scatter(x=pred_res['set']['forecast_df'].index, y=pred_res['set']['forecast_df'][col], mode='lines', line=dict(dash='dot'), name=f'Pred {col}'))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Forecast for Position 6 Entity")
            fig = go.Figure()
            for col in pred_res['entity']['train_data'].columns:
                fig.add_trace(go.Scatter(x=pred_res['entity']['train_data'].index, y=pred_res['entity']['train_data'][col], mode='lines', name=f'Hist {col}', line_color='red'))
                fig.add_trace(go.Scatter(x=pred_res['entity']['forecast_df'].index, y=pred_res['entity']['forecast_df'][col], mode='lines', line=dict(dash='dot'), name=f'Pred {col}', line_color='orange'))
            st.plotly_chart(fig, use_container_width=True)
            
        with st.expander("🔬 Methodology & Significance"):
            st.markdown("""
            Two independent **Random Forest** models are trained to produce these forecasts.
            - **Model 1 (The Set):** A multi-output model predicts all 5 digits simultaneously.
            - **Model 2 (The Entity):** A single-output model predicts only the 6th digit.
            
            **Significance:** Both models use the **entire 6-digit history** as input features. This allows us to test for **causality and influence**. If the model for the 6th digit relies heavily on lagged features from the 5-digit set (or vice versa), it suggests a predictive link between the two systems, even if they appear independent. This is a powerful technique for uncovering hidden cross-system dynamics.
            """)

    # --- MODULE 3: Dynamics Analysis ---
    with st.expander("MODULE 3 — System Dynamics & Regime Discovery", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(st.session_state.clustering_results, use_container_width=True)
        with col2:
            st.plotly_chart(st.session_state.distribution_results, use_container_width=True)
            
        with st.expander("🔬 Methodology & Significance"):
            st.markdown("""
            This module visualizes the *state space* of each system component.
            - **Left (The Set):** We use **UMAP** to reduce the 5D space of the set to 2D, then **HDBSCAN** to find dense clusters. Each cluster represents a 'family' or 'regime' of draw sets with similar characteristics. Tracking the system's movement between these clusters reveals high-level dynamic shifts.
            - **Right (The Entity):** For the 1D entity, we use **Kernel Density Estimation (KDE)** to plot its probability distribution. This shows which values the 6th digit is most likely to take, revealing any biases that are not apparent from simple histograms.
            
            **Significance:** Comparing the two plots allows for a deep understanding of the system's overall behavior. For example, we could filter the left plot to a single cluster (a specific 'regime') and see if the probability distribution on the right changes. If it does, it implies that the behavior of the 6th digit is **conditionally dependent** on the current regime of the 5-digit set.
            """)
