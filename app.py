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
    page_title="Configurable Stochastic Systems Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔧"
)

# --- App Styling ---
st.markdown("""
<style>
    .stExpander { border: 1px solid #2c3e50; border-radius: 10px; }
    .stExpander>div[data-baseweb="expander"]>div { background-color: #f0f2f6; }
    .stMetric { border-left: 5px solid #1f77b4; padding-left: 15px; border-radius: 5px; background-color: #fafafa; }
    .uploaded-file-info { padding: 10px; border-radius: 5px; background-color: #e8f0fe; border: 1px solid #1e88e5; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    div[data-testid="stNumberInput"] p { font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE 1: CORE ANALYTICAL ENGINE (Functions)
# ==============================================================================
@st.cache_data
def run_time_series_analysis(_df, digit_col):
    series = _df[digit_col].values
    fft_vals, fft_freq = np.fft.fft(series), np.fft.fftfreq(len(series))
    fig_fft = px.line(x=fft_freq[1:len(fft_freq)//2], y=np.abs(fft_vals)[1:len(fft_vals)//2], labels={'x': 'Frequency', 'y': 'Amplitude'})
    coeffs = pywt.wavedec(series, 'db4', level=4)
    fig_wavelet = make_subplots(rows=len(coeffs), cols=1, subplot_titles=[f'Lvl {i}' for i in range(len(coeffs))])
    for i, c in enumerate(coeffs): fig_wavelet.add_trace(go.Scatter(y=c, mode='lines'), row=i+1, col=1)
    fig_wavelet.update_layout(height=400, showlegend=False)
    rp = RecurrencePlot(threshold='point', percentage=20)
    X_rp = rp.fit_transform(series.reshape(1, -1))
    fig_rp = px.imshow(X_rp[0])
    return {"fft": fig_fft, "wavelet": fig_wavelet, "recurrence": fig_rp}

# ==============================================================================
# MODULE 2: BIFURCATED PREDICTIVE MODELING (Live Analysis)
# ==============================================================================
def create_features_for_model(df, target_cols, lags=3):
    df_feat = df.copy()
    for lag in range(1, lags + 1):
        df_feat[[f'{col}_lag_{lag}' for col in df.columns]] = df.shift(lag)
    df_feat = df_feat[target_cols + [col for col in df_feat.columns if col not in target_cols]]
    return df_feat.dropna()

@st.cache_data
def run_bifurcated_predictive_modeling(_df_full, training_size, forecast_horizon, _min_ranges, _max_ranges):
    # Convert range dicts to lists for clipping
    set_min = [_min_ranges[f'd{i}'] for i in range(1, 6)]
    set_max = [_max_ranges[f'd{i}'] for i in range(1, 6)]
    entity_min, entity_max = _min_ranges['d6'], _max_ranges['d6']

    # Separate data
    _df_set = _df_full.iloc[:, :5]
    _df_entity = _df_full.iloc[:, 5:]

    # Train Model for Set
    features_df_set = create_features_for_model(_df_full.tail(training_size), _df_set.columns)
    model_set = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(features_df_set.drop(columns=_df_set.columns), features_df_set[_df_set.columns])
    
    # Train Model for Entity
    features_df_entity = create_features_for_model(_df_full.tail(training_size), _df_entity.columns)
    model_entity = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(features_df_entity.drop(columns=_df_entity.columns), features_df_entity[_df_entity.columns])

    # Autoregressive Forecasting
    history = _df_full.tail(training_size).copy()
    set_forecasts, entity_forecasts = [], []
    for _ in range(forecast_horizon):
        last_features = create_features_for_model(history, _df_full.columns).iloc[-1:].drop(columns=_df_full.columns)
        pred_set = np.round(model_set.predict(last_features)).astype(int).flatten()
        pred_entity = np.round(model_entity.predict(last_features)).astype(int).flatten()

        # CRITICAL: Clip predictions to respect user-defined ranges
        pred_set_clipped = np.clip(pred_set, set_min, set_max)
        pred_entity_clipped = np.clip(pred_entity, entity_min, entity_max)

        set_forecasts.append(pred_set_clipped)
        entity_forecasts.append(pred_entity_clipped)

        next_full_pred = np.concatenate([pred_set_clipped, pred_entity_clipped])
        history = pd.concat([history, pd.DataFrame([next_full_pred], columns=_df_full.columns, index=[history.index[-1] + pd.Timedelta(days=1)])])

    # Package results
    results = {'set': {}, 'entity': {}}
    index = pd.date_range(start=_df_full.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
    results['set']['forecast_df'] = pd.DataFrame(set_forecasts, columns=_df_set.columns, index=index)
    results['entity']['forecast_df'] = pd.DataFrame(entity_forecasts, columns=_df_entity.columns, index=index)
    results['set']['train_data'] = _df_set.tail(training_size)
    results['entity']['train_data'] = _df_entity.tail(training_size)
    return results

# ==============================================================================
# MODULE 3: BIFURCATED DYNAMICS (Functions)
# ==============================================================================
@st.cache_data
def run_set_clustering_analysis(_df_set, min_cluster_size):
    data_scaled = StandardScaler().fit_transform(_df_set)
    embedding = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, random_state=42).fit_transform(data_scaled)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(embedding)
    plot_df = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])
    plot_df['Cluster'] = clusterer.labels_.astype(str)
    fig = px.scatter(plot_df, x='UMAP_1', y='UMAP_2', color='Cluster', title="Latent Space of 5-Digit Set Dynamics", color_discrete_sequence=px.colors.qualitative.Vivid)
    return fig

@st.cache_data
def run_entity_distribution_analysis(_df_entity, min_val, max_val):
    data = _df_entity.iloc[:, 0].values
    kde = gaussian_kde(data)
    x_range = np.linspace(min_val, max_val, 200)
    pdf = kde(x_range)
    fig = px.area(x=x_range, y=pdf, title=f"Probability Density of Position 6 (Range: {min_val}-{max_val})", labels={'x': 'Digit Value', 'y': 'Density'})
    fig.update_layout(yaxis_visible=False)
    return fig

# ==============================================================================
# MAIN APP INTERFACE
# ==============================================================================
st.title("🔧 Configurable Stochastic Systems Dashboard")
st.markdown("An interactive laboratory analyzing a 6-digit system with user-defined ranges for each position.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ System Configuration")
    with st.expander("Set Independent Digit Ranges", expanded=True):
        cols = st.columns(6)
        min_ranges, max_ranges = {}, {}
        for i in range(6):
            with cols[i]:
                st.markdown(f"**Pos {i+1}**")
                min_ranges[f'd{i+1}'] = st.number_input("Min", value=0, key=f"min_{i}", min_value=-1000, max_value=1000)
                max_ranges[f'd{i+1}'] = st.number_input("Max", value=9 if i<5 else 20, key=f"max_{i}", min_value=-1000, max_value=1000)

    st.header("🔬 Analysis Controls")
    data_loaded = 'data_full' in st.session_state and st.session_state.data_full is not None
    
    with st.form(key="analysis_form"):
        st.header("🧠 Predictive Model Tuning")
        training_size = st.slider("Training History Size (Draws)", 50, 5000, 250, 50, disabled=not data_loaded)
        forecast_horizon = st.slider("Forecast Horizon (Draws)", 5, 50, 10, disabled=not data_loaded)
        st.header("🌐 Cluster Analysis Tuning")
        cluster_sensitivity = st.slider("Cluster Sensitivity (Set)", 5, 50, 15, disabled=not data_loaded)
        run_button = st.form_submit_button("🚀 Run Full System Analysis", type="primary", use_container_width=True, disabled=not data_loaded)

# --- MODULE 0: DATA INGESTION ---
st.header("Module 0: Data Ingestion & Validation")
st.markdown("Upload a 6-column numeric CSV. Data will be validated against the ranges set in the sidebar.")

def create_sample_csv(min_r, max_r):
    data = np.zeros((100, 6), dtype=int)
    for i in range(6):
        data[:, i] = np.random.randint(min_r[f'd{i+1}'], max_r[f'd{i+1}'] + 1, size=100)
    return pd.DataFrame(data).to_csv(index=False, header=False).encode('utf-8')
st.download_button("Download Configured Sample CSV", create_sample_csv(min_ranges, max_ranges), "sample_data.csv")

uploaded_file = st.file_uploader("Upload your historical data (CSV)", type=['csv'])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, header=None, dtype=float, usecols=range(6))
        
        # --- CRITICAL: Validation against user-defined ranges ---
        validation_passed = True
        for i in range(6):
            col_name = f'd{i+1}'
            min_val, max_val = min_ranges[col_name], max_ranges[col_name]
            if not df.iloc[:, i].between(min_val, max_val).all():
                st.error(f"Validation Error in Position {i+1}: Found values outside the configured range [{min_val}, {max_val}].")
                validation_passed = False
                break
        
        if validation_passed:
            df.columns = [f'd{i+1}' for i in range(6)]
            st.session_state.data_full = df
            st.session_state.data_set = df.iloc[:, :5]
            st.session_state.data_entity = df.iloc[:, 5:]
            st.success(f"File validated against configured ranges. {len(df)} draws loaded.")
            st.markdown('<div class="uploaded-file-info"><strong>Data Loaded.</strong> Tune parameters and run analysis.</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing file: {e}. Please ensure it has 6 numeric columns.")
        st.session_state.data_full = None

# --- Analysis Execution & Display ---
if run_button and data_loaded:
    with st.spinner("Executing bifurcated analysis... This may take a moment."):
        # Pass ranges to predictive model
        st.session_state.predictive_results = run_bifurcated_predictive_modeling(st.session_state.data_full, training_size, forecast_horizon, min_ranges, max_ranges)
        # Pass ranges to entity distribution plot
        st.session_state.distribution_results = run_entity_distribution_analysis(st.session_state.data_entity, min_ranges['d6'], max_ranges['d6'])
        st.session_state.clustering_results = run_set_clustering_analysis(st.session_state.data_set, cluster_sensitivity)
    st.toast("Analysis Complete!", icon="✅")
    st.session_state.analysis_run = True

if 'analysis_run' in st.session_state and st.session_state.analysis_run:
    # --- MODULE 1: Analytical Engine ---
    with st.expander("MODULE 1 — Core Analytical Engine: Time Evolution Modeling", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Analysis of the 5-Digit Set")
            selected_digit_set = st.selectbox("Select Digit from Set", options=st.session_state.data_set.columns)
            analysis_set = run_time_series_analysis(st.session_state.data_set, selected_digit_set)
            tab1, tab2, tab3 = st.tabs(["Fourier", "Wavelet", "Recurrence"])
            with tab1: st.plotly_chart(analysis_set['fft'], use_container_width=True)
            with tab2: st.plotly_chart(analysis_set['wavelet'], use_container_width=True)
            with tab3: st.plotly_chart(analysis_set['recurrence'], use_container_width=True)
        with col2:
            st.subheader("Analysis of the Position 6 Entity")
            analysis_entity = run_time_series_analysis(st.session_state.data_entity, 'd6')
            tab4, tab5, tab6 = st.tabs(["Fourier", "Wavelet", "Recurrence"])
            with tab4: st.plotly_chart(analysis_entity['fft'], use_container_width=True)
            with tab5: st.plotly_chart(analysis_entity['wavelet'], use_container_width=True)
            with tab6: st.plotly_chart(analysis_entity['recurrence'], use_container_width=True)
        
        with st.expander("🔬 Methodology & Significance"):
            # This section is preserved as requested
            st.markdown("""
            #### Methodology: Differential Diagnosis via Data Integration
            This module performs a **differential diagnosis** on the time-series properties of the two system components.
            - **Fourier Analysis**: Compares the dominant frequencies. Does the 6th entity have a simpler or more complex cyclical structure than the digits in the set?
            - **Wavelet Decomposition**: Compares transient behavior. Does the 6th entity exhibit volatility at different times or scales compared to the set?
            - **Recurrence Plot**: Compares determinism. Is the 6th entity more chaotic (random speckles) or more predictable (strong diagonals) than the digits within the set?
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
                fig.add_trace(go.Scatter(x=pred_res['entity']['train_data'].index, y=pred_res['entity']['train__data'][col], mode='lines', name=f'Hist {col}', line_color='red'))
                fig.add_trace(go.Scatter(x=pred_res['entity']['forecast_df'].index, y=pred_res['entity']['forecast_df'][col], mode='lines', line=dict(dash='dot'), name=f'Pred {col}', line_color='orange'))
            st.plotly_chart(fig, use_container_width=True)
            
        with st.expander("🔬 Methodology & Significance"):
            # This section is preserved as requested
            st.markdown("""
            #### Methodology: Differential Diagnosis via Data Integration
            This tool performs an automated differential diagnosis for failed experiments. Its power comes from integrating and comparing data from three distinct internal sources:
            1.  **Your Submitted Protocol**: The specific steps you took for the failed run.
            2.  **The System of Record (SOPs)**: The validated, official procedure for this assay from our document control system.
            3.  **Real-time System State**: Data from other hubs, including known issues with specific reagent lots (from the Reagent Hub) and the maintenance/error status of the instrument used (from the Operations Hub).
            
            The AI cross-references these sources to identify deviations and known issues, then ranks them based on their likely impact on the observed failure mode (e.g., high adapter-dimer content).
            
            Two independent **Random Forest** models are trained to produce these forecasts.
            - **Model 1 (The Set):** A multi-output model predicts all 5 digits simultaneously.
            - **Model 2 (The Entity):** A single-output model predicts only the 6th digit.
            
            **Significance:** Both models use the **entire 6-digit history** as input features. This allows us to test for **causality and influence**. If the model for the 6th entity relies heavily on lagged features from the 5-digit set (or vice versa), it suggests a predictive link between the two systems, even if they appear independent. This is a powerful technique for uncovering hidden cross-system dynamics.
            """)

    # --- MODULE 3: Dynamics Analysis ---
    with st.expander("MODULE 3 — System Dynamics & Regime Discovery", expanded=True):
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(st.session_state.clustering_results, use_container_width=True)
        with col2: st.plotly_chart(st.session_state.distribution_results, use_container_width=True)
            
        with st.expander("🔬 Methodology & Significance"):
            # This section is preserved as requested
            st.markdown("""
            #### The Deviation Dashboard: Visualizing the Gap
            The "Deviation Dashboard" is the key visualization. It provides an immediate, at-a-glance summary of where your protocol diverged from the validated state or known good conditions. Red "❗️" cards instantly draw the scientist's attention to the most critical discrepancies, while green "✅" cards confirm which parts of the protocol were likely performed correctly, saving time by ruling out potential causes.

            #### Significance of Results: Accelerating OOS Investigations
            The significance of this tool is a dramatic reduction in the time and resources required to resolve an Out-of-Specification (OOS) or non-conforming laboratory result. Instead of a traditional, unfocused investigation that might involve randomly re-running experiments with one variable changed at a time, this tool provides an **immediate, evidence-based, and prioritized action plan**.

            For this specific failure (low Q30, high adapter content), the analysis points directly to a known problematic reagent lot and two significant deviations in the library preparation protocol. A scientist can now proceed with high confidence by taking the recommended corrective actions, potentially resolving a multi-day investigation in a matter of hours. This directly translates to increased R&D velocity and reduced operational downtime.
            """)
