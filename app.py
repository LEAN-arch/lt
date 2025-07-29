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
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --- Page Configuration ---
st.set_page_config(
    page_title="Definitive Systems Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="👑"
)

# --- App Styling ---
st.markdown("""
<style>
    .stExpander { border: 1px solid #2c3e50; border-radius: 10px; }
    .stExpander>div[data-baseweb="expander"]>div { background-color: #f0f2f6; }
    .stMetric { border-left: 5px solid #1f77b4; padding-left: 15px; border-radius: 5px; background-color: #fafafa; }
    div[data-testid="stNumberInput"] p { font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# MODULE 1: CORE ANALYTICAL ENGINE (Functions)
# ==============================================================================
@st.cache_data
def run_time_series_analysis(_df, column):
    series = _df[column].values
    fft_vals, fft_freq = np.fft.fft(series), np.fft.fftfreq(len(series))
    fig_fft = px.line(x=fft_freq[1:len(fft_freq)//2], y=np.abs(fft_vals)[1:len(fft_vals)//2], title=f"Fourier: {column}", labels={'x': 'Freq', 'y': 'Amp'})
    coeffs = pywt.wavedec(series, 'db4', level=4)
    fig_wavelet = make_subplots(rows=len(coeffs), cols=1, subplot_titles=[f'Lvl {i}' for i in range(len(coeffs))])
    for i, c in enumerate(coeffs): fig_wavelet.add_trace(go.Scatter(y=c, mode='lines'), row=i+1, col=1)
    fig_wavelet.update_layout(height=400, showlegend=False, title=f"Wavelet: {column}")
    rp = RecurrencePlot(threshold='point', percentage=20)
    X_rp = rp.fit_transform(series.reshape(1, -1))
    fig_rp = px.imshow(X_rp[0], title=f'Recurrence: {column}')
    return {"fft": fig_fft, "wavelet": fig_wavelet, "recurrence": fig_rp}

# ==============================================================================
# MODULE 2: ADAPTIVE PREDICTIVE MODELING (Functions with Maximum Rigor)
# ==============================================================================
def create_features_for_model(df, lags=5):
    df_feat = df.copy()
    for lag in range(1, lags + 1):
        df_feat[[f'{col}_lag_{lag}' for col in df.columns]] = df.shift(lag)
    return df_feat.dropna()

def train_and_forecast(model, history, forecast_horizon, min_ranges, max_ranges):
    forecasts, uncertainties = [], []
    for _ in range(forecast_horizon):
        last_features = create_features_for_model(history).iloc[-1:].drop(columns=history.columns)
        tree_preds = np.array([tree.predict(last_features) for tree in model.estimators_])
        mean_pred = np.clip(tree_preds.mean(axis=0), min_ranges, max_ranges).flatten()
        std_pred = tree_preds.std(axis=0).flatten()
        forecasts.append(np.round(mean_pred).astype(int))
        uncertainties.append(std_pred)
        history = pd.concat([history, pd.DataFrame([np.round(mean_pred)], columns=history.columns, index=[history.index[-1] + pd.Timedelta(days=1)])])
    return forecasts, uncertainties

@st.cache_data
def run_expert_predictive_modeling(_df_full, training_size, forecast_horizon, _min_ranges, _max_ranges, _is_bifurcated):
    
    def perform_validation_and_forecast(target_df, full_history, min_r, max_r):
        errors, top3_hits = [], []
        val_window = min(50, len(full_history) - training_size - 1)
        if val_window <= 0: return {'forecast_df': pd.DataFrame(), 'uncertainty_df': pd.DataFrame(), 'metrics': {'oos_loss': -1, 'forecast_stability': -1, 'top_n_accuracy': -1}}

        for i in range(val_window):
            train_end = len(full_history) - val_window + i
            train_df = full_history.iloc[train_end - training_size : train_end]
            features_df = create_features_for_model(train_df)
            X_train = features_df.drop(columns=full_history.columns); y_train = features_df[target_df.columns]
            model = RandomForestRegressor(n_estimators=30, random_state=42).fit(X_train, y_train)
            last_features = create_features_for_model(train_df).iloc[-1:].drop(columns=full_history.columns)
            tree_preds = np.array([tree.predict(last_features) for tree in model.estimators_]); pred = tree_preds.mean(axis=0)
            true = full_history[target_df.columns].iloc[train_end].values
            errors.append(mean_squared_error(true.flatten(), pred.flatten()))
            if target_df.shape[1] == 1:
                top3_preds = np.round(np.percentile(tree_preds, [25, 50, 75], axis=0)).astype(int).flatten()
                top3_hits.append(true[0] in top3_preds)
        
        final_features = create_features_for_model(full_history.tail(training_size))
        X_final = final_features.drop(columns=full_history.columns); y_final = final_features[target_df.columns]
        final_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_final, y_final)
        forecasts, uncertainties = train_and_forecast(final_model, full_history.tail(training_size), forecast_horizon, min_r, max_r)
        index = pd.date_range(start=full_history.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
        return {
            'forecast_df': pd.DataFrame(forecasts, columns=target_df.columns, index=index),
            'uncertainty_df': pd.DataFrame(uncertainties, columns=target_df.columns, index=index),
            'metrics': {'oos_loss': np.mean(errors), 'forecast_stability': np.std(errors), 'top_n_accuracy': np.mean(top3_hits) if top3_hits else None}
        }

    results = {}
    if _is_bifurcated:
        df_set, df_entity = _df_full.iloc[:, :5], _df_full.iloc[:, 5:]
        set_min = list(_min_ranges.values())[:5]; set_max = list(_max_ranges.values())[:5]
        entity_min = list(_min_ranges.values())[5:]; entity_max = list(_max_ranges.values())[5:]
        results['set'] = perform_validation_and_forecast(df_set, _df_full, set_min, set_max)
        results['entity'] = perform_validation_and_forecast(df_entity, _df_full, entity_min, entity_max)
    else:
        all_min = list(_min_ranges.values()); all_max = list(_max_ranges.values())
        results['unified'] = perform_validation_and_forecast(_df_full, _df_full, all_min, all_max)
    return results

# ==============================================================================
# MODULE 3: DYNAMICS & CLUSTERING (Functions)
# ==============================================================================
@st.cache_data
def run_clustering_analysis(_df, min_cluster_size):
    if len(_df) < min_cluster_size or _df.shape[1] < 2: return go.Figure(), None
    data_scaled = StandardScaler().fit_transform(_df); embedding = umap.UMAP(n_neighbors=min(15, len(_df)-1), n_components=2, random_state=42).fit_transform(data_scaled)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(embedding)
    plot_df = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'], index=_df.index); plot_df['Cluster'] = clusterer.labels_.astype(str)
    fig = px.scatter(plot_df, x='UMAP_1', y='UMAP_2', color='Cluster', title=f"Latent Space of {len(_df.columns)}-D System", color_discrete_sequence=px.colors.qualitative.Vivid)
    return fig, plot_df

@st.cache_data
def run_entity_distribution_analysis(_df_entity, min_val, max_val, title_suffix=""):
    data = _df_entity.iloc[:, 0].values
    if len(data) < 2: return go.Figure()
    kde = gaussian_kde(data); x_range = np.linspace(min_val, max_val, 200)
    fig = px.area(x=x_range, y=kde(x_range), title=f"Prob. Density {title_suffix}", labels={'x': 'Value', 'y': 'Density'}); fig.update_layout(yaxis_visible=False)
    return fig


# ==============================================================================
# MAIN APP INTERFACE
# ==============================================================================
st.title("👑 Definitive Adaptive Systems Dashboard")
st.markdown("An intelligent analysis engine that ingests numerical time-series data, automatically deduces its properties, and configures a rigorous analysis pipeline.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ System State-Space")
    data_loaded = 'data_full' in st.session_state and st.session_state.data_full is not None
    
    if not data_loaded: st.info("Upload data to automatically detect and configure column ranges.")
    else: st.success("Ranges Detected & Configurable")
    
    num_cols_for_ui = st.session_state.get('num_columns', 6)
    cols = st.columns(num_cols_for_ui)
    min_ranges, max_ranges = {}, {}
    for i in range(num_cols_for_ui):
        with cols[i]:
            col_name = f'd{i+1}'
            st.markdown(f"**Pos {i+1}**")
            min_val = int(st.session_state.get('detected_min_ranges', {}).get(col_name, 0))
            max_val = int(st.session_state.get('detected_max_ranges', {}).get(col_name, 10))
            min_ranges[col_name] = st.number_input("Min", value=min_val, key=f"min_{i}")
            max_ranges[col_name] = st.number_input("Max", value=max_val, key=f"max_{i}", min_value=min_ranges[col_name])

    st.header("🔬 Analysis Controls")
    with st.form(key="analysis_form"):
        training_size = st.slider("Training History Size", 50, 5000, 250, 50, disabled=not data_loaded)
        forecast_horizon = st.slider("Forecast Horizon", 5, 50, 10, disabled=not data_loaded)
        cluster_sensitivity = st.slider("Cluster Sensitivity", 5, 50, 15, disabled=not data_loaded)
        run_button = st.form_submit_button("🚀 Run Full System Analysis", type="primary", use_container_width=True, disabled=not data_loaded)

# --- Data Ingestion ---
st.header("Module 0: Data Ingestion & System Detection")
st.markdown("**Note:** This application assumes your data is ordered chronologically, with the **last row being the most recent event.**")
uploaded_file = st.file_uploader("Upload your historical data (CSV)", type=['csv'])

if uploaded_file is not None and uploaded_file.name != st.session_state.get('processed_file_name'):
    try:
        df = pd.read_csv(uploaded_file, header=None, dtype=float)
        st.session_state.processed_file_name = uploaded_file.name
        num_cols = df.shape[1]
        st.session_state.num_columns = num_cols; st.session_state.is_bifurcated = num_cols >= 6
        st.session_state.column_names = [f'd{i+1}' for i in range(num_cols)]
        df.columns = st.session_state.column_names
        st.session_state.detected_min_ranges = {col: df[col].min() for col in df.columns}
        st.session_state.detected_max_ranges = {col: df[col].max() for col in df.columns}
        df.index = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='D'))
        st.session_state.data_full = df
        st.success(f"File interpreted. Detected {num_cols}-D system. Sidebar ranges auto-configured.")
        if 'analysis_run' in st.session_state: del st.session_state.analysis_run
    except Exception as e: st.error(f"Error processing file: {e}"); st.session_state.clear(); st.session_state.processed_file_name = uploaded_file.name

# --- Analysis Execution & Display ---
if run_button and data_loaded:
    with st.spinner("Executing rigorous adaptive analysis..."):
        st.session_state.predictive_results = run_expert_predictive_modeling(st.session_state.data_full, training_size, forecast_horizon, min_ranges, max_ranges, st.session_state.is_bifurcated)
        df_for_clustering = st.session_state.data_full.iloc[:, :5] if st.session_state.is_bifurcated else st.session_state.data_full
        st.session_state.clustering_fig, st.session_state.clustering_df = run_clustering_analysis(df_for_clustering, cluster_sensitivity)
    st.toast("Analysis Complete!", icon="✅"); st.session_state.analysis_run = True

if 'analysis_run' in st.session_state and st.session_state.analysis_run:
    st.subheader("🔮 Next Predicted Event")
    pred_res = st.session_state.predictive_results
    with st.container(border=True):
        if st.session_state.is_bifurcated:
            next_set, next_entity = pred_res['set']['forecast_df'].iloc[0], pred_res['entity']['forecast_df'].iloc[0]
            cols = st.columns(6); 
            for i in range(5): cols[i].metric(f"{next_set.index[i]} (Set)", int(next_set.values[i]))
            cols[5].metric(f"{next_entity.index[0]} (Entity)", int(next_entity.values[0]))
        else:
            next_unified = pred_res['unified']['forecast_df'].iloc[0]
            cols = st.columns(st.session_state.num_columns)
            for i in range(st.session_state.num_columns): cols[i].metric(f"{next_unified.index[i]}", int(next_unified.values[i]))
    
    with st.expander("MODULE 1 — Core Analytical Engine: Time Evolution", expanded=False):
        if st.session_state.is_bifurcated:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Analysis of the Set (d1-d5)")
                sel_set = st.selectbox("Select Column from Set", options=st.session_state.data_full.columns[:5])
                res_set = run_time_series_analysis(st.session_state.data_full, sel_set)
                st.plotly_chart(res_set['fft'], use_container_width=True)
            with c2:
                st.subheader("Analysis of the Entity (d6)")
                res_ent = run_time_series_analysis(st.session_state.data_full, 'd6')
                st.plotly_chart(res_ent['fft'], use_container_width=True)
        else:
            sel_col = st.selectbox("Select Column for Analysis", options=st.session_state.data_full.columns)
            res_uni = run_time_series_analysis(st.session_state.data_full, sel_col)
            st.plotly_chart(res_uni['fft'], use_container_width=True)

    with st.expander("MODULE 2 — Predictive Modeling & Stability", expanded=True):
        def display_forecast_plot(title, train_df, forecast_df, uncertainty_df):
            st.write(f"#### {title}")
            fig = go.Figure()
            for col in train_df.columns:
                fig.add_trace(go.Scatter(x=train_df.index, y=train_df[col], mode='lines', name=f'Hist {col}'))
                fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[col], mode='lines', line=dict(dash='dot'), name=f'Pred {col}'))
                fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[col] + 1.96 * uncertainty_df[col], mode='lines', line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[col] - 1.96 * uncertainty_df[col], mode='lines', fill='tonexty', fillcolor='rgba(255,165,0,0.2)', showlegend=False))
            st.plotly_chart(fig, use_container_width=True)

        if st.session_state.is_bifurcated:
            st.subheader("Bifurcated System Performance")
            m1, m2, m3 = st.columns(3)
            m1.metric("Set (d1-d5) OOS Loss", f"{pred_res['set']['metrics']['oos_loss']:.3f}", help="Avg. prediction error on unseen data.")
            m2.metric("Entity (d6) OOS Loss", f"{pred_res['entity']['metrics']['oos_loss']:.3f}", help="Avg. prediction error on unseen data.")
            m3.metric("Entity Top-3 Accuracy", f"{pred_res['entity']['metrics']['top_n_accuracy']:.2%}" if pred_res['entity']['metrics']['top_n_accuracy'] is not None else "N/A")
            display_forecast_plot("Forecast for 5-Entity Set (with 95% Confidence)", st.session_state.data_full.iloc[:, :5].tail(training_size), pred_res['set']['forecast_df'], pred_res['set']['uncertainty_df'])
        else:
            st.subheader("Unified System Performance")
            m1, m2 = st.columns(2)
            m1.metric("Unified System OOS Loss", f"{pred_res['unified']['metrics']['oos_loss']:.3f}")
            m2.metric("Forecast Stability", f"{pred_res['unified']['metrics']['forecast_stability']:.3f}")
            display_forecast_plot(f"Forecast for {st.session_state.num_columns}-D Unified System", st.session_state.data_full.tail(training_size), pred_res['unified']['forecast_df'], pred_res['unified']['uncertainty_df'])

    with st.expander("MODULE 3 — System Dynamics & Regime Discovery", expanded=True):
        clustering_df = st.session_state.get('clustering_df')
        if clustering_df is not None:
            if st.session_state.is_bifurcated:
                st.subheader("Interactive Conditional Distribution Analysis")
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(st.session_state.clustering_fig, use_container_width=True)
                    available_clusters = sorted(clustering_df['Cluster'].unique())
                    selected_cluster = st.selectbox("Analyze Entity behavior when the Set is in Regime:", options=available_clusters)
                with c2:
                    indices = clustering_df[clustering_df['Cluster'] == selected_cluster].index
                    filtered_entity_df = st.session_state.data_full.loc[indices].iloc[:, 5:]
                    fig_cond = run_entity_distribution_analysis(filtered_entity_df, min_ranges['d6'], max_ranges['d6'], title_suffix=f"(Set in Regime '{selected_cluster}')")
                    st.plotly_chart(fig_cond, use_container_width=True)
            else: st.plotly_chart(st.session_state.clustering_fig, use_container_width=True)
