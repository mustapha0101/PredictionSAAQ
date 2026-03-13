import streamlit as st

# --- Page config ---
st.set_page_config(page_title="Prévisions accidents et victimes", layout="wide")

# --- Custom Loader Animation CEI2A ---
loader_placeholder = st.empty()
loader_placeholder.markdown("""
<style>
.loader-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-color: #0E1117;
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 999999;
}
.loader-text {
    font-size: 6rem;
    font-weight: 900;
    background: linear-gradient(90deg, #1f77b4, #4CAF50, #1f77b4);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 15px;
    font-family: 'Helvetica Neue', sans-serif;
    animation: shine 2s linear infinite;
}
@keyframes shine {
    to {
        background-position: 200% center;
    }
}
</style>
<div class="loader-container" id="cei2a-loader">
    <div class="loader-text">CEI2A</div>
</div>
""", unsafe_allow_html=True)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm  # pour compatibilité si besoin
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Try importing Prophet. If unavailable, set Prophet to None.
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    from prophet.plot import plot_cross_validation_metric
    prophet_available = True
except Exception:
    Prophet = None
    prophet_available = False

# --- Sidebar Header with Logo ---
st.sidebar.markdown("""
<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; margin-bottom: 2rem;">
    <img src="https://upload.wikimedia.org/wikipedia/fr/thumb/9/99/SAAQ_logo.svg/3840px-SAAQ_logo.svg.png" alt="SAAQ Logo" style="width: 80%; margin-bottom: 1rem;">
    <h2 style="color: #2E4057; margin-bottom: 0;">CEI2A</h2>
    <p style="color: #6C757D; margin-top: 5px; font-weight: bold; font-size: 0.9rem;">Centre d'expertise en analytique et IA</p>
    <hr style="width: 100%; border-top: 1px solid #ddd;">
</div>
""", unsafe_allow_html=True)
st.title("📊 Analyse et prévisions des accidents et des victimes")

# --- Load data ---
@st.cache_data
def load_data():
    file_names = [f'data/Rapport_Accident_{year}.csv' for year in range(2016, 2023)]
    frames = []
    for f in file_names:
        try:
            df = pd.read_csv(f, engine='python')
        except FileNotFoundError:
            continue
        # Remove BOM in column names if present
        df.columns = df.columns.str.replace('\ufeff', '', regex=False)
        if 'AN' in df.columns and 'MS_ACCDN' in df.columns:
            df['MS_ACCDN'] = pd.to_numeric(df['MS_ACCDN'], errors='coerce')
            df['AN'] = pd.to_numeric(df['AN'], errors='coerce')
            df = df.dropna(subset=['MS_ACCDN', 'AN'])
            df['MS_ACCDN'] = df['MS_ACCDN'].astype(int)
            df['AN'] = df['AN'].astype(int)
            df['date'] = pd.to_datetime(dict(year=df['AN'], month=df['MS_ACCDN'], day=1))
            frames.append(df)
    if not frames:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)
    data = pd.concat(frames, ignore_index=True)
    accidents_series = data.groupby('date').size().sort_index()
    accidents_series.index.freq = pd.infer_freq(accidents_series.index) or 'MS'
    victims_series = pd.Series(dtype=float)
    if 'NB_VICTIMES_TOTAL' in data.columns:
        victims_series = data.groupby('date')['NB_VICTIMES_TOTAL'].sum().sort_index()
        victims_series.index.freq = pd.infer_freq(victims_series.index) or 'MS'
    return data, accidents_series, victims_series

# Load data
data, series, victims_series = load_data()

# Suppression du loader une fois l'import/chargement terminé
loader_placeholder.empty()

# --- Forecast function ---
@st.cache_data
def forecast_models(series, horizon=36, start_today=False,rerun=False):
    # require at least some data
    if len(series) < 2:
        return {}
    # Holt-Winters models
    hw_add = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12, initialization_method='estimated').fit()
    hw_mul = ExponentialSmoothing(series, trend='add', seasonal='mul', seasonal_periods=12, initialization_method='estimated').fit()
    se_add = hw_add.resid.dropna().std() if hasattr(hw_add, 'resid') else 0.0
    se_mul = hw_mul.resid.dropna().std() if hasattr(hw_mul, 'resid') else 0.0
    if start_today:
        start_fc = pd.to_datetime(datetime.today().strftime("%Y-%m-01"))
    else:
        start_fc = series.index[-1] + pd.offsets.MonthBegin(1)
    idx = pd.date_range(start=start_fc, periods=horizon, freq='MS')
    hw_add_fc = hw_add.forecast(horizon)
    hw_mul_fc = hw_mul.forecast(horizon)
    
    # Correction des intervalles de confiance (croissance exponentielle façon "tube")
    h_array = np.arange(1, horizon + 1)
    expansion_factor = np.sqrt(h_array) * (1 + 0.05 * h_array)
    
    hw_add_ci = pd.DataFrame({
        'pred': hw_add_fc.round(0),
        'low': (hw_add_fc - 1.96 * se_add * expansion_factor).round(0),
        'high': (hw_add_fc + 1.96 * se_add * expansion_factor).round(0)
    }, index=idx)
    hw_mul_ci = pd.DataFrame({
        'pred': hw_mul_fc.round(0),
        'low': (hw_mul_fc - 1.96 * se_mul * expansion_factor).round(0),
        'high': (hw_mul_fc + 1.96 * se_mul * expansion_factor).round(0)
    }, index=idx)
    results = {'HW_add': hw_add_ci, 'HW_mul': hw_mul_ci}
    
    # Prophet model if available
    if prophet_available and Prophet is not None and len(series) >= 2:
        df_prophet = series.reset_index()
        df_prophet.columns = ['ds', 'y']
        try:
            covid_dates = pd.DataFrame({
              'holiday': 'covid_lockdown',
              'ds': pd.to_datetime(['2020-03-01', '2020-04-01', '2020-05-01', '2020-12-01', '2021-01-01']),
              'lower_window': 0,
              'upper_window': 1,
            })
            m = Prophet(
                yearly_seasonality=True, 
                weekly_seasonality=False, 
                daily_seasonality=False, 
                interval_width=0.95,  # Augmentation de la largeur de l'intervalle pour un meilleur coverage
                holidays=covid_dates,
                changepoint_prior_scale=0.1  # Plus grande flexibilité de la tendance
            )
            
            # --- Ajout d'un régresseur externe (Météo difficile) ---
            # Si nous prédisons la série principale et qu'on a le df_prophet
            m.add_regressor('mauvaise_meteo', prior_scale=0.5)
            
            # Préparer la donnée météo
            # 13: Brouillard, 14: Pluie, 15: Averse, 17: Neige, 18: Poudrerie, 19: Verglas
            mauvaise_meteo_codes = [13, 14, 15, 17, 18, 19]
            # On va compter la proportion ou le nombre d'accidents par mois avec mauvaise météo dans "data"
            # Note: cela nécessite que "data" soit accessible tel quel. 
            if 'CD_COND_METEO' in data.columns:
                df_meteo = data.copy()
                df_meteo['is_bad_weather'] = df_meteo['CD_COND_METEO'].isin(mauvaise_meteo_codes).astype(int)
                meteo_monthly = df_meteo.groupby('date')['is_bad_weather'].sum().reset_index()
                meteo_monthly.columns = ['ds', 'mauvaise_meteo']
                # Merge avec df_prophet
                df_prophet = df_prophet.merge(meteo_monthly, on='ds', how='left').fillna(0)
            else:
                df_prophet['mauvaise_meteo'] = 0
            
            m.fit(df_prophet)
            future = m.make_future_dataframe(periods=horizon, freq='MS')
            
            # Il faut fournir la variable exogène au futur (On projette la moyenne mensuelle historique)
            if 'mauvaise_meteo' in df_prophet.columns:
                df_prophet['month'] = df_prophet['ds'].dt.month
                monthly_avg_meteo = df_prophet.groupby('month')['mauvaise_meteo'].mean().to_dict()
                future['month'] = future['ds'].dt.month
                future['mauvaise_meteo'] = future['month'].map(monthly_avg_meteo)
                future = future.drop(columns=['month'])
                
            forecast = m.predict(future)
            if start_today:
                idx2 = pd.date_range(start=start_fc, periods=horizon, freq='MS')
                fc = forecast.set_index('ds').loc[idx2]
            else:
                idx2 = pd.date_range(start=series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq='MS')
                fc = forecast.set_index('ds').loc[idx2]
            prophet_ci = pd.DataFrame({
                'pred': fc['yhat'].round(0),
                'low': fc['yhat_lower'].round(0),
                'high': fc['yhat_upper'].round(0)
            }, index=idx2)
            results['Prophet'] = prophet_ci
        except Exception:
            # si Prophet échoue, on ignore silencieusement (déjà géré par prophet_available)
            pass
    return results

# --- Evaluation function (Train/Test Split) ---
@st.cache_data
def evaluate_models(series, test_size=12):
    if len(series) <= 24: # Need enough data to train and test
        return pd.DataFrame()
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    
    results = []
    
    # 1. HW Additive
    try:
        hw_add = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12, initialization_method='estimated').fit()
        pred_add = hw_add.forecast(test_size)
        results.append({'Modèle': 'HW_add', 'RMSE': np.sqrt(((test - pred_add)**2).mean()), 'MAE': np.abs(test - pred_add).mean()})
    except Exception:
        pass
        
    # 2. HW Multiplicative
    try:
        hw_mul = ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=12, initialization_method='estimated').fit()
        pred_mul = hw_mul.forecast(test_size)
        results.append({'Modèle': 'HW_mul', 'RMSE': np.sqrt(((test - pred_mul)**2).mean()), 'MAE': np.abs(test - pred_mul).mean()})
    except Exception:
        pass
        
    # 3. Prophet
    if prophet_available and Prophet is not None:
        try:
            df_prophet = train.reset_index()
            df_prophet.columns = ['ds', 'y']
            
            covid_dates = pd.DataFrame({
              'holiday': 'covid_lockdown',
              'ds': pd.to_datetime(['2020-03-01', '2020-04-01', '2020-05-01', '2020-12-01', '2021-01-01']),
              'lower_window': 0,
              'upper_window': 1,
            })
            
            m = Prophet(
                yearly_seasonality=True, 
                weekly_seasonality=False, 
                daily_seasonality=False, 
                interval_width=0.95,
                holidays=covid_dates,
                changepoint_prior_scale=0.1
            )
            m.add_regressor('mauvaise_meteo', prior_scale=0.5)
            
            if 'CD_COND_METEO' in data.columns:
                df_meteo = data.copy()
                mauvaise_meteo_codes = [13, 14, 15, 17, 18, 19]
                df_meteo['is_bad_weather'] = df_meteo['CD_COND_METEO'].isin(mauvaise_meteo_codes).astype(int)
                meteo_monthly = df_meteo.groupby('date')['is_bad_weather'].sum().reset_index()
                meteo_monthly.columns = ['ds', 'mauvaise_meteo']
                df_prophet = df_prophet.merge(meteo_monthly, on='ds', how='left').fillna(0)
            else:
                df_prophet['mauvaise_meteo'] = 0
                
            m.fit(df_prophet)
            
            future = m.make_future_dataframe(periods=test_size, freq='MS')
            if 'mauvaise_meteo' in df_prophet.columns:
                df_prophet['month'] = df_prophet['ds'].dt.month
                monthly_avg_meteo = df_prophet.groupby('month')['mauvaise_meteo'].mean().to_dict()
                future['month'] = future['ds'].dt.month
                future['mauvaise_meteo'] = future['month'].map(monthly_avg_meteo)
                future = future.drop(columns=['month'])
                
            forecast = m.predict(future)
            pred_prophet = forecast.set_index('ds').loc[test.index, 'yhat']
            
            results.append({'Modèle': 'Prophet', 'RMSE': np.sqrt(((test - pred_prophet)**2).mean()), 'MAE': np.abs(test - pred_prophet).mean()})
        except Exception:
            pass
            
    if results:
        df_res = pd.DataFrame(results).set_index('Modèle').round(2)
        df_res = df_res.sort_values('MAE')
        return df_res
    return pd.DataFrame()

# --- Sidebar: paramètres ---
st.sidebar.header("⚙️ Paramètres de prévision")

# Horizon par défaut
default_horizon = 36
horizon = st.sidebar.number_input(
    "Horizon de prévision (mois)",
    min_value=12, max_value=90, value=default_horizon
)

# --- Bouton pour réentraîner ---
if st.sidebar.button("♻️ Réentraîner de zéro"):
    st.session_state['rerun_flag'] = not st.session_state.get('rerun_flag', False)
    # Recalculer immédiatement les modèles
    st.session_state['results_accidents'] = forecast_models(series, horizon, rerun=st.session_state['rerun_flag'])
    st.session_state['results_victims'] = forecast_models(victims_series, horizon, rerun=st.session_state['rerun_flag'])
    st.success("✅ Modèles réentraînés avec succès !")


# --- Initialisation des prévisions ---
# Accidents
if 'results_accidents' not in st.session_state:
    st.session_state['results_accidents'] = forecast_models(
        series, horizon, rerun=st.session_state.get('rerun', False)
    )

# Victimes
if 'results_victims' not in st.session_state:
    st.session_state['results_victims'] = forecast_models(
        victims_series, horizon, rerun=st.session_state.get('rerun', False)
    )


# Define tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Exploration", "Prévisions", "Analyse par variable", "Comparaison", "Prévisions par région"])

# --- Helper: plot decomposition with changepoints and outliers ---
def plot_decomposition_with_changepoints_and_outliers(series, title_prefix="Série"):
    """
    series : pd.Series with DatetimeIndex
    """
    if len(series) < 12:
        st.warning(f"Pas assez de données pour afficher la décomposition de {title_prefix} (min 12 mois).")
        return

    # Decompose
    decomposition = seasonal_decompose(series, model='additive', period=12, extrapolate_trend='freq')

    # Prepare df for Prophet if available
    df_prophet = series.reset_index()
    df_prophet.columns = ['ds', 'y']

    # Detect changepoints using Prophet if available
    changepoints = []
    if prophet_available and Prophet is not None:
        try:
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.fit(df_prophet)
            # m.changepoints is a pandas Series of timestamps
            cps = getattr(m, 'changepoints', None)
            if cps is not None:
                changepoints = list(cps)
        except Exception:
            changepoints = []

    # Residuals and outliers
    resid = decomposition.resid.dropna()
    resid_mean = resid.mean()
    resid_std = resid.std()
    if pd.isna(resid_std) or resid_std == 0:
        outliers_mask = pd.Series(False, index=resid.index)
    else:
        outliers_mask = (resid - resid_mean).abs() > 2 * resid_std

    # Plot with matplotlib (3 rows)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    dates_full = series.index

    # Tendance (align trend with index)
    trend = decomposition.trend
    axes[0].plot(dates_full, series.values, label="Historique", alpha=0.3)
    axes[0].plot(trend.index, trend.values, label="Tendance", color='tab:blue')
    axes[0].set_title(f"{title_prefix} — Tendance")
    axes[0].legend()

    # Saison
    seasonal = decomposition.seasonal
    axes[1].plot(seasonal.index, seasonal.values, label="Saisonnalité", color='tab:orange')
    axes[1].set_title(f"{title_prefix} — Saisonnalité")
    axes[1].legend()

    # Residus + outliers
    axes[2].plot(resid.index, resid.values, label="Résidus", color='tab:green')
    if outliers_mask.any():
        axes[2].scatter(resid.index[outliers_mask], resid[outliers_mask], color='red', label="Outliers (>|2σ|)")
    axes[2].axhline(resid_mean, color="black", linestyle="--", alpha=0.7, label="Moyenne résidus")
    axes[2].set_title(f"{title_prefix} — Résidus (avec outliers)")
    axes[2].legend()

    # Add changepoints as vertical lines on all axes
    for cp in changepoints:
        for ax in axes:
            ax.axvline(cp, color="red", linestyle="--", alpha=0.7)

    plt.tight_layout()
    st.pyplot(fig)

    # Explanations
    explanation = f"""
    **Interprétation pour : {title_prefix}**

    - **Tendance** : évolution générale de la série au fil du temps.
    - **Saisonnalité** : composante récurrente (périodicité annuelle / mensuelle).
    - **Résidus** : partie non expliquée par la somme (tendance + saisonnalité).
    - **Outliers** (points rouges) : résidus au-delà de ±2 écarts-types — ce sont des valeurs inhabituelles qui méritent une vérification (erreur de saisie, événement rare, changement structurel).
    - **Lignes verticales rouges** : changepoints détectés automatiquement par Prophet (si disponible) — points où la tendance a potentiellement changé.
    """
    st.markdown(explanation)


# --- Tab 1: Exploration ---
with tab1:
    st.header("📈 Exploration Accidents & Victimes")

    # --- Période (Filtre masqué) ---
    min_date = data['date'].min()
    max_date = data['date'].max()
    start_date = min_date
    end_date = max_date

    # Filtrage des données
    data_filtered = data[(data['date'] >= pd.to_datetime(start_date)) & (data['date'] <= pd.to_datetime(end_date))]
    accidents_series_filtered = series[(series.index >= pd.to_datetime(start_date)) & (series.index <= pd.to_datetime(end_date))]
    victims_series_filtered = victims_series[(victims_series.index >= pd.to_datetime(start_date)) & (victims_series.index <= pd.to_datetime(end_date))]

    # --- Graphiques côte-à-côte ---
    st.subheader("📊 Séries temporelles")
    col1, col2 = st.columns(2)

    # Accidents
    with col1:
        st.markdown("**Accidents**")
        if len(accidents_series_filtered) >= 12:
            decomp_acc = seasonal_decompose(accidents_series_filtered, model='additive', period=12)
            trend_acc, seasonal_acc, resid_acc = decomp_acc.trend, decomp_acc.seasonal, decomp_acc.resid
            resid_std_acc = resid_acc.dropna().std()
            outliers_acc = resid_acc.abs() > 2*resid_std_acc

            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=accidents_series_filtered.index, y=accidents_series_filtered.values.round(0),
                                         mode="lines", name="Historique"))
            fig_acc.add_trace(go.Scatter(x=trend_acc.index, y=trend_acc.values.round(0),
                                         mode="lines", name="Tendance", line=dict(dash="dash")))
            fig_acc.add_trace(go.Scatter(x=seasonal_acc.index, y=seasonal_acc.values.round(0),
                                         mode="lines", name="Saisonnalité", line=dict(dash="dot")))
            fig_acc.add_trace(go.Scatter(x=resid_acc.index[outliers_acc], y=accidents_series_filtered.loc[outliers_acc].round(0),
                                         mode="markers", name="Outliers", marker=dict(color="red", size=8)))
            fig_acc.update_layout(title="Accidents : Historique + Tendance + Saison + Outliers",
                                  xaxis_title="Date", yaxis_title="Nombre d'accidents")
            st.plotly_chart(fig_acc, use_container_width=True)
        else:
            st.info("Pas assez de données pour la décomposition des accidents (min 12 mois).")

    # Victimes
    with col2:
        st.markdown("**Victimes**")
        if len(victims_series_filtered) >= 12:
            decomp_vic = seasonal_decompose(victims_series_filtered, model='additive', period=12)
            trend_vic, seasonal_vic, resid_vic = decomp_vic.trend, decomp_vic.seasonal, decomp_vic.resid
            resid_std_vic = resid_vic.dropna().std()
            outliers_vic = resid_vic.abs() > 2*resid_std_vic

            fig_vic = go.Figure()
            fig_vic.add_trace(go.Scatter(x=victims_series_filtered.index, y=victims_series_filtered.values.round(0),
                                         mode="lines", name="Historique"))
            fig_vic.add_trace(go.Scatter(x=trend_vic.index, y=trend_vic.values.round(0),
                                         mode="lines", name="Tendance", line=dict(dash="dash")))
            fig_vic.add_trace(go.Scatter(x=seasonal_vic.index, y=seasonal_vic.values.round(0),
                                         mode="lines", name="Saisonnalité", line=dict(dash="dot")))
            fig_vic.add_trace(go.Scatter(x=resid_vic.index[outliers_vic], y=victims_series_filtered.loc[outliers_vic].round(0),
                                         mode="markers", name="Outliers", marker=dict(color="red", size=8)))
            fig_vic.update_layout(title="Victimes : Historique + Tendance + Saison + Outliers",
                                  xaxis_title="Date", yaxis_title="Nombre de victimes")
            st.plotly_chart(fig_vic, use_container_width=True)
        else:
            st.info("Pas assez de données pour la décomposition des victimes (min 12 mois).")

    # --- Statistiques descriptives combinées ---
    st.subheader("📋 Statistiques descriptives combinées (/ mois pour toute la province)")
    stats_df = pd.DataFrame({
        'Nb valeurs': [accidents_series_filtered.count(), victims_series_filtered.count()],
        'Moyenne': [accidents_series_filtered.mean(), victims_series_filtered.mean()],
        'Médiane': [accidents_series_filtered.median(), victims_series_filtered.median()],
        'Écart-type': [accidents_series_filtered.std(), victims_series_filtered.std()],
        'Min': [accidents_series_filtered.min(), victims_series_filtered.min()],
        'Max': [accidents_series_filtered.max(), victims_series_filtered.max()]
    }, index=['Accidents', 'Victimes']).reset_index().rename(columns={'index':'Variable'})
    st.dataframe(stats_df)

    # --- Corrélations simples entre accidents et victimes ---
    st.subheader("🔗 Corrélation Accidents ↔ Victimes")
    if len(accidents_series_filtered) > 1 and len(victims_series_filtered) > 1:
        combined_df = pd.DataFrame({
            'Accidents': accidents_series_filtered,
            'Victimes': victims_series_filtered
        }).dropna()
        corr_val = combined_df.corr().iloc[0,1]
        st.metric("Coefficient de corrélation (Pearson)", f"{corr_val:.2f}")
    else:
        st.info("Pas assez de données pour calculer la corrélation.")

        # --- Victimes ---
    st.subheader("📈 Série des victimes avec décomposition des tendances et anomalies")
    if len(victims_series_filtered) >= 12:
        decomp_victims = seasonal_decompose(victims_series_filtered)
        trend_v, seasonal_v, resid_v = decomp_victims.trend, decomp_victims.seasonal, decomp_victims.resid

        # Outliers et ruptures
        resid_std_v = resid_v.dropna().std()
        outliers_mask_v = resid_v.abs() > 2*resid_std_v
        trend_diff_v = trend_v.diff().abs()
        thresh_v = trend_diff_v.mean() + 2*trend_diff_v.std()
        breaks_mask_v = trend_diff_v > thresh_v

        # Limitation des points pour Plotly
        max_points = 200
        idx_plot_v = victims_series_filtered.index
        if len(idx_plot_v) > max_points:
            idx_plot_v = idx_plot_v[-max_points:]
            trend_v = trend_v.loc[idx_plot_v]
            seasonal_v = seasonal_v.loc[idx_plot_v]
            resid_v = resid_v.loc[idx_plot_v]
            outliers_mask_v = outliers_mask_v.loc[idx_plot_v]
            breaks_mask_v = breaks_mask_v.loc[idx_plot_v]

        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=idx_plot_v, y=victims_series_filtered.loc[idx_plot_v].values.round(0),
                                mode="lines", name="Historique"))
        fig_v.add_trace(go.Scatter(x=trend_v.index, y=trend_v.values.round(0),
                                mode="lines", name="Tendance", line=dict(dash="dash")))
        fig_v.add_trace(go.Scatter(x=seasonal_v.index, y=seasonal_v.values.round(0),
                                mode="lines", name="Saisonnalité", line=dict(dash="dot")))
        fig_v.add_trace(go.Scatter(x=resid_v.index[outliers_mask_v], y=victims_series_filtered.loc[outliers_mask_v].round(0),
                                mode="markers", name="Outliers", marker=dict(color="red", size=10)))
        fig_v.add_trace(go.Scatter(x=trend_diff_v.index[breaks_mask_v], y=victims_series_filtered.loc[breaks_mask_v].round(0),
                                mode="markers", name="Ruptures", marker=dict(color="green", size=12, symbol="x")))
        fig_v.update_layout(title="Victimes : série historique avec décomposition, outliers et ruptures",
                            xaxis_title="Date", yaxis_title="Nombre de victimes")
        st.plotly_chart(fig_v, use_container_width=True)

    else:
        st.info("Pas assez de données pour la décomposition des victimes (minimum 12 mois requis).")

        # --- Statistiques descriptives pour victimes ---
        st.subheader("📋 Statistiques descriptives des victimes")
        if len(victims_series_filtered) > 0:
            stats_v = pd.DataFrame({
                'Nb valeurs': victims_series_filtered.count(),
                'Moyenne': victims_series_filtered.mean(),
                'Médiane': victims_series_filtered.median(),
                'Écart-type': victims_series_filtered.std(),
                'Min': victims_series_filtered.min(),
                'Max': victims_series_filtered.max()
            }, index=['Victimes']).reset_index().rename(columns={'index':'Variable'})
            st.dataframe(stats_v)


            # --- Corrélation ---
            st.subheader("🔗 Corrélations")
            st.markdown("""
                ## Corrélation de Pearson
                La corrélation de Pearson mesure la **force et la direction** de la relation linéaire entre deux variables.  
                    Sa valeur varie entre **-1** et **1** :

                    - **1** : corrélation positive parfaite  
                    - **0** : absence de corrélation linéaire  
                    - **-1** : corrélation négative parfaite

                La corrélation entre les deux variables  indique :

                - Relation (ex : **positive forte**)  : quand une variable augmente, l'autre tend à augmenter aussi.  
                - Pas de **causalité** : cela montre seulement une association linéaire.
                """)
            numeric_cols = data_filtered.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) >= 2:
                corr_matrix = data_filtered[numeric_cols].corr(method='pearson')
                fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', title="Matrice de corrélation (Pearson)")
                st.plotly_chart(fig_corr, use_container_width=True)
            
            else:
                st.info("Pas assez de colonnes numériques pour calculer la corrélation.")


# --- Tab 2: Predictions ---
with tab2:
    st.header("🔮 Prévisions")
    option = st.selectbox("Sélectionnez la série à prédire", ['Accidents', 'Victimes'])
    current_series = series if option == 'Accidents' else victims_series
    if len(current_series) < 1:
        st.warning("La série sélectionnée est vide ou indisponible.")
    else:
        av_models = list(forecast_models(current_series, horizon).keys())
        default_idx = av_models.index('Prophet') if 'Prophet' in av_models else 0
        model_choice = st.selectbox("Sélectionnez un modèle", av_models, index=default_idx)
        hor = st.slider("Horizon (mois)", 36, 90, horizon, key='pred_hor')
        results_for = forecast_models(current_series, hor)
        if model_choice not in results_for:
            st.error("Le modèle sélectionné n'est pas disponible pour cette série.")
        else:
            dfc = results_for[model_choice]
            show_hist = st.checkbox("Afficher l'historique", value=True)
            fig_fc = go.Figure()
            if show_hist:
                fig_fc.add_trace(go.Scatter(x=current_series.index, y=current_series.values.round(0), mode='lines', name='Historique'))
            fig_fc.add_trace(go.Scatter(x=dfc.index, y=dfc['pred'], mode='lines+markers', name=f'Prévisions {model_choice}'))
            fig_fc.add_trace(go.Scatter(
                x=list(dfc.index) + list(dfc.index)[::-1],
                y=list(dfc['high']) + list(dfc['low'])[::-1],
                fill='toself', fillcolor='rgba(255,165,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip', showlegend=True, name='Intervalle de confiance 95%'
            ))
            fig_fc.add_vline(x=pd.Timestamp.today(), line_width=2, line_dash='dash', line_color='black')
            fig_fc.update_layout(title=f'Prévisions de {option} avec {model_choice}', xaxis_title='Date', yaxis_title='Valeurs')
            st.plotly_chart(fig_fc, use_container_width=True)
            st.subheader("Tableau des prévisions")
            st.dataframe(dfc)
    st.markdown("""
        ## Comparaison des méthodes de prévision

        - **HW_Add (Holt-Winters Additif)** :  
        Prévision avec **tendance et saisonnalité additive**. Utile quand les variations saisonnières sont **constantes dans le temps**.

        - **HW_Mul (Holt-Winters Multiplicatif)** :  
        Prévision avec **tendance et saisonnalité multiplicative**. Adapté quand l’amplitude saisonnière **augmente avec le niveau de la série**.

        - **Prophet** :  
        Modèle robuste développé par Facebook pour les séries temporelles avec **tendance, saisonnalité, jours fériés** et pouvant gérer **changements soudains**. Plus flexible que Holt-Winters pour des séries complexes.
        """)
        

# --- Tab 3: Variable analysis ---
with tab3:
    st.header("🔍 Analyse par variables")
    st.markdown("Explorez l'impact des conditions environnementales sur la fréquence et la **sévérité** des accidents.")
    
    categorical_vars = {
        'CD_ETAT_SURFC': {'desc': 'État de la chaussée', 'mapping': {11: 'Sèche', 12: 'Mouillée', 13: 'Eau', 14: 'Neige', 15: 'Glace', 99: 'Autre'}},
        'CD_ECLRM': {'desc': 'Éclairement', 'mapping': {1: 'Jour', 2: 'Crépuscule', 3: 'Nuit éclairée', 4: 'Nuit non éclairée'}},
        'CD_ENVRN_ACCDN': {'desc': 'Environnement', 'mapping': {1: 'Scolaire', 2: 'Résidentiel', 3: 'Affaires', 4: 'Industriel', 5: 'Rural', 6: 'Forestier'}},
        'CD_COND_METEO': {'desc': 'Météo', 'mapping': {11: 'Clair', 12: 'Couvert', 13: 'Brouillard', 14: 'Pluie', 15: 'Averse', 16: 'Vent', 17: 'Neige', 18: 'Poudrerie', 19: 'Verglas', 99: 'Autre'}}
    }
    
    selected_var_key = st.selectbox("Sélectionnez le facteur à analyser :", list(categorical_vars.keys()), format_func=lambda x: categorical_vars[x]['desc'])
    meta = categorical_vars[selected_var_key]
    var = selected_var_key

    if var in data.columns:
        st.markdown(f"### Détails pour : {meta['desc']}")
        
        # Application du mapping
        data_mapped = data.copy()
        data_mapped[var] = data_mapped[var].map(meta['mapping']).fillna('Inconnu')
        
        # Filtre optionnel pour enlever "Inconnu"
        data_mapped = data_mapped[data_mapped[var] != 'Inconnu']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Répartition (%)
            counts = data_mapped[var].value_counts().reset_index()
            counts.columns = [meta['desc'], "Nombre d'accidents"]
            fig_pie = px.pie(counts, values="Nombre d'accidents", names=meta['desc'], title=f"Répartition par {meta['desc'].lower()}", hole=0.4, color_discrete_sequence=px.colors.sequential.Blues_r)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            # Sévérité (Victimes par accident)
            if 'NB_VICTIMES_TOTAL' in data_mapped.columns:
                severity = data_mapped.groupby(var)['NB_VICTIMES_TOTAL'].mean().reset_index()
                severity.columns = [meta['desc'], "Victimes moy. / accident"]
                severity = severity.sort_values(by="Victimes moy. / accident", ascending=False)
                fig_sev = px.bar(severity, x=meta['desc'], y="Victimes moy. / accident", title="Sévérité du risque (Victimes par accident)", color="Victimes moy. / accident", color_continuous_scale="Reds")
                st.plotly_chart(fig_sev, use_container_width=True)
            else:
                st.info("Sévérité indisponible (colonne NB_VICTIMES_TOTAL manquante).")
                severity = pd.DataFrame()
        
        # Evolution temporelle
        st.markdown(f"#### Évolution mensuelle selon : {meta['desc'].lower()}")
        time_dist = data_mapped.groupby(['date', var]).size().reset_index(name='Nombre d\'accidents')
        fig_time = px.line(time_dist, x='date', y='Nombre d\'accidents', color=var, title=f"Tendances historiques croisées")
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Insights texte narratif
        if not counts.empty:
            st.info(f"📌 **Fréquence** : La condition **{counts.iloc[0][0]}** est historiquement la plus fréquente avec un total de **{counts.iloc[0][1]}** accidents enregistrés.")
            if 'NB_VICTIMES_TOTAL' in data_mapped.columns and not severity.empty:
                st.warning(f"⚠️ **Facteur de Risque Majeur** : Bien que moins fréquente, la situation **{severity.iloc[0][0]}** génère les accidents les plus graves (en moyenne **{severity.iloc[0][1]:.2f}** victimes à chaque incident).")
    else:
        st.error(f"La colonne {var} est introuvable dans le jeu de données SAAQ.")

# --- Tab 4: Model comparison ---
with tab4:
    st.header("⚖️ Comparaison des modèles")
    
    st.subheader("📊 Évaluation des modèles (Train/Test Split)")
    st.markdown("Les modèles sont évalués en utilisant la dernière année complète (12 derniers mois) comme ensemble de test. Les métriques **RMSE** (Erreur Quadratique Moyenne) et **MAE** (Erreur Absolue Moyenne) permettent d'identifier le meilleur modèle.")
    
    option_eval = st.radio("Sélectionnez la série à évaluer", ['Accidents', 'Victimes'], horizontal=True)
    eval_series = series if option_eval == 'Accidents' else victims_series
    
    eval_metrics = evaluate_models(eval_series, test_size=12)
    if not eval_metrics.empty:
        st.dataframe(eval_metrics.style.highlight_min(color='lightgreen', axis=0))
        best_model = eval_metrics.index[0]
        st.success(f"🏆 Le modèle recommandé pour les {option_eval.lower()} est **{best_model}** (selon le score MAE).")
    else:
        st.info("Pas assez de données pour l'évaluation Train/Test (besoin de > 24 mois).")
        
    # --- Native Prophet Cross Validation ---
    if prophet_available and Prophet is not None:
        st.subheader("🔁 Validation Croisée (Cross-Validation) Avancée avec Prophet")
        st.markdown(
            "Prophet permet une évaluation robuste en simulant plusieurs prévisions dans le passé (*backtesting*). "
            "Il coupe l'historique plusieurs fois pour tester sa capacité de généralisation."
        )
        if st.button("Lancer la Cross-Validation Prophet (Peut prendre quelques secondes)"):
            with st.spinner("Exécution de la validation croisée de Prophet..."):
                try:
                    df_cv_prophet = eval_series.reset_index()
                    df_cv_prophet.columns = ['ds', 'y']
                    
                    covid_dates = pd.DataFrame({
                      'holiday': 'covid_lockdown',
                      'ds': pd.to_datetime(['2020-03-01', '2020-04-01', '2020-05-01', '2020-12-01', '2021-01-01']),
                      'lower_window': 0,
                      'upper_window': 1,
                    })
                    
                    m_cv = Prophet(
                        yearly_seasonality=True, 
                        weekly_seasonality=False, 
                        daily_seasonality=False, 
                        interval_width=0.95,
                        holidays=covid_dates,
                        changepoint_prior_scale=0.1
                    )
                    
                    m_cv.fit(df_cv_prophet)
                    # ex: Initial 3 ans, horizon 1 an, on teste tous les 6 mois
                    df_cv_result = cross_validation(m_cv, initial='1095 days', period='180 days', horizon='365 days')
                    df_p = performance_metrics(df_cv_result)
                    
                    st.success("Validation croisée terminée avec succès!")
                    st.dataframe(df_p[['horizon', 'rmse', 'mae', 'mape', 'coverage']].head(10))
                    
                    # Plot RMSE
                    fig_cv = go.Figure()
                    
                    # Convert timedelta 'horizon' to integer days for proper plotly display
                    horizon_days = df_p['horizon'].dt.days
                    
                    fig_cv.add_trace(go.Scatter(x=horizon_days, y=df_p['rmse'], mode='lines+markers', name='RMSE (Erreur)'))
                    fig_cv.update_layout(title="Erreur RMSE en fonction de l'horizon de prédiction", xaxis_title="Horizon (en Jours)", yaxis_title="RMSE")
                    st.plotly_chart(fig_cv, use_container_width=True)
                except Exception as e:
                    st.error(f"La validation croisée a échoué. Détails : {str(e)}")

    st.subheader("📈 Projection de comparaison dans le futur")
    if len(series) > 0:
        hor2 = st.slider("Horizon (mois) pour comparaison", 36, 90, default_horizon, key='comp_hor')
        results_comp = forecast_models(series, hor2)
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name='Historique'))
        for name, result in results_comp.items():
            fig4.add_trace(go.Scatter(x=result.index, y=result['pred'], mode='lines', name=name))
        fig4.update_layout(title="Comparaison visuelle des prévisions futures", xaxis_title='Date', yaxis_title='Valeur')
        st.plotly_chart(fig4, use_container_width=True)

# --- Tab 5: Predictions by region ---
with tab5:
    st.header("🌎 Prévisions par région & série")
    if 'REG_ADM' not in data.columns:
        st.warning("⚠️ Pas de colonne 'REG_ADM' dans les données.")
    else:
        regions = sorted(data['REG_ADM'].dropna().unique())
        
        st.info("💡 **Conseil :** Sélectionnez une région à la fois pour éviter de surcharger le serveur de prévision.")
        selected_regions = st.multiselect("Sélectionner une ou plusieurs régions", regions, default=[])
        
        if len(selected_regions) > 2:
            st.warning("⚠️ Attention : Entraîner les modèles (surtout Prophet) sur plusieurs régions simultanément peut saturer la mémoire (Erreur 502) en production.")

        type_ser = st.selectbox("Sélectionner la série pour la région", ['Accidents', 'Victimes'])
        horizon_reg = st.slider("Horizon (mois)", min_value=36, max_value=90, value=36, key='horizon_reg')
        models = st.multiselect("Modèles", ['Holt-Winters Additif', 'Holt-Winters Multiplicatif', 'Prophet'], default=['Holt-Winters Additif', 'Holt-Winters Multiplicatif', 'Prophet'])
        mapping = {'Holt-Winters Additif': 'HW_add', 'Holt-Winters Multiplicatif': 'HW_mul', 'Prophet': 'Prophet'}
        colors = {'Holt-Winters Additif': 'blue', 'Holt-Winters Multiplicatif': 'green', 'Prophet': 'red'}
        fills = {'Holt-Winters Additif': 'rgba(0,0,255,0.2)', 'Holt-Winters Multiplicatif': 'rgba(0,255,0,0.2)', 'Prophet': 'rgba(255,0,0,0.2)'}
        for reg in selected_regions:
            st.subheader(f"Prévisions pour {reg} - {type_ser}")
            df_reg = data[data['REG_ADM'] == reg]
            if type_ser == 'Accidents':
                s = df_reg.groupby('date').size().sort_index()
            else:
                if 'NB_VICTIMES_TOTAL' in df_reg.columns:
                    s = df_reg.groupby('date')['NB_VICTIMES_TOTAL'].sum().sort_index()
                else:
                    s = pd.Series(dtype=float)
            if len(s) < 12:
                st.warning(f"Pas assez de données pour {reg} / {type_ser} pour générer des prévisions")
                continue
            res = forecast_models(s, horizon_reg)
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(x=s.index, y=s.values, mode='lines', name='Historique'))
            for m in models:
                key = mapping.get(m)
                if key not in res:
                    continue
                dfc2 = res[key]
                fig_reg.add_trace(go.Scatter(x=dfc2.index, y=dfc2['pred'], mode='lines+markers', name=m, line=dict(color=colors[m])))
                fig_reg.add_trace(go.Scatter(
                    x=list(dfc2.index) + list(dfc2.index)[::-1],
                    y=list(dfc2['high']) + list(dfc2['low'])[::-1],
                    fill='toself', fillcolor=fills[m], line=dict(color='rgba(255,255,255,0)'), showlegend=False
                ))
            fig_reg.update_layout(title=f"{type_ser} pour {reg}", xaxis_title='Date', yaxis_title='Nombre')
            st.plotly_chart(fig_reg, use_container_width=True)
