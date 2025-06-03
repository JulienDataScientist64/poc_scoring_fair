# --- Core Python and File System ---
import os
import re
import requests
import importlib.util
import sys
from typing import List, Dict, Any, Optional

# --- Core Data Handling & Computation ---
import pandas as pd
import numpy as np
import joblib

# --- Streamlit (Application Framework) ---
import streamlit as st

# --- Plotting & Visualization ---
import plotly.express as px 
import plotly.graph_objects as go 
import plotly.figure_factory as ff 

# --- Fairness Libraries ---
from fairlearn.metrics import (
    MetricFrame, 
    selection_rate as fairlearn_selection_rate, 
    demographic_parity_difference,
    equalized_odds_difference # Métrique clé pour la nouvelle fonctionnalité
)

# --- Scikit-learn Metrics ---
from sklearn.metrics import (
    roc_auc_score,
    roc_curve, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --- CHEMINS ET CONSTANTES ---
RAW_DATA_FILENAME: str = "application_train.csv" 
MODEL_BASELINE_FILENAME: str = "lgbm_baseline.joblib"
BASELINE_THRESHOLD_FILENAME: str = "baseline_threshold.joblib"
MODEL_WRAPPED_EO_FILENAME: str = "eo_wrapper_with_proba.joblib"
WRAPPER_EO_MODULE_FILENAME: str = "wrapper_eo.py"
SK_ID_CURR_COL: str = 'SK_ID_CURR' # Constante pour le nom de la colonne ID client

# Dictionnaire des artefacts à télécharger
ARTEFACTS: Dict[str, str] = {
    RAW_DATA_FILENAME: "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/main/application_train.csv",
    BASELINE_THRESHOLD_FILENAME: "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/main/baseline_threshold.joblib",
    MODEL_WRAPPED_EO_FILENAME: "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/main/eo_wrapper_with_proba.joblib",
    MODEL_BASELINE_FILENAME: "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/main/lgbm_baseline.joblib",
    WRAPPER_EO_MODULE_FILENAME: "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/main/wrapper_eo.py",
    "X_test_pre.parquet": "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/main/X_test_pre.parquet",
    "y_test.parquet": "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/main/y_test.parquet",
    "A_test.parquet": "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/main/A_test.parquet",
    "X_valid_pre.parquet": "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/main/X_valid_pre.parquet", 
    "y_valid.parquet": "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/main/y_valid.parquet", 
}

# Features catégorielles pour l'analyse d'équité détaillée
CATEGORICAL_FEATURES_FOR_FAIRNESS_ANALYSIS: List[str] = [
    'CODE_GENDER', # Feature sensible principale, pour référence
    'NAME_CONTRACT_TYPE',
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY',
    'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE',
    'OCCUPATION_TYPE',
    'ORGANIZATION_TYPE',
    'WEEKDAY_APPR_PROCESS_START'
]

# -- Streamlit config --
st.set_page_config(
    page_title="POC Scoring Équitable (Détaillé)", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- Token Hugging Face pour API privée si besoin --
HF_TOKEN: Optional[str] = st.secrets.get("HF_TOKEN", None)
HEADERS: Dict[str, str] = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def download_if_missing(filename: str, url: str) -> None:
    if not os.path.exists(filename):
        st.info(f"Téléchargement de {filename}...")
        try:
            with requests.get(url, stream=True, headers=HEADERS) as r:
                r.raise_for_status()
                with open(filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            st.success(f"{filename} téléchargé.")
        except Exception as e:
            st.error(f"Erreur lors du téléchargement de {filename}: {e}")
            if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
                st.error(f"Réponse du serveur: {e.response.status_code} - {e.response.text}")
            st.stop()

for fname, url in ARTEFACTS.items():
    download_if_missing(fname, url)

def ensure_eowrapper_in_main(wrapper_file_path: str = WRAPPER_EO_MODULE_FILENAME) -> Optional[type]:
    try:
        temp_mod_name = "eowrapper_dyn_detailed_fairness_feature" 
        spec = importlib.util.spec_from_file_location(temp_mod_name, wrapper_file_path)
        if spec is None or spec.loader is None:
            st.error(f"Impossible de créer la spec pour le module depuis {wrapper_file_path}")
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, "EOWrapper", None)
        if cls is None:
            st.error(f"Classe EOWrapper non trouvée dans {wrapper_file_path}")
            return None
        cls.__module__ = "__main__"
        setattr(sys.modules["__main__"], "EOWrapper", cls)
        from fairlearn.reductions import ExponentiatedGradient 
        setattr(sys.modules["__main__"], "ExponentiatedGradient", ExponentiatedGradient)
        return cls
    except Exception as e:
        st.error(f"Erreur lors du chargement dynamique de EOWrapper: {e}")
        st.exception(e)
        return None

@st.cache_data
def load_parquet_file(path: str, index_col: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Charge un fichier Parquet, avec option de définir une colonne d'index."""
    try:
        df = pd.read_parquet(path)
        if index_col and index_col in df.columns:
            df = df.set_index(index_col)
        return df
    except Exception as e:
        st.error(f"Erreur de chargement du fichier Parquet {path}: {e}")
        return None

@st.cache_data
def load_csv_data(filename: str, usecols: Optional[List[str]] = None, index_col: Optional[str] = None, sample_frac: Optional[float] = None) -> Optional[pd.DataFrame]:
    """Charge des données depuis un fichier CSV, avec options de sélection de colonnes, index et échantillonnage."""
    try:
        df = pd.read_csv(filename, usecols=usecols)
        if index_col and index_col in df.columns:
            df = df.set_index(index_col)
        if sample_frac and 0.0 < sample_frac < 1.0:
            if len(df) * sample_frac >= 1:
                df = df.sample(frac=sample_frac, random_state=42)
        return df
    except FileNotFoundError:
        st.error(f"Fichier CSV non trouvé: {filename}")
        return None
    except Exception as e:
        st.error(f"Erreur de chargement du fichier CSV {filename}: {e}")
        return None

@st.cache_resource
def load_model_joblib(path: str) -> Any:
    st.info(f"Tentative de chargement du modèle depuis {path}...")
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Erreur de chargement du modèle {path}: {e}")
        st.exception(e)
        return None

def sanitize_feature_names(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()
    # Conserver l'index s'il existe et a un nom
    idx_name = df.index.name
    df = df.reset_index() # Pour nettoyer les noms de colonnes y compris l'index s'il est une colonne
    
    cleaned_columns = {}
    for col in df.columns:
        if col == idx_name and idx_name is not None : # Ne pas nettoyer le nom de l'index s'il était déjà nommé
             cleaned_columns[col] = col
        else:
            cleaned_columns[col] = re.sub(r"[^a-zA-Z0-9_]", "_", str(col))
    
    df = df.rename(columns=cleaned_columns)

    if idx_name is not None and idx_name in df.columns : # Rétablir l'index
        df = df.set_index(idx_name)
    elif idx_name is not None and cleaned_columns.get(idx_name) in df.columns: # Si l'index a été nettoyé
         df = df.set_index(cleaned_columns[idx_name])

    return df

# === Chargement des modèles et données ===
model_baseline = load_model_joblib(MODEL_BASELINE_FILENAME)
optimal_thresh_baseline = load_model_joblib(BASELINE_THRESHOLD_FILENAME)
if optimal_thresh_baseline is None:
    st.warning(f"Seuil baseline ('{BASELINE_THRESHOLD_FILENAME}') non trouvé. Fallback à 0.5.")
    optimal_thresh_baseline = 0.5
else:
    st.sidebar.info(f"Seuil optimal baseline : {optimal_thresh_baseline:.3f}")

EOWrapper_class = ensure_eowrapper_in_main()
model_eo_wrapper = None
if EOWrapper_class is not None:
    model_eo_wrapper = load_model_joblib(MODEL_WRAPPED_EO_FILENAME)

if model_baseline: st.sidebar.success("Modèle baseline chargé.")
if model_eo_wrapper:
    st.sidebar.success("EO Wrapper chargé.")
    if hasattr(model_eo_wrapper, 'threshold'):
         st.sidebar.info(f"Seuil EO Wrapper : {model_eo_wrapper.threshold:.4f}")
    else:
        st.sidebar.warning("L'objet EO Wrapper chargé n'a pas d'attribut 'threshold'.")

# Données de test (s'assurer que SK_ID_CURR_COL est l'index)
X_test_raw = load_parquet_file("X_test_pre.parquet", index_col=SK_ID_CURR_COL)
y_test_df = load_parquet_file("y_test.parquet", index_col=SK_ID_CURR_COL)
A_test_df = load_parquet_file("A_test.parquet", index_col=SK_ID_CURR_COL)

X_test, y_test, A_test = None, None, None
if X_test_raw is not None:
    X_test = sanitize_feature_names(X_test_raw) # Sanitize après avoir setté l'index
    st.sidebar.info("Données de test (X_test) chargées et nettoyées.")
if y_test_df is not None:
    y_test = y_test_df.squeeze()
    st.sidebar.info("Données de test (y_test) chargées.")
if A_test_df is not None:
    A_test = A_test_df.squeeze()
    st.sidebar.info("Données de test (A_test) chargées.")

# Données de validation
X_valid_raw = load_parquet_file("X_valid_pre.parquet", index_col=SK_ID_CURR_COL)
y_valid_df = load_parquet_file("y_valid.parquet", index_col=SK_ID_CURR_COL)
X_valid, y_valid = None, None
if X_valid_raw is not None:
    X_valid = sanitize_feature_names(X_valid_raw)
    st.sidebar.info("Données de validation (X_valid) chargées et nettoyées.")
if y_valid_df is not None:
    y_valid = y_valid_df.squeeze()
    st.sidebar.info("Données de validation (y_valid) chargées.")

# Données brutes pour EDA et analyse d'équité par feature
cols_for_fairness_eda = [SK_ID_CURR_COL] + CATEGORICAL_FEATURES_FOR_FAIRNESS_ANALYSIS + ['TARGET', 'AMT_INCOME_TOTAL']
# S'assurer de ne pas avoir de doublons si SK_ID_CURR_COL est déjà dans CATEGORICAL_FEATURES_FOR_FAIRNESS_ANALYSIS
cols_for_fairness_eda = sorted(list(set(cols_for_fairness_eda))) 
df_raw_full_relevant_cols = load_csv_data(RAW_DATA_FILENAME, usecols=cols_for_fairness_eda, index_col=SK_ID_CURR_COL)
df_eda_sample = None
if df_raw_full_relevant_cols is not None:
    df_eda_sample = df_raw_full_relevant_cols.sample(frac=0.05, random_state=42) if len(df_raw_full_relevant_cols) * 0.05 >=1 else df_raw_full_relevant_cols
    st.sidebar.info("Données brutes (colonnes pertinentes) et échantillon EDA chargés.")


# === Fonctions métriques ===
def compute_classification_metrics(y_true, y_pred_hard, y_pred_proba_positive_class):
    metrics = {}
    try:
        metrics["AUC"] = roc_auc_score(y_true, y_pred_proba_positive_class)
        metrics["Accuracy"] = accuracy_score(y_true, y_pred_hard)
        metrics["Precision (1)"] = precision_score(y_true, y_pred_hard, pos_label=1, zero_division=0)
        metrics["Recall (1)"] = recall_score(y_true, y_pred_hard, pos_label=1, zero_division=0)
        metrics["F1 (1)"] = f1_score(y_true, y_pred_hard, pos_label=1, zero_division=0)
        metrics["Taux de sélection global"] = np.mean(y_pred_hard) 
    except Exception as e:
        st.warning(f"Erreur calcul métriques classification: {e}")
        for k in ["AUC", "Accuracy", "Precision (1)", "Recall (1)", "F1 (1)", "Taux de sélection global"]:
            metrics.setdefault(k, np.nan)
    return metrics

def compute_fairness_metrics(y_true, y_pred_hard, sensitive_features):
    metrics = {}
    try:
        metrics["DPD"] = demographic_parity_difference(y_true, y_pred_hard, sensitive_features=sensitive_features)
        metrics["EOD"] = equalized_odds_difference(y_true, y_pred_hard, sensitive_features=sensitive_features)
    except Exception as e:
        st.warning(f"Erreur calcul métriques d'équité: {e}")
        metrics.setdefault("DPD", np.nan)
        metrics.setdefault("EOD", np.nan)
    return metrics

# === Sidebar navigation ===
st.sidebar.title("📊 POC Scoring Équitable")
page_options: List[str] = [
    "Analyse Exploratoire (EDA)",
    "Résultats & Comparaisons",
    "Prédiction sur Client Sélectionné",
    "Courbes ROC & Probabilités - Baseline",
    "Courbes ROC & Probabilités - EO Wrapper"
]
default_page_index: int = 0
session_key_page_index = "current_page_index_poc_scoring_detailed_fairness_feature" 
if session_key_page_index not in st.session_state:
    st.session_state[session_key_page_index] = default_page_index
page: str = st.sidebar.radio(
    "Navigation", page_options, index=st.session_state[session_key_page_index],
    key="nav_radio_poc_scoring_detailed_fairness_feature" 
)
if page_options.index(page) != st.session_state[session_key_page_index]:
    st.session_state[session_key_page_index] = page_options.index(page)
    st.rerun()

# === Contenu des Pages ===
if page == "Analyse Exploratoire (EDA)":
    st.header("🔎 Analyse Exploratoire des Données (EDA)")
    if df_eda_sample is not None and not df_eda_sample.empty:
        st.caption(f"Basée sur un échantillon de {len(df_eda_sample)} lignes du fichier `{RAW_DATA_FILENAME}`.")
        st.subheader("Aperçu des données (échantillon)")
        st.dataframe(df_eda_sample.head(), use_container_width=True)

        st.subheader("Statistiques descriptives (variables numériques de l'échantillon)")
        st.dataframe(df_eda_sample.describe(include=np.number).T, use_container_width=True)
        
        if "TARGET" in df_eda_sample.columns:
            st.subheader("Distribution de la variable cible 'TARGET' (échantillon)")
            # ... (code EDA existant) ...
            target_counts = df_eda_sample["TARGET"].value_counts()
            target_counts_percent = df_eda_sample["TARGET"].value_counts(normalize=True) * 100
            col1, col2 = st.columns(2)
            with col1: st.dataframe(target_counts.rename("Comptage Absolu"))
            with col2: st.dataframe(target_counts_percent.map("{:.2f}%".format).rename("Pourcentage"))
            try:
                fig_target_hist = px.histogram(df_eda_sample, x="TARGET", color="TARGET", title="Histogramme de 'TARGET'", labels={"TARGET": "Classe de défaut"}, text_auto=True)
                fig_target_hist.update_layout(bargap=0.2) 
                st.plotly_chart(fig_target_hist, use_container_width=True)
            except Exception as e: st.warning(f"Erreur histogramme TARGET: {e}")
        else: st.warning("Colonne 'TARGET' non trouvée dans l'échantillon EDA.")

        numerical_col_for_eda = 'AMT_INCOME_TOTAL' 
        if numerical_col_for_eda in df_eda_sample.columns:
            st.subheader(f"Distribution de '{numerical_col_for_eda}' (échantillon)")
            # ... (code EDA existant pour AMT_INCOME_TOTAL) ...
            df_positive_income = df_eda_sample[df_eda_sample[numerical_col_for_eda] > 0]
            income_cap_val = 0
            if not df_positive_income.empty:
                income_cap_val = df_positive_income[numerical_col_for_eda].quantile(0.99)
                df_filtered_income = df_positive_income[df_positive_income[numerical_col_for_eda] < income_cap_val]
            else: 
                df_filtered_income = df_eda_sample
                if not df_eda_sample.empty : income_cap_val = df_eda_sample[numerical_col_for_eda].max()
            try:
                fig_income_dist = px.histogram(df_filtered_income, x=numerical_col_for_eda, color="TARGET" if "TARGET" in df_filtered_income else None, marginal="box", title=f"Distribution de '{numerical_col_for_eda}' (plafonné à {income_cap_val:,.0f})", labels={numerical_col_for_eda: "Revenu total", "TARGET": "Classe de défaut"})
                st.plotly_chart(fig_income_dist, use_container_width=True)
            except Exception as e: st.warning(f"Erreur histogramme {numerical_col_for_eda}: {e}")
        else: st.info(f"Colonne '{numerical_col_for_eda}' non trouvée dans l'échantillon EDA.")
    else: st.error("Échantillon de données pour l'EDA non chargé.")

elif page == "Résultats & Comparaisons":
    st.header("📊 Résultats comparatifs sur jeu de test")
    # ... (début du code de la page Résultats & Comparaisons existant) ...
    if model_baseline is not None and model_eo_wrapper is not None and \
       X_test is not None and y_test is not None and A_test is not None and \
       optimal_thresh_baseline is not None:
        try:
            y_test_proba_b: np.ndarray = model_baseline.predict_proba(X_test)[:, 1]
            y_test_pred_b: np.ndarray = (y_test_proba_b >= optimal_thresh_baseline).astype(int)
            metrics_b: Dict[str, float] = compute_classification_metrics(y_test, y_test_pred_b, y_test_proba_b)
            fairness_b: Dict[str, float] = compute_fairness_metrics(y_test, y_test_pred_b, A_test) # A_test est la feature sensible principale
            cm_b = confusion_matrix(y_test, y_test_pred_b)
            
            y_test_proba_eo: np.ndarray = model_eo_wrapper.predict_proba(X_test)[:, 1]
            y_test_pred_eo: np.ndarray = model_eo_wrapper.predict(X_test) 
            metrics_eo: Dict[str, float] = compute_classification_metrics(y_test, y_test_pred_eo, y_test_proba_eo)
            fairness_eo: Dict[str, float] = compute_fairness_metrics(y_test, y_test_pred_eo, A_test) # A_test est la feature sensible principale
            cm_eo = confusion_matrix(y_test, y_test_pred_eo)

            st.subheader("Tableau récapitulatif des métriques globales")
            df_res = pd.DataFrame([
                {"Modèle": "Baseline", **metrics_b, **fairness_b},
                {"Modèle": "EO Wrapper", **metrics_eo, **fairness_eo}
            ])
            st.dataframe(df_res.set_index("Modèle").style.format("{:.3f}", na_rep="-"), use_container_width=True)
            
            st.subheader("Matrices de Confusion (sur feature sensible principale)")
            # ... (code matrices de confusion existant) ...
            col1_cm, col2_cm = st.columns(2)
            labels_cm = ['Non-Défaut (0)', 'Défaut (1)']
            with col1_cm:
                st.markdown("**Modèle Baseline**")
                z_text_b = [[str(y) for y in x] for x in cm_b]
                fig_cm_b = ff.create_annotated_heatmap(cm_b, x=labels_cm, y=labels_cm, annotation_text=z_text_b, colorscale='Blues')
                fig_cm_b.update_layout(title_text="<i>Baseline</i>", xaxis_title="Prédit", yaxis_title="Réel")
                st.plotly_chart(fig_cm_b, use_container_width=True)
            with col2_cm:
                st.markdown("**Modèle EO Wrapper**")
                z_text_eo = [[str(y) for y in x] for x in cm_eo]
                fig_cm_eo = ff.create_annotated_heatmap(cm_eo, x=labels_cm, y=labels_cm, annotation_text=z_text_eo, colorscale='Greens')
                fig_cm_eo.update_layout(title_text="<i>EO Wrapper</i>", xaxis_title="Prédit", yaxis_title="Réel")
                st.plotly_chart(fig_cm_eo, use_container_width=True)

            st.subheader("Taux de Sélection par Groupe (sur feature sensible principale `A_test`)")
            # ... (code taux de sélection existant basé sur A_test) ...
            mf_selection_baseline = MetricFrame(metrics=fairlearn_selection_rate, y_true=y_test, y_pred=y_test_pred_b, sensitive_features=A_test)
            mf_selection_eo = MetricFrame(metrics=fairlearn_selection_rate, y_true=y_test, y_pred=y_test_pred_eo, sensitive_features=A_test)
            df_selection_rates = pd.DataFrame({"Groupe Sensible": mf_selection_baseline.by_group.index, "Taux Sélection Baseline": mf_selection_baseline.by_group.values, "Taux Sélection EO Wrapper": mf_selection_eo.by_group.values}).set_index("Groupe Sensible")
            st.dataframe(df_selection_rates.style.format("{:.3f}"), use_container_width=True)
            df_selection_plot = df_selection_rates.reset_index().melt(id_vars="Groupe Sensible", value_vars=["Taux Sélection Baseline", "Taux Sélection EO Wrapper"], var_name="Modèle", value_name="Taux de Sélection")
            fig_sr = px.bar(df_selection_plot, x="Groupe Sensible", y="Taux de Sélection", color="Modèle", barmode="group", title="Taux de Sélection (feature sensible principale)", labels={"Groupe Sensible": "Groupe (Feature A_test)", "Taux de Sélection": "Taux d'approbation"})
            st.plotly_chart(fig_sr, use_container_width=True)

            # --- NOUVELLE SECTION : Analyse d'équité par feature catégorielle sélectionnée ---
            st.divider()
            st.subheader("Analyse d'Équité Détaillée par Feature Catégorielle")
            
            # S'assurer que df_raw_full_relevant_cols est chargé et contient les features
            if df_raw_full_relevant_cols is not None:
                # Filtrer les features qui sont réellement dans les colonnes chargées pour le dropdown
                available_features_for_dropdown = [f for f in CATEGORICAL_FEATURES_FOR_FAIRNESS_ANALYSIS if f in df_raw_full_relevant_cols.columns]
                
                if not available_features_for_dropdown:
                    st.warning("Aucune des features catégorielles spécifiées n'a été trouvée dans les données chargées.")
                else:
                    selected_feature_for_fairness = st.selectbox(
                        "Choisissez une feature pour analyser l'équité (EOD) par groupe :",
                        options=available_features_for_dropdown,
                        index=0 
                    )

                    if selected_feature_for_fairness and X_test is not None and y_test is not None:
                        # Utiliser df_raw_full_relevant_cols qui est déjà indexé par SK_ID_CURR_COL
                        # et contient la selected_feature_for_fairness
                        
                        # Aligner avec X_test.index (qui est SK_ID_CURR_COL)
                        common_indices = X_test.index.intersection(df_raw_full_relevant_cols.index)
                        
                        if not common_indices.empty:
                            aligned_y_test = y_test.loc[common_indices]
                            
                            # Convertir les prédictions numpy en Series pandas avec l'index correct pour l'alignement
                            y_test_pred_b_series = pd.Series(y_test_pred_b, index=X_test.index)
                            y_test_pred_eo_series = pd.Series(y_test_pred_eo, index=X_test.index)

                            aligned_pred_b = y_test_pred_b_series.loc[common_indices].values
                            aligned_pred_eo = y_test_pred_eo_series.loc[common_indices].values
                            
                            aligned_sensitive_feature = df_raw_full_relevant_cols.loc[common_indices, selected_feature_for_fairness]
                            aligned_sensitive_feature = aligned_sensitive_feature.fillna("Manquant").astype(str)

                            st.markdown(f"**Equalized Odds Difference (EOD) pour les groupes de '{selected_feature_for_fairness}'**")

                            mf_eod_baseline = MetricFrame(metrics=equalized_odds_difference,
                                                          y_true=aligned_y_test, y_pred=aligned_pred_b,
                                                          sensitive_features=aligned_sensitive_feature)
                            mf_eod_eo = MetricFrame(metrics=equalized_odds_difference,
                                                    y_true=aligned_y_test, y_pred=aligned_pred_eo,
                                                    sensitive_features=aligned_sensitive_feature)

                            df_eod_by_group = pd.DataFrame({
                                f"EOD Baseline": mf_eod_baseline.by_group,
                                f"EOD EO Wrapper": mf_eod_eo.by_group
                            })
                            df_eod_by_group.index.name = f"Groupe ({selected_feature_for_fairness})"
                            st.dataframe(df_eod_by_group.style.format("{:.3f}"), use_container_width=True)

                            df_eod_plot = df_eod_by_group.reset_index().melt(
                                id_vars=f"Groupe ({selected_feature_for_fairness})",
                                var_name="Modèle", value_name="EOD"
                            )
                            fig_eod = px.bar(df_eod_plot, x=f"Groupe ({selected_feature_for_fairness})", y="EOD",
                                             color="Modèle", barmode="group",
                                             title=f"EOD par groupe de '{selected_feature_for_fairness}'")
                            st.plotly_chart(fig_eod, use_container_width=True)
                        else:
                            st.warning(f"Aucun client commun trouvé entre X_test et les données brutes pour la feature '{selected_feature_for_fairness}'.")
            else:
                st.warning("Données brutes pertinentes non chargées, impossible d'effectuer l'analyse d'équité par feature.")

        except Exception as e:
            st.error(f"Erreur lors du calcul ou de l'affichage des résultats comparatifs: {e}")
            st.exception(e)
    else:
        st.warning("Des éléments sont manquants pour afficher les résultats. Vérifiez les messages dans la barre latérale.")
        # ... (messages d'erreur existants) ...

elif page == "Prédiction sur Client Sélectionné":
    # ... (code existant de la page Prédiction) ...
    st.header("🔍 Prédiction sur un Client Sélectionné du Jeu de Test")
    if X_test is not None and model_baseline is not None and model_eo_wrapper is not None and optimal_thresh_baseline is not None and y_test is not None:
        client_ids = X_test.index.tolist()
        if not client_ids: st.warning("Aucun ID client disponible dans le jeu de test.")
        else:
            max_clients_in_selectbox = 2000 
            client_ids_to_display = client_ids[:max_clients_in_selectbox] if len(client_ids) > max_clients_in_selectbox else client_ids
            if len(client_ids) > max_clients_in_selectbox: st.info(f"Affichage des {max_clients_in_selectbox} premiers IDs clients.")
            selected_client_id_str = st.selectbox("Choisissez un ID client:", options=[str(id_val) for id_val in client_ids_to_display])
            selected_client_id: Any
            try:
                if X_test.index.dtype == 'int64' or X_test.index.dtype == 'int32': selected_client_id = int(selected_client_id_str)
                elif X_test.index.dtype == 'float64' or X_test.index.dtype == 'float32': selected_client_id = float(selected_client_id_str)
                else: selected_client_id = selected_client_id_str
            except ValueError: 
                st.error(f"ID client '{selected_client_id_str}' invalide."); st.stop()
            if selected_client_id in X_test.index:
                client_features = X_test.loc[[selected_client_id]] 
                client_true_target = y_test.loc[selected_client_id] if selected_client_id in y_test.index else "Inconnue"
                st.subheader(f"Données du client ID: {selected_client_id}")
                st.write(f"Vraie cible (TARGET) : **{client_true_target}**")
                st.dataframe(client_features.T.rename(columns={0: "Valeur"}), use_container_width=True)
                try:
                    proba_baseline = model_baseline.predict_proba(client_features)[0, 1]
                    pred_baseline = (proba_baseline >= optimal_thresh_baseline).astype(int)
                    proba_eo = model_eo_wrapper.predict_proba(client_features)[0, 1]
                    pred_eo = model_eo_wrapper.predict(client_features)[0]
                    st.subheader("Résultats de la Prédiction")
                    results_data = {"Métrique": ["Probabilité de défaut (classe 1)", "Prédiction (0 ou 1)"], "Modèle Baseline": [f"{proba_baseline:.4f}", pred_baseline], "Modèle EO Wrapper": [f"{proba_eo:.4f}", pred_eo]}
                    df_pred_results = pd.DataFrame(results_data)
                    st.table(df_pred_results.set_index("Métrique")) 
                except Exception as e_pred: st.error(f"Erreur prédiction client {selected_client_id}: {e_pred}")
            else: st.error(f"ID client {selected_client_id} non trouvé après conversion.")
    else: st.warning("Éléments manquants pour prédiction: modèles ou données de test.")


elif page == "Courbes ROC & Probabilités - Baseline":
    # ... (code existant de la page ROC Baseline) ...
    st.header("Courbes ROC & Distribution des Probabilités - Modèle Baseline")
    st.caption("Calculé sur le jeu de validation.")
    if model_baseline is not None and X_valid is not None and y_valid is not None and optimal_thresh_baseline is not None:
        try:
            y_val_proba_b: np.ndarray = model_baseline.predict_proba(X_valid)[:, 1]
            fpr, tpr, thresholds_roc = roc_curve(y_valid, y_val_proba_b)
            auc_b_val: float = roc_auc_score(y_valid, y_val_proba_b)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Baseline (AUC={auc_b_val:.3f})', line=dict(color='blue')))
            fig_roc.add_shape(type='line', x0=0, x1=1, y0=0, y1=1, line=dict(dash='dash', color='grey'))
            optimal_idx_val = np.argmin(np.abs(thresholds_roc - optimal_thresh_baseline))
            fig_roc.add_trace(go.Scatter(x=[fpr[optimal_idx_val]], y=[tpr[optimal_idx_val]], mode='markers', marker=dict(size=10, color='red'), name=f'Seuil Optimal ({optimal_thresh_baseline:.3f})'))
            fig_roc.update_layout(title_text="Courbe ROC - Modèle Baseline (Validation)", xaxis_title="Taux de Faux Positifs (FPR)", yaxis_title="Taux de Vrais Positifs (TPR)", legend_title_text="Légende")
            st.plotly_chart(fig_roc, use_container_width=True)
            st.subheader("Distribution des Scores de Probabilité (Baseline, Validation)")
            df_dist_b = pd.DataFrame({"proba_baseline": y_val_proba_b, "y_true": y_valid.astype(str)}) 
            fig_dist_b = px.histogram(df_dist_b, x="proba_baseline", color="y_true", nbins=50, barmode='overlay', marginal="rug", title="Distribution des Scores (Baseline)", labels={"proba_baseline": "Score Prédit (Baseline)", "y_true": "Vraie Cible"}, color_discrete_map={'0': 'green', '1': 'red'})
            fig_dist_b.add_vline(x=optimal_thresh_baseline, line_color="black", line_dash="dash", annotation_text=f"Seuil={optimal_thresh_baseline:.3f}", annotation_position="top right")
            st.plotly_chart(fig_dist_b, use_container_width=True)
        except Exception as e: st.error(f"Erreur graphiques Baseline (Validation): {e}"); st.exception(e)
    else: st.warning("Modèle Baseline, données de validation ou seuil non chargés.")

elif page == "Courbes ROC & Probabilités - EO Wrapper":
    # ... (code existant de la page ROC EO Wrapper) ...
    st.header("Courbes ROC & Distribution des Probabilités - Modèle EO Wrapper")
    st.caption("Calculé sur le jeu de validation.")
    if model_eo_wrapper is not None and hasattr(model_eo_wrapper, 'threshold') and X_valid is not None and y_valid is not None:
        try:
            wrapper_threshold = model_eo_wrapper.threshold
            y_val_proba_eo: np.ndarray = model_eo_wrapper.predict_proba(X_valid)[:, 1]
            fpr_eo, tpr_eo, thresholds_roc_eo = roc_curve(y_valid, y_val_proba_eo)
            auc_eo_val: float = roc_auc_score(y_valid, y_val_proba_eo)
            fig_roc_eo = go.Figure()
            fig_roc_eo.add_trace(go.Scatter(x=fpr_eo, y=tpr_eo, mode='lines', name=f'ROC EO Wrapper (AUC={auc_eo_val:.3f})', line=dict(color='green')))
            fig_roc_eo.add_shape(type='line', x0=0, x1=1, y0=0, y1=1, line=dict(dash='dash', color='grey'))
            optimal_idx_eo_val = np.argmin(np.abs(thresholds_roc_eo - wrapper_threshold))
            fig_roc_eo.add_trace(go.Scatter(x=[fpr_eo[optimal_idx_eo_val]], y=[tpr_eo[optimal_idx_eo_val]], mode='markers', marker=dict(size=10, color='orange'), name=f'Seuil du Wrapper ({wrapper_threshold:.3f})'))
            fig_roc_eo.update_layout(title_text="Courbe ROC - Modèle EO Wrapper (Validation)", xaxis_title="Taux de Faux Positifs (FPR)", yaxis_title="Taux de Vrais Positifs (TPR)", legend_title_text="Légende")
            st.plotly_chart(fig_roc_eo, use_container_width=True)
            st.subheader("Distribution des Scores de Probabilité (EO Wrapper, Validation)")
            df_dist_eo = pd.DataFrame({"proba_eo": y_val_proba_eo, "y_true": y_valid.astype(str)}) 
            fig_dist_eo = px.histogram(df_dist_eo, x="proba_eo", color="y_true", nbins=50, barmode='overlay', marginal="rug", title="Distribution des Scores (EO Wrapper)", labels={"proba_eo": "Score Prédit (EO Wrapper)", "y_true": "Vraie Cible"}, color_discrete_map={'0': 'green', '1': 'red'})
            fig_dist_eo.add_vline(x=wrapper_threshold, line_color="black", line_dash="dash", annotation_text=f"Seuil={wrapper_threshold:.3f}", annotation_position="top right")
            st.plotly_chart(fig_dist_eo, use_container_width=True)
        except Exception as e: st.error(f"Erreur graphiques EO Wrapper (Validation): {e}"); st.exception(e)
    else: st.warning("Modèle EO Wrapper, son seuil, ou données de validation non chargés.")

st.sidebar.markdown("---")
st.sidebar.caption("Version avec EDA, résultats détaillés (incl. équité par feature), prédiction client et courbes ROC.")

