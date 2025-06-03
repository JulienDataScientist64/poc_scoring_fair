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

# --- Fairness Libraries ---
# ExponentiatedGradient n'est importé que si EOWrapper en dépend réellement à la désérialisation
# from fairlearn.reductions import ExponentiatedGradient 
from fairlearn.metrics import (
    MetricFrame,
    selection_rate as fairlearn_selection_rate,
    demographic_parity_difference,
    equalized_odds_difference
)

# --- Scikit-learn Metrics ---
from sklearn.metrics import (
    roc_auc_score,
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
}

# Colonnes nécessaires du fichier brut pour l'analyse d'intersectionnalité (Optimisation RAM)
INTERSECTIONALITY_COLUMNS: List[str] = ['SK_ID_CURR', 'DAYS_BIRTH', 'CODE_GENDER']

# -- Streamlit config --
st.set_page_config(
    page_title="POC Scoring Équitable (Simplifié)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- Token Hugging Face pour API privée si besoin --
HF_TOKEN: Optional[str] = st.secrets.get("HF_TOKEN", None)
HEADERS: Dict[str, str] = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def download_if_missing(filename: str, url: str) -> None:
    """Télécharge le fichier depuis Hugging Face si absent localement."""
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
    """Charge dynamiquement EOWrapper et l'injecte dans __main__."""
    try:
        temp_mod_name = "eowrapper_dyn_simplified"
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
        # Importer ExponentiatedGradient ici si EOWrapper en dépend pour la désérialisation
        # Cela peut être nécessaire si EOWrapper hérite de ExponentiatedGradient ou l'utilise d'une manière
        # qui est capturée par pickle.
        from fairlearn.reductions import ExponentiatedGradient # Assurer sa présence pour joblib
        setattr(sys.modules["__main__"], "ExponentiatedGradient", ExponentiatedGradient)

        return cls
    except Exception as e:
        st.error(f"Erreur lors du chargement dynamique de EOWrapper: {e}")
        st.exception(e)
        return None

@st.cache_data
def load_parquet_file(path: str) -> Optional[pd.DataFrame]:
    """Charge un fichier Parquet."""
    try:
        return pd.read_parquet(path)
    except Exception as e:
        st.error(f"Erreur de chargement du fichier Parquet {path}: {e}")
        return None

@st.cache_data
def load_raw_data_for_intersectionality(filename: str, columns_to_load: List[str]) -> Optional[pd.DataFrame]:
    """Charge des colonnes spécifiques d'un fichier CSV pour l'intersectionnalité."""
    try:
        df = pd.read_csv(filename, usecols=columns_to_load)
        if 'SK_ID_CURR' in columns_to_load:
            df = df.set_index('SK_ID_CURR')
        return df
    except Exception as e:
        st.error(f"Erreur de chargement des données brutes {filename} pour intersectionnalité: {e}")
        return None

@st.cache_resource
def load_model_joblib(path: str) -> Any:
    """Charge un modèle sauvegardé avec joblib."""
    st.info(f"Tentative de chargement du modèle depuis {path}...")
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Erreur de chargement du modèle {path}: {e}")
        st.exception(e) # Affiche la trace complète pour le débogage
        return None # Retourne None pour permettre à l'app de continuer avec un message d'erreur

def sanitize_feature_names(df_input: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les noms de colonnes."""
    df = df_input.copy()
    cleaned_columns = [re.sub(r"[^a-zA-Z0-9_]", "_", str(col)) for col in df.columns]
    df.columns = cleaned_columns
    return df

# === Chargement des modèles et données ===
# Modèles
model_baseline = load_model_joblib(MODEL_BASELINE_FILENAME)
optimal_thresh_baseline = load_model_joblib(BASELINE_THRESHOLD_FILENAME)
if optimal_thresh_baseline is None:
    st.warning(f"Seuil baseline ('{BASELINE_THRESHOLD_FILENAME}') non trouvé ou erreur. Fallback à 0.5.")
    optimal_thresh_baseline = 0.5
else:
    st.sidebar.info(f"Seuil optimal baseline : {optimal_thresh_baseline:.3f}")

EOWrapper_class = ensure_eowrapper_in_main()
model_eo_wrapper = None
if EOWrapper_class is not None:
    model_eo_wrapper = load_model_joblib(MODEL_WRAPPED_EO_FILENAME)

if model_baseline:
    st.sidebar.success("Modèle baseline chargé.")
if model_eo_wrapper:
    st.sidebar.success("EO Wrapper chargé.")
    if hasattr(model_eo_wrapper, 'threshold'):
         st.sidebar.info(f"Seuil EO Wrapper : {model_eo_wrapper.threshold:.4f}")
    else:
        st.sidebar.warning("L'objet EO Wrapper chargé n'a pas d'attribut 'threshold'.")


# Données de test
X_test_raw = load_parquet_file("X_test_pre.parquet")
y_test = load_parquet_file("y_test.parquet")
A_test = load_parquet_file("A_test.parquet")

X_test = None
if X_test_raw is not None:
    X_test = sanitize_feature_names(X_test_raw)
    st.sidebar.info("Données de test (X_test) nettoyées.")
if y_test is not None:
    y_test = y_test.squeeze()
    st.sidebar.info("Données de test (y_test) chargées.")
if A_test is not None:
    A_test = A_test.squeeze()
    st.sidebar.info("Données de test (A_test) chargées.")

# Données pour intersectionnalité
df_raw_for_intersection = load_raw_data_for_intersectionality(RAW_DATA_FILENAME, INTERSECTIONALITY_COLUMNS)
if df_raw_for_intersection is not None:
    st.sidebar.info("Données brutes pour intersectionnalité chargées.")


# === Fonctions métriques ===
def compute_classification_metrics(
    y_true: pd.Series, 
    y_pred_hard: np.ndarray, 
    y_pred_proba_positive_class: np.ndarray
) -> Dict[str, float]:
    """Calcule les métriques de classification standard."""
    metrics = {}
    try:
        metrics["AUC"] = roc_auc_score(y_true, y_pred_proba_positive_class)
        metrics["Accuracy"] = accuracy_score(y_true, y_pred_hard)
        metrics["Precision (1)"] = precision_score(y_true, y_pred_hard, pos_label=1, zero_division=0)
        metrics["Recall (1)"] = recall_score(y_true, y_pred_hard, pos_label=1, zero_division=0)
        metrics["F1 (1)"] = f1_score(y_true, y_pred_hard, pos_label=1, zero_division=0)
        metrics["Taux de sélection"] = np.mean(y_pred_hard)
    except Exception as e:
        st.warning(f"Erreur calcul métriques classification: {e}")
        # Initialiser avec NaN si erreur
        for k in ["AUC", "Accuracy", "Precision (1)", "Recall (1)", "F1 (1)", "Taux de sélection"]:
            metrics.setdefault(k, np.nan)
    return metrics

def compute_fairness_metrics(
    y_true: pd.Series, 
    y_pred_hard: np.ndarray, 
    sensitive_features: pd.Series
) -> Dict[str, float]:
    """Calcule les métriques d'équité (DPD, EOD)."""
    metrics = {}
    try:
        metrics["DPD"] = demographic_parity_difference(y_true, y_pred_hard, sensitive_features=sensitive_features)
        metrics["EOD"] = equalized_odds_difference(y_true, y_pred_hard, sensitive_features=sensitive_features)
    except Exception as e:
        st.warning(f"Erreur calcul métriques d'équité: {e}")
        metrics.setdefault("DPD", np.nan)
        metrics.setdefault("EOD", np.nan)
    return metrics

# === Sidebar navigation (simplifiée) ===
st.sidebar.title("📊 POC Scoring Équitable (Simplifié)")
page_options: List[str] = [
    "Résultats & Comparaisons",
    "Intersectionnalité"
]
default_page_index: int = 0

if 'current_page_index_simplified' not in st.session_state: # Clé de session unique
    st.session_state.current_page_index_simplified = default_page_index

page: str = st.sidebar.radio(
    "Navigation",
    page_options,
    index=st.session_state.current_page_index_simplified,
    key="nav_radio_simplified" # Clé de widget unique
)
if page_options.index(page) != st.session_state.current_page_index_simplified:
    st.session_state.current_page_index_simplified = page_options.index(page)
    st.rerun()

# === Contenu des Pages ===
if page == "Résultats & Comparaisons":
    st.header("📊 Résultats comparatifs sur jeu de test")
    if model_baseline is not None and model_eo_wrapper is not None and \
       X_test is not None and y_test is not None and A_test is not None and \
       optimal_thresh_baseline is not None:
        try:
            y_test_proba_b: np.ndarray = model_baseline.predict_proba(X_test)[:, 1]
            y_test_pred_b: np.ndarray = (y_test_proba_b >= optimal_thresh_baseline).astype(int)
            metrics_b: Dict[str, float] = compute_classification_metrics(y_test, y_test_pred_b, y_test_proba_b)
            fairness_b: Dict[str, float] = compute_fairness_metrics(y_test, y_test_pred_b, A_test)

            y_test_proba_eo: np.ndarray = model_eo_wrapper.predict_proba(X_test)[:, 1]
            y_test_pred_eo: np.ndarray = model_eo_wrapper.predict(X_test) 
            metrics_eo: Dict[str, float] = compute_classification_metrics(y_test, y_test_pred_eo, y_test_proba_eo)
            fairness_eo: Dict[str, float] = compute_fairness_metrics(y_test, y_test_pred_eo, A_test)

            df_res = pd.DataFrame([
                {"Modèle": "Baseline", **metrics_b, **fairness_b},
                {"Modèle": "EO Wrapper", **metrics_eo, **fairness_eo}
            ])
            st.dataframe(df_res.set_index("Modèle").style.format("{:.3f}", na_rep="-"), use_container_width=True)
        except Exception as e:
            st.error(f"Erreur lors du calcul ou de l'affichage des résultats comparatifs: {e}")
            st.exception(e)
    else:
        st.warning("Des éléments sont manquants pour afficher les résultats : modèles, données de test ou seuil baseline.")
        if model_baseline is None: st.warning("- Modèle baseline non chargé.")
        if model_eo_wrapper is None: st.warning("- Modèle EO wrapper non chargé.")
        if X_test is None: st.warning("- Données X_test non chargées.")
        if y_test is None: st.warning("- Données y_test non chargées.")
        if A_test is None: st.warning("- Données A_test non chargées.")
        if optimal_thresh_baseline is None: st.warning("- Seuil optimal baseline non chargé.")


elif page == "Intersectionnalité":
    st.header("Équité intersectionnelle (Genre & Tranche d'âge)")
    st.caption("Analyse sur le jeu de test.")

    if df_raw_for_intersection is not None and not df_raw_for_intersection.empty and \
       model_baseline is not None and model_eo_wrapper is not None and hasattr(model_eo_wrapper, 'threshold') and \
       X_test is not None and y_test is not None and optimal_thresh_baseline is not None:

        if "DAYS_BIRTH" in df_raw_for_intersection.columns and "CODE_GENDER" in df_raw_for_intersection.columns:
            common_indices = X_test.index.intersection(df_raw_for_intersection.index)

            if not common_indices.empty:
                X_test_aligned = X_test.loc[common_indices]
                y_test_aligned = y_test.loc[common_indices]
                
                df_aligned_raw_data = df_raw_for_intersection.loc[common_indices]

                df_analysis_inter = pd.DataFrame(index=common_indices)
                df_analysis_inter['AGE_YEARS'] = -df_aligned_raw_data["DAYS_BIRTH"] / 365.0 
                df_analysis_inter['AGE_BIN'] = pd.cut(
                    df_analysis_inter["AGE_YEARS"],
                    bins=[17, 25, 35, 45, 55, 65, 100], 
                    labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
                    right=True 
                ).astype(str) # Convertir en str pour éviter problèmes avec NaN potentiels
                df_analysis_inter['CODE_GENDER'] = df_aligned_raw_data["CODE_GENDER"].astype(str)
                
                sensitive_group_inter = (
                    df_analysis_inter["CODE_GENDER"] +
                    " | " +
                    df_analysis_inter["AGE_BIN"]
                )
                sensitive_group_inter.name = "Groupe_Intersectionnel"
                
                # Remplacer les NaN dans sensitive_group_inter par une catégorie 'Inconnu'
                # pour éviter les erreurs dans MetricFrame si des AGE_BIN sont NaN
                sensitive_group_inter = sensitive_group_inter.fillna("Inconnu | Inconnu")


                y_pred_b_inter: np.ndarray = (model_baseline.predict_proba(X_test_aligned)[:, 1] >= optimal_thresh_baseline).astype(int)
                y_pred_eo_inter: np.ndarray = model_eo_wrapper.predict(X_test_aligned)

                metrics_to_compute: Dict[str, Any] = {
                    "Taux de Sélection": fairlearn_selection_rate,
                    "Rappel (Recall)": lambda yt, yp: recall_score(yt, yp, pos_label=1, zero_division=0),
                    "Précision (Precision)": lambda yt, yp: precision_score(yt, yp, pos_label=1, zero_division=0)
                }
                try:
                    mf_baseline = MetricFrame(
                        metrics=metrics_to_compute, y_true=y_test_aligned,
                        y_pred=y_pred_b_inter, sensitive_features=sensitive_group_inter
                    )
                    mf_eo = MetricFrame(
                        metrics=metrics_to_compute, y_true=y_test_aligned,
                        y_pred=y_pred_eo_inter, sensitive_features=sensitive_group_inter
                    )

                    df_plot = pd.concat([
                        mf_baseline.by_group.rename(columns=lambda c: f"{c} - Baseline"),
                        mf_eo.by_group.rename(columns=lambda c: f"{c} - EO Wrapper")
                    ], axis=1).reset_index()

                    st.dataframe(df_plot.style.format("{:.3f}", na_rep="-"), use_container_width=True)

                    for metric_base_name in metrics_to_compute.keys():
                        col_baseline = f"{metric_base_name} - Baseline"
                        col_eo = f"{metric_base_name} - EO Wrapper"
                        if col_baseline in df_plot.columns and col_eo in df_plot.columns:
                            fig = px.bar(df_plot, y="Groupe_Intersectionnel", x=[col_baseline, col_eo],
                                         barmode="group", title=f"{metric_base_name} par groupe (Baseline vs EO)",
                                         color_discrete_sequence=px.colors.qualitative.Plotly)
                            fig.update_layout(yaxis={'categoryorder':'total ascending'}) 
                            st.plotly_chart(fig, use_container_width=True)
                except ValueError as ve:
                    st.error(f"Erreur lors du calcul des métriques d'intersectionnalité avec MetricFrame: {ve}")
                    st.write("Cela peut être dû à des valeurs NaN dans les features sensibles ou des groupes vides.")
                    st.write("Groupes sensibles uniques:", sensitive_group_inter.unique())

            else:
                st.warning("Aucun indice commun trouvé entre X_test et les données brutes pour l'intersectionnalité.")
        else:
            st.warning("Colonnes 'DAYS_BIRTH' ou 'CODE_GENDER' manquantes dans les données chargées pour l'analyse intersectionnelle.")
    else:
        st.info("Données ou modèles non prêts pour l'analyse intersectionnelle.")
        if df_raw_for_intersection is None or df_raw_for_intersection.empty : st.warning("- Données brutes pour intersectionnalité non chargées.")
        if model_baseline is None: st.warning("- Modèle baseline non chargé.")
        if model_eo_wrapper is None: st.warning("- Modèle EO wrapper non chargé.")
        if X_test is None: st.warning("- Données X_test non chargées.")
        if y_test is None: st.warning("- Données y_test non chargées.")

