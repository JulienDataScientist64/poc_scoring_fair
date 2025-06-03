import os
import re
import requests
import importlib.util
import sys # For debugging module paths if necessary

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import shap
import dalex as dx

from fairlearn.reductions import ExponentiatedGradient # Bien que non utilisé directement ici, EOWrapper en dépend
from fairlearn.metrics import (
    MetricFrame,
    selection_rate as fairlearn_selection_rate,
    demographic_parity_difference,
    equalized_odds_difference
)
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Configuration de la page Streamlit
st.set_page_config(
    page_title="POC Scoring Crédit Équitable",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Note about potential OS-level errors
st.sidebar.caption("""
    Note: If you encounter 'OSError: [Errno 28] inotify watch limit reached', 
    it's an OS limit. On systems you control, you can increase this limit. 
    Alternatively, for Streamlit, consider setting `server.fileWatcherType = "none"` 
    in your Streamlit config if frequent file watching isn't needed.
""")


# Récupération du token Hugging Face depuis les secrets Streamlit
HF_TOKEN = st.secrets.get("HF_TOKEN", None)
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# === Chemins & artefacts Hugging Face ===
# Dictionnaire des artefacts à télécharger
ARTEFACTS = {
    "application_train.csv": "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/main/application_train.csv",
    "baseline_threshold.joblib": "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/main/baseline_threshold.joblib",
    "eo_wrapper_with_proba.joblib": "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/main/eo_wrapper_with_proba.joblib",
    "lgbm_baseline.joblib": "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/main/lgbm_baseline.joblib",
    "wrapper_eo.py": "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/main/wrapper_eo.py",
    "X_valid_pre.parquet": "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/main/X_valid_pre.parquet",
    "y_valid.parquet": "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/main/y_valid.parquet",
    "A_valid.parquet": "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/main/A_valid.parquet",
    "X_test_pre.parquet": "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/main/X_test_pre.parquet",
    "y_test.parquet": "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/main/y_test.parquet",
    "A_test.parquet": "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/main/A_test.parquet"
}

def download_if_missing(filename, url):
    """Télécharge le fichier depuis Hugging Face si absent localement, en utilisant les HEADERS si définis."""
    if not os.path.exists(filename):
        st.info(f"Téléchargement de {filename} depuis Hugging Face...")
        try:
            with requests.get(url, stream=True, headers=HEADERS) as r: # Utilisation de HEADERS
                r.raise_for_status()
                with open(filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            st.success(f"{filename} téléchargé.")
        except Exception as e:
            st.error(f"Erreur lors du téléchargement de {filename}: {e}")
            if isinstance(e, requests.exceptions.HTTPError):
                st.error(f"Réponse du serveur: {e.response.status_code} - {e.response.text}")
            st.stop()

# --- Téléchargement de tous les artefacts nécessaires depuis Hugging Face ---
for fname, url in ARTEFACTS.items():
    download_if_missing(fname, url)


# === Import dynamique de la classe EOWrapper (mis en cache) ===
@st.cache_resource
def get_ewrapper_class(wrapper_file_path="wrapper_eo.py"):
    """
    Dynamically imports and returns the EOWrapper class from the specified file.
    Caches the result to ensure the class object is stable across Streamlit reruns.
    """
    # Using a potentially unique module name for clarity in logs, though cache_resource handles identity.
    module_name = "wrapper_eo_module_cached_by_streamlit" 
    
    st.info(f"Attempting dynamic import of EOWrapper from '{wrapper_file_path}' as module '{module_name}'.")
    
    if not os.path.exists(wrapper_file_path):
        # This check is crucial before attempting spec_from_file_location.
        raise FileNotFoundError(f"Wrapper file '{wrapper_file_path}' not found. Ensure it's downloaded and path is correct.")

    spec = importlib.util.spec_from_file_location(module_name, wrapper_file_path)
    
    if spec is None: # spec_from_file_location can return None if file doesn't exist or isn't accessible
        raise ImportError(f"Could not create module spec for '{wrapper_file_path}'. File might be inaccessible or not a Python module.")
    if spec.loader is None: # Should be caught by spec is None, but as a safeguard.
        raise ImportError(f"Module spec for '{wrapper_file_path}' has no loader. Cannot execute module.")
        
    # Create a new module object or get from sys.modules if already imported by this exact mechanism
    # Note: @st.cache_resource ensures this function's body effectively runs once for the same inputs.
    _wrapper_eo_module = importlib.util.module_from_spec(spec)
    
    # Add to sys.modules BEFORE exec_module. This is important for pickle to find the module
    # if it was pickled with this module_name.
    # However, the name `wrapper_eo_module_cached_by_streamlit` is internal to this import.
    # The pickled object likely refers to `wrapper_eo.EOWrapper` or `__main__.EOWrapper`.
    # The key is that `EOWrapper_class_global` will hold THE SAME class object across reruns.
    sys.modules[module_name] = _wrapper_eo_module 
    
    spec.loader.exec_module(_wrapper_eo_module) # Execute the module code to define the class
    
    if not hasattr(_wrapper_eo_module, "EOWrapper"):
        raise AttributeError(f"Module '{module_name}' (from '{wrapper_file_path}') does not define class 'EOWrapper'.")
        
    actual_class = getattr(_wrapper_eo_module, "EOWrapper")
    st.info(f"EOWrapper class imported successfully. Module: {actual_class.__module__}, Name: {actual_class.__name__}, ID: {id(actual_class)}")
    return actual_class

EOWrapper_class_global = None 
try:
    EOWrapper_class_global = get_ewrapper_class("wrapper_eo.py") 
    st.sidebar.success("Classe EOWrapper prête.")
except Exception as e:
    st.error(f"Erreur critique lors de l'obtention de la classe EOWrapper: {e}")
    st.info("L'application ne peut pas continuer sans la classe EOWrapper.")
    st.exception(e) 
    st.stop()


# === Définition des chemins (après téléchargement) ===
RAW_DATA_FILENAME = "application_train.csv"
MODEL_BASELINE_FILENAME = "lgbm_baseline.joblib"
BASELINE_THRESHOLD_FILENAME = "baseline_threshold.joblib"
MODEL_WRAPPED_EO_FILENAME = "eo_wrapper_with_proba.joblib"

# === Fonctions de chargement / cache Streamlit ===
@st.cache_data
def load_parquet_file(path):
    return pd.read_parquet(path)

@st.cache_data
def load_csv_sample(filename, sample_frac=0.3):
    df = pd.read_csv(filename)
    if sample_frac < 1.0 and len(df) * sample_frac >= 1:
        df = df.sample(frac=sample_frac, random_state=42)
    return df

@st.cache_resource
def load_model_joblib(path):
    return joblib.load(path)

def sanitize_feature_names(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()
    new_columns = []
    counts = {} 
    for col in df.columns:
        new_col = str(col)
        new_col = re.sub(r"[^a-zA-Z0-9_]", "_", new_col)
        if new_col in counts:
            counts[new_col] += 1
            final_col = f"{new_col}_{counts[new_col]}"
        else:
            counts[new_col] = 0 
            final_col = new_col
        new_columns.append(final_col)
    df.columns = new_columns
    return df

# === Chargement effectif des artefacts ===
model_baseline = None
optimal_thresh_baseline = 0.5 # Default fallback
model_eo_wrapper = None

try:
    model_baseline = load_model_joblib(MODEL_BASELINE_FILENAME)
    st.sidebar.success("Modèle baseline chargé !")
except Exception as e:
    st.error(f"Erreur de chargement du modèle baseline ('{MODEL_BASELINE_FILENAME}'): {e}")
    st.stop()

try:
    optimal_thresh_baseline = load_model_joblib(BASELINE_THRESHOLD_FILENAME)
    st.sidebar.info(f"Seuil optimal baseline : {optimal_thresh_baseline:.3f}")
except Exception as e:
    st.warning(f"Seuil baseline ('{BASELINE_THRESHOLD_FILENAME}') non trouvé ou erreur. Fallback à 0.5. Erreur: {e}")
    optimal_thresh_baseline = 0.5

if EOWrapper_class_global is not None:
    try:
        model_eo_wrapper = load_model_joblib(MODEL_WRAPPED_EO_FILENAME)
        
        st.info(f"--- Debugging EOWrapper isinstance ---")
        st.info(f"Type de 'model_eo_wrapper' chargé: {type(model_eo_wrapper)}")
        st.info(f"Module de 'model_eo_wrapper': {type(model_eo_wrapper).__module__}")
        st.info(f"Nom de classe de 'model_eo_wrapper': {type(model_eo_wrapper).__name__}")
        st.info(f"ID de la classe de 'model_eo_wrapper': {id(type(model_eo_wrapper))}")
        
        st.info(f"Type de 'EOWrapper_class_global' attendu: {EOWrapper_class_global}")
        st.info(f"Module de 'EOWrapper_class_global': {EOWrapper_class_global.__module__}")
        st.info(f"Nom de 'EOWrapper_class_global': {EOWrapper_class_global.__name__}")
        st.info(f"ID de 'EOWrapper_class_global': {id(EOWrapper_class_global)}")
        st.info(f"--- Fin Debugging ---")

        if not isinstance(model_eo_wrapper, EOWrapper_class_global):
            st.error(f"L'objet chargé depuis '{MODEL_WRAPPED_EO_FILENAME}' n'est pas une instance de la classe EOWrapper attendue.")
            st.error("Cela signifie que la classe du modèle désérialisé et la classe EOWrapper importée dynamiquement sont des objets différents en mémoire.")
            st.error("Vérifiez que 'eo_wrapper_with_proba.joblib' a été créé avec une classe EOWrapper dont la définition est identique à celle dans 'wrapper_eo.py'.")
            st.error(f"Module de l'objet chargé: '{type(model_eo_wrapper).__module__}', Module de la classe attendue: '{EOWrapper_class_global.__module__}'")
            st.stop()
        st.sidebar.success("EO Wrapper chargé et vérifié !")
    except Exception as e:
        st.error(f"Erreur de chargement du modèle EO Wrapper ('{MODEL_WRAPPED_EO_FILENAME}'): {e}")
        st.exception(e)
        st.stop()
else:
    # This case should ideally be caught by the st.stop() in the EOWrapper_class_global try-except block.
    st.error("La classe EOWrapper (EOWrapper_class_global) n'a pas pu être initialisée. Impossible de charger le modèle EO Wrapper.")
    st.stop()

# Données de validation et de test
X_valid_raw, y_valid, A_valid, X_test_raw, y_test, A_test = None, None, None, None, None, None
try:
    X_valid_raw = load_parquet_file("X_valid_pre.parquet")
    y_valid = load_parquet_file("y_valid.parquet").squeeze()
    A_valid = load_parquet_file("A_valid.parquet").squeeze()
    X_test_raw = load_parquet_file("X_test_pre.parquet")
    y_test = load_parquet_file("y_test.parquet").squeeze()
    A_test = load_parquet_file("A_test.parquet").squeeze()
    st.sidebar.info("Données de validation et de test chargées.")
except Exception as e:
    st.error(f"Erreur de chargement des splits de données (Parquet) : {e}")
    st.stop()

# Nettoyage des noms de features pour X_valid et X_test
X_valid, X_test = None, None
try:
    X_valid = sanitize_feature_names(X_valid_raw)
    X_test = sanitize_feature_names(X_test_raw)
    st.sidebar.info("Noms des features nettoyés pour X_valid et X_test.")
    if not X_valid_raw.columns.equals(X_valid.columns):
        st.sidebar.warning("Certaines colonnes de X_valid ont été renommées par sanitize_feature_names.")
    if not X_test_raw.columns.equals(X_test.columns):
        st.sidebar.warning("Certaines colonnes de X_test ont été renommées par sanitize_feature_names.")
except Exception as e:
    st.error(f"Erreur lors du nettoyage des noms de features : {e}")
    st.stop()

# EDA brute pour analyse (échantillon)
df_eda_raw_sample = None 
try:
    df_eda_raw_sample = load_csv_sample(RAW_DATA_FILENAME, sample_frac=0.1) 
except Exception as e:
    st.warning(f"Impossible de charger l'échantillon pour l'EDA ('{RAW_DATA_FILENAME}'): {e}")

# === Fonctions métriques ===
def compute_classification_metrics(y_true, y_pred_hard, y_pred_proba_positive_class):
    return {
        "AUC": roc_auc_score(y_true, y_pred_proba_positive_class),
        "Accuracy": accuracy_score(y_true, y_pred_hard),
        "Precision (1)": precision_score(y_true, y_pred_hard, pos_label=1, zero_division=0),
        "Recall (1)": recall_score(y_true, y_pred_hard, pos_label=1, zero_division=0),
        "F1 (1)": f1_score(y_true, y_pred_hard, pos_label=1, zero_division=0),
        "Taux de sélection": np.mean(y_pred_hard),
    }

def compute_fairness_metrics(y_true, y_pred_hard, sensitive_features):
    try:
        dpd = demographic_parity_difference(y_true, y_pred_hard, sensitive_features=sensitive_features)
        eod = equalized_odds_difference(y_true, y_pred_hard, sensitive_features=sensitive_features)
        return {"DPD": dpd, "EOD": eod}
    except Exception as e:
        st.warning(f"Erreur lors du calcul des métriques d'équité: {e}")
        return {"DPD": np.nan, "EOD": np.nan}

# === Sidebar navigation ===
st.sidebar.title("📊 POC Scoring Équitable")
page_options = [
    "Contexte & Objectifs", "Méthodologie", "Analyse Exploratoire", "Résultats & Comparaisons",
    "ROC/Proba - Baseline", "ROC/Proba - EO Wrapper",
    "Intersectionnalité", "Explicabilité Locale", "Explicabilité Globale"
]
default_page_index = 0
if 'current_page_index' not in st.session_state:
    st.session_state.current_page_index = default_page_index

page = st.sidebar.radio(
    "Navigation", 
    page_options, 
    index=st.session_state.current_page_index,
    key="nav_radio" 
)
if page_options.index(page) != st.session_state.current_page_index:
    st.session_state.current_page_index = page_options.index(page)

st.sidebar.markdown("---")
st.sidebar.info(f"Seuil Baseline : {optimal_thresh_baseline:.4f}")
if model_eo_wrapper and hasattr(model_eo_wrapper, 'threshold'):
    st.sidebar.info(f"Seuil EO Wrapper : {model_eo_wrapper.threshold:.4f}")
else:
    st.sidebar.warning("Seuil EO Wrapper non disponible ou modèle non chargé.")


# === Contenu des Pages ===

if page == "Contexte & Objectifs":
    st.header("Contexte & Références")
    st.markdown(
        """
        **Pourquoi l’équité dans le scoring crédit ?**
        - Les régulateurs (comme l’IA Act et les lois anti-discrimination) imposent que les modèles de scoring crédit n’avantagent ni ne désavantagent un groupe (par exemple le genre).
        - Ce POC compare deux approches :
          1. **LightGBM classique** (modèle standard de machine learning)
          2. **LightGBM associé à Fairlearn EG-EO** (ajout d’une contrainte d’équité sur la prédiction)

        **Objectif métier :**
        Obtenir un modèle performant mais qui reste juste entre les différents groupes (ex : hommes/femmes).
        """
    )
    st.subheader("Papiers de référence")
    with st.expander("Hardt, Price & Srebro (2016) – Equalized Odds"):
        st.write(
            """
            **Résumé pédagogique :**
            - Equalized Odds impose que le taux de bonne détection (rappel) soit similaire pour chaque groupe (par exemple hommes et femmes), pour les personnes qui remboursent ou non.
            - Un modèle respectant bien Equalized Odds limite donc les écarts d’erreur selon le groupe sensible.
            """
        )
        st.markdown("[Lire le papier (arXiv)](https://arxiv.org/abs/1610.02413)")

    with st.expander("Agarwal et al. (2019) – Exponentiated Gradient"):
        st.write(
            """
            **Résumé pédagogique :**
            - L’algorithme Exponentiated Gradient combine plusieurs modèles en ajustant leurs poids pour trouver un compromis optimal entre performance et équité.
            - À chaque étape, il renforce les modèles qui respectent le mieux la contrainte d’équité.
            - Cette méthode permet d’obtenir un modèle global qui ne discrimine pas, tout en gardant un bon niveau de prédiction.
            """
        )
        st.markdown("[Lire le papier (ACM)](https://dl.acm.org/doi/10.1145/3287560.3287572)")

    st.subheader("Métriques d'équité utilisées")
    st.markdown(
        """
        - **Demographic Parity Difference (DPD) :**
          > Mesure la différence de taux d’attribution positive du crédit entre groupes (idéal : zéro différence).
        - **Equalized Odds Difference (EOD) :**
          > Mesure l’écart de performance du modèle (sensibilité/spécificité) selon le groupe sensible. Un modèle équitable aura un EOD proche de zéro.
        - **Exponentiated Gradient (EG) :**
          > Méthode pour trouver un compromis entre performance et équité, en combinant plusieurs modèles de façon intelligente.
        """
    )

elif page == "Méthodologie":
    st.header("Méthodologie")
    st.subheader("Données & Préparation")
    st.write(
        """
        - **Jeu de données** : Home Credit (~300 000 clients, 120 variables socio-économiques).
        - Découpage en 3 parties : apprentissage (80%), validation (10%), test (10%).
        - **Nettoyage** : gestion des valeurs bizarres ou manquantes, suppression des doublons, filtrage sur le genre, plafonnement des revenus extrêmes.
        - **Nouvelles variables** : création de ratios simples (ex : mensualité/revenu, crédit/revenu), transformation de l’âge.
        - **Mise en forme** : transformation des variables catégorielles, découpage de l’âge en tranches, etc.
        - **Encodage & imputation** : gestion automatique des valeurs manquantes et transformation des variables pour les modèles.
        - **Nettoyage des noms de features** : Standardisation des noms de variables pour éviter les problèmes techniques (e.g., caractères spéciaux).
        """
    )
    st.subheader("Modèle de base (LightGBM)")
    st.write(
        """
        - Modèle classique de machine learning qui apprend à prédire le défaut de remboursement.
        - Prend en compte le déséquilibre entre bons et mauvais payeurs.
        - Le seuil de décision (pour dire “défaut” ou “pas défaut”) est choisi de façon optimale sur la validation.
        """
    )
    st.subheader("Modèle équitable (EG-EO)")
    st.write(
        """
        - Modèle LightGBM ajusté avec Fairlearn pour garantir l’équité entre hommes et femmes (variable sensible `CODE_GENDER`).
        - La méthode `ExponentiatedGradient` avec la contrainte `EqualizedOdds` combine plusieurs modèles et ajuste leurs poids pour minimiser les écarts de traitement selon le genre.
        - On fixe une tolérance maximale sur l’écart d’équité autorisé (`eps`).
        - Le modèle final est un "wrapper" qui encapsule cette logique et un seuil de décision optimisé.
        """
    )
    st.subheader("Évaluation et comparaison")
    st.write(
        """
        - **Performances mesurées** : capacité à bien trier les clients (AUC, précision, rappel, F1).
        - **Équité** : on vérifie que le modèle ne favorise pas un groupe par rapport à l’autre, via des métriques spécifiques (DPD, EOD).
        - **Analyse détaillée** : matrice de confusion et taux de sélection par groupe.
        """
    )

elif page == "Analyse Exploratoire":
    st.header("🔎 Analyse exploratoire (EDA)")
    if df_eda_raw_sample is not None and not df_eda_raw_sample.empty:
        st.subheader(f"Aperçu échantillon ({len(df_eda_raw_sample)} lignes)")
        st.dataframe(df_eda_raw_sample.head(20), use_container_width=True)
        
        st.subheader("Statistiques descriptives (variables numériques)")
        st.dataframe(df_eda_raw_sample.describe(include=np.number).T, use_container_width=True)
        
        if "TARGET" in df_eda_raw_sample.columns:
            st.subheader("Distribution de la cible (TARGET)")
            target_counts = df_eda_raw_sample["TARGET"].value_counts(normalize=True) * 100
            target_counts_df = pd.DataFrame(target_counts).T
            st.dataframe(target_counts_df.style.format("{:.2f}%"), use_container_width=True)
            
            try:
                fig = px.histogram(df_eda_raw_sample, x="TARGET", color="TARGET",
                                   title="Distribution de la variable cible 'TARGET'",
                                   color_discrete_sequence=px.colors.qualitative.Safe)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Impossible de générer l'histogramme de TARGET: {e}")
        else:
            st.warning("La colonne 'TARGET' n'est pas présente dans l'échantillon de données pour l'EDA.")
    else:
        st.info("Aucune donnée disponible pour l'analyse exploratoire.")

elif page == "Résultats & Comparaisons":
    st.header("📊 Résultats comparatifs sur jeu de test")
    # Ensure models and data are loaded before attempting predictions
    if model_baseline is not None and model_eo_wrapper is not None and X_test is not None and y_test is not None and A_test is not None:
        try:
            # Baseline
            y_test_proba_b = model_baseline.predict_proba(X_test)[:, 1]
            y_test_pred_b = (y_test_proba_b >= optimal_thresh_baseline).astype(int)
            metrics_b = compute_classification_metrics(y_test, y_test_pred_b, y_test_proba_b)
            fairness_b = compute_fairness_metrics(y_test, y_test_pred_b, A_test)
            
            # EO Wrapper
            y_test_proba_eo = model_eo_wrapper.predict_proba(X_test)[:, 1] 
            y_test_pred_eo = model_eo_wrapper.predict(X_test) 
            metrics_eo = compute_classification_metrics(y_test, y_test_pred_eo, y_test_proba_eo)
            fairness_eo = compute_fairness_metrics(y_test, y_test_pred_eo, A_test)
            
            df_res = pd.DataFrame([
                {"Modèle": "Baseline", **metrics_b, **fairness_b},
                {"Modèle": "EO Wrapper", **metrics_eo, **fairness_eo}
            ])
            st.dataframe(df_res.style.format("{:.3f}", na_rep="-"), use_container_width=True)
        except Exception as e:
            st.error(f"Erreur lors du calcul ou de l'affichage des résultats comparatifs: {e}")
            st.exception(e)
    else:
        st.warning("Les modèles ou les données de test ne sont pas complètement chargés. Impossible d'afficher les résultats.")


elif page == "ROC/Proba - Baseline":
    st.header("Courbe ROC & Distribution de probas - Baseline (sur jeu de validation)")
    if model_baseline is not None and X_valid is not None and y_valid is not None:
        try:
            y_val_proba_b = model_baseline.predict_proba(X_valid)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_valid, y_val_proba_b)
            auc_b_val = roc_auc_score(y_valid, y_val_proba_b)
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={auc_b_val:.3f})'))
            fig_roc.add_shape(type='line', x0=0, x1=1, y0=0, y1=1, line=dict(dash='dash', color='grey'))
            
            optimal_idx_val = np.argmin(np.abs(thresholds - optimal_thresh_baseline)) 
            fig_roc.add_trace(go.Scatter(
                x=[fpr[optimal_idx_val]], y=[tpr[optimal_idx_val]], mode='markers',
                marker=dict(size=10, color='red'), name=f'Seuil Optimal ({optimal_thresh_baseline:.3f})'
            ))
            fig_roc.update_layout(title="Courbe ROC (Validation, Baseline)",
                                  xaxis_title="Taux de Faux Positifs (FPR)",
                                  yaxis_title="Taux de Vrais Positifs (TPR)")
            st.plotly_chart(fig_roc, use_container_width=True)
            
            st.subheader("Distribution des probabilités (Baseline, Validation)")
            df_dist_b = pd.DataFrame({"proba_baseline": y_val_proba_b, "y_true": y_valid})
            fig_dist_b = px.histogram(df_dist_b, x="proba_baseline", color="y_true", nbins=50, 
                                      barmode='overlay', marginal="rug",
                                      color_discrete_sequence=px.colors.qualitative.Safe,
                                      labels={"y_true": "Cible réelle", "proba_baseline": "Score Baseline"})
            fig_dist_b.add_vline(x=optimal_thresh_baseline, line_color="red", line_dash="dash", 
                                annotation_text=f"Seuil={optimal_thresh_baseline:.3f}", annotation_position="top right")
            st.plotly_chart(fig_dist_b, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur lors de la génération des graphiques pour Baseline: {e}")
    else:
        st.warning("Modèle Baseline ou données de validation non chargés.")


elif page == "ROC/Proba - EO Wrapper":
    st.header("Courbe ROC & Distribution de probas - EO Wrapper (sur jeu de validation)")
    if model_eo_wrapper is not None and X_valid is not None and y_valid is not None:
        try:
            y_val_proba_eo = model_eo_wrapper.predict_proba(X_valid)[:, 1]
            fpr_eo, tpr_eo, thresholds_eo = roc_curve(y_valid, y_val_proba_eo)
            auc_eo_val = roc_auc_score(y_valid, y_val_proba_eo)
            
            fig_roc_eo = go.Figure()
            fig_roc_eo.add_trace(go.Scatter(x=fpr_eo, y=tpr_eo, mode='lines', name=f'ROC (AUC={auc_eo_val:.3f})'))
            fig_roc_eo.add_shape(type='line', x0=0, x1=1, y0=0, y1=1, line=dict(dash='dash', color='grey'))
            
            optimal_idx_eo_val = np.argmin(np.abs(thresholds_eo - model_eo_wrapper.threshold))
            fig_roc_eo.add_trace(go.Scatter(
                x=[fpr_eo[optimal_idx_eo_val]], y=[tpr_eo[optimal_idx_eo_val]], mode='markers',
                marker=dict(size=10, color='red'), name=f'Seuil du Wrapper ({model_eo_wrapper.threshold:.3f})'
            ))
            fig_roc_eo.update_layout(title="Courbe ROC (Validation, EO Wrapper)",
                                     xaxis_title="Taux de Faux Positifs (FPR)",
                                     yaxis_title="Taux de Vrais Positifs (TPR)")
            st.plotly_chart(fig_roc_eo, use_container_width=True)
            
            st.subheader("Distribution des probabilités (EO Wrapper, Validation)")
            df_dist_eo = pd.DataFrame({"proba_eo": y_val_proba_eo, "y_true": y_valid})
            fig_dist_eo = px.histogram(df_dist_eo, x="proba_eo", color="y_true", nbins=50, 
                                       barmode='overlay', marginal="rug",
                                       color_discrete_sequence=px.colors.qualitative.Safe,
                                       labels={"y_true": "Cible réelle", "proba_eo": "Score EO Wrapper"})
            fig_dist_eo.add_vline(x=model_eo_wrapper.threshold, line_color="red", line_dash="dash",
                                  annotation_text=f"Seuil={model_eo_wrapper.threshold:.3f}", annotation_position="top right")
            st.plotly_chart(fig_dist_eo, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur lors de la génération des graphiques pour EO Wrapper: {e}")
    else:
        st.warning("Modèle EO Wrapper ou données de validation non chargés.")

elif page == "Intersectionnalité":
    st.header("Équité intersectionnelle (Genre & Tranche d'âge)")
    st.caption("Analyse sur le jeu de test, en utilisant les données brutes pour l'âge.")
    
    df_raw_full = None
    try:
        df_raw_full = load_csv_sample(RAW_DATA_FILENAME, sample_frac=1.0) 
    except Exception as e:
        st.warning(f"Impossible de charger le fichier brut complet '{RAW_DATA_FILENAME}' pour l'analyse intersectionnelle: {e}")

    if df_raw_full is not None and not df_raw_full.empty and \
       model_baseline is not None and model_eo_wrapper is not None and \
       X_test is not None and y_test is not None:
        if "DAYS_BIRTH" in df_raw_full.columns and "CODE_GENDER" in df_raw_full.columns:
            df_raw_full_with_age = df_raw_full.assign(
                AGE_YEARS = lambda x: -x["DAYS_BIRTH"] / 365,
                AGE_BIN = lambda x: pd.cut(
                    x["AGE_YEARS"],
                    bins=[17, 25, 35, 45, 55, 65, 100],
                    labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
                    right=True 
                )
            )
            common_indices = X_test.index.intersection(df_raw_full_with_age.index)
            
            if not common_indices.empty:
                X_test_aligned = X_test.loc[common_indices]
                y_test_aligned = y_test.loc[common_indices]
                
                sensitive_group_inter = (
                    df_raw_full_with_age.loc[common_indices, "CODE_GENDER"].astype(str) + 
                    " | " + 
                    df_raw_full_with_age.loc[common_indices, "AGE_BIN"].astype(str)
                )
                sensitive_group_inter.name = "Groupe_Intersectionnel"

                y_pred_b_inter = (model_baseline.predict_proba(X_test_aligned)[:, 1] >= optimal_thresh_baseline).astype(int)
                y_pred_eo_inter = model_eo_wrapper.predict(X_test_aligned)

                metrics_to_compute = {
                    "Taux de Sélection": fairlearn_selection_rate,
                    "Rappel (Recall)": lambda yt, yp: recall_score(yt, yp, pos_label=1, zero_division=0),
                    "Précision (Precision)": lambda yt, yp: precision_score(yt, yp, pos_label=1, zero_division=0)
                }
                
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
                
                st.dataframe(df_plot.style.format("{:.3f}"), use_container_width=True)

                for metric_base_name in metrics_to_compute.keys():
                    col_baseline = f"{metric_base_name} - Baseline"
                    col_eo = f"{metric_base_name} - EO Wrapper"
                    if col_baseline in df_plot.columns and col_eo in df_plot.columns:
                        fig = px.bar(df_plot, y="Groupe_Intersectionnel", x=[col_baseline, col_eo], 
                                     barmode="group", title=f"{metric_base_name} par groupe (Baseline vs EO)",
                                     color_discrete_sequence=px.colors.qualitative.Plotly)
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Aucun indice commun trouvé entre X_test et les données brutes pour l'intersectionnalité.")
        else:
            st.warning("Colonnes 'DAYS_BIRTH' ou 'CODE_GENDER' manquantes pour l'analyse intersectionnelle.")
    else:
        st.info("Données ou modèles non prêts pour l'analyse intersectionnelle.")


elif page == "Explicabilité Locale":
    st.header("Explicabilité locale (SHAP Force Plot)")
    st.caption("Affiche l'explication d'une prédiction pour un client sélectionné du jeu de test.")
    st.markdown(
        """
        **Note sur l'explicabilité du EO Wrapper :**
        L'algorithme Exponentiated Gradient (utilisé dans le EO Wrapper) est un ensemble de plusieurs modèles LightGBM.
        Pour des raisons de complexité et de performance avec SHAP, l'explication ci-dessous pour le "EO Wrapper" 
        est basée sur le **premier estimateur (modèle LightGBM) de cet ensemble**. 
        Ce n'est donc qu'une approximation de l'explication du modèle complet.
        """
    )
    
    if X_test is not None and model_baseline is not None and model_eo_wrapper is not None:
        idx_options = X_test.index.astype(str).tolist()
        max_options = 1000 
        if len(idx_options) > max_options:
            st.info(f"Affichage des {max_options} premiers clients pour la sélection.")
            idx_options = idx_options[:max_options]

        if not idx_options:
            st.warning("Aucun client disponible pour l'explication locale.")
        else:
            if 'selected_client_id_shap' not in st.session_state or st.session_state.selected_client_id_shap not in idx_options:
                st.session_state.selected_client_id_shap = idx_options[0]
            
            idx_selected_str = st.selectbox(
                "ID Client à expliquer (depuis X_test) :", 
                idx_options, 
                index=idx_options.index(st.session_state.selected_client_id_shap),
                key="client_select_shap"
            )
            st.session_state.selected_client_id_shap = idx_selected_str

            try:
                idx_selected_original_type = int(idx_selected_str) if X_test.index.dtype == 'int64' else idx_selected_str
            except ValueError:
                st.error(f"Impossible de convertir l'ID client '{idx_selected_str}' au type d'index original.")
                st.stop()

            if idx_selected_original_type not in X_test.index:
                st.error(f"L'ID client sélectionné '{idx_selected_original_type}' n'est pas trouvé dans l'index de X_test.")
            else:
                client_feat = X_test.loc[[idx_selected_original_type]] 
                try:
                    st.markdown("**Force Plot – Modèle Baseline**")
                    with st.spinner("Calcul SHAP Baseline..."):
                        explainer_b = shap.TreeExplainer(model_baseline)
                        shap_val_b = explainer_b.shap_values(client_feat) 
                        expected_val_b = explainer_b.expected_value
                    
                    shap_values_for_plot_b = shap_val_b[1] if isinstance(shap_val_b, list) else shap_val_b
                    expected_value_for_plot_b = expected_val_b[1] if isinstance(expected_val_b, list) else expected_val_b
                    if shap_values_for_plot_b.ndim > 1: shap_values_for_plot_b = shap_values_for_plot_b[0]

                    plot_html_b = shap.force_plot(
                        expected_value_for_plot_b, shap_values_for_plot_b,
                        client_feat, matplotlib=False, show=False
                    ).html()
                    st.components.v1.html(f"<head>{shap.getjs()}</head><body>{plot_html_b}</body>", height=220, scrolling=True)

                    st.markdown("**Force Plot – EO Wrapper (1er estimateur de l'ensemble)**")
                    if hasattr(model_eo_wrapper, 'mitigator') and \
                       hasattr(model_eo_wrapper.mitigator, 'predictors_') and \
                       len(model_eo_wrapper.mitigator.predictors_) > 0:
                        with st.spinner("Calcul SHAP EO Wrapper..."):
                            first_estimator_eo = model_eo_wrapper.mitigator.predictors_[0]
                            explainer_eo = shap.TreeExplainer(first_estimator_eo)
                            shap_val_eo = explainer_eo.shap_values(client_feat)
                            expected_val_eo = explainer_eo.expected_value

                        shap_values_for_plot_eo = shap_val_eo[1] if isinstance(shap_val_eo, list) else shap_val_eo
                        expected_value_for_plot_eo = expected_val_eo[1] if isinstance(expected_val_eo, list) else expected_val_eo
                        if shap_values_for_plot_eo.ndim > 1: shap_values_for_plot_eo = shap_values_for_plot_eo[0]
                        
                        plot_html_eo = shap.force_plot(
                            expected_value_for_plot_eo, shap_values_for_plot_eo,
                            client_feat, matplotlib=False, show=False
                        ).html()
                        st.components.v1.html(f"<head>{shap.getjs()}</head><body>{plot_html_eo}</body>", height=220, scrolling=True)
                    else:
                        st.warning("Structure interne du EO Wrapper non conforme pour SHAP (mitigator.predictors_).")
                except Exception as e:
                    st.error(f"Erreur SHAP force plots: {e}")
                    st.exception(e)
    else:
        st.warning("Données ou modèles non prêts pour l'explicabilité locale.")


elif page == "Explicabilité Globale":
    st.header("Explicabilité globale (SHAP & DALEX)")
    st.caption("Basée sur un échantillon du jeu de validation.")
    st.markdown(
        """
        **Note sur l'explicabilité du EO Wrapper :**
        L'analyse SHAP et DALEX pour le "EO Wrapper" est basée sur le 
        **premier estimateur (modèle LightGBM) de l'ensemble ExponentiatedGradient**. 
        """
    )

    if X_valid is not None and y_valid is not None and model_baseline is not None and model_eo_wrapper is not None:
        sample_size = min(500, X_valid.shape[0]) 
        if sample_size < 1:
            st.warning("Pas assez de données dans X_valid pour l'explicabilité globale.")
        else:
            X_sample = X_valid.sample(n=sample_size, random_state=42)
            y_sample = y_valid.loc[X_sample.index] 

            try:
                st.subheader("SHAP - Importance globale (Baseline)")
                with st.spinner("Calcul SHAP global Baseline..."):
                    explainer_b_glob = shap.TreeExplainer(model_baseline)
                    shap_val_b_glob = explainer_b_glob.shap_values(X_sample)
                    shap_val_b_for_plot = shap_val_b_glob[1] if isinstance(shap_val_b_glob, list) else shap_val_b_glob
                mean_abs_shap_b = np.abs(shap_val_b_for_plot).mean(axis=0)
                df_shap_b = pd.DataFrame({"Feature": X_sample.columns, "Importance_SHAP": mean_abs_shap_b})
                df_shap_b = df_shap_b.sort_values("Importance_SHAP", ascending=False).head(20)
                
                fig_shap_summary_b = px.bar(df_shap_b.sort_values("Importance_SHAP", ascending=True), 
                                            x="Importance_SHAP", y="Feature", orientation="h",
                                            title="Top 20 Features (Baseline, mean |SHAP value|)",
                                            color="Importance_SHAP", color_continuous_scale=px.colors.sequential.Plasma)
                st.plotly_chart(fig_shap_summary_b, use_container_width=True)
                with st.expander("Données SHAP (Baseline)"): st.dataframe(df_shap_b, use_container_width=True)

                st.subheader("SHAP - Importance globale (EO Wrapper - 1er estimateur)")
                if hasattr(model_eo_wrapper, 'mitigator') and hasattr(model_eo_wrapper.mitigator, 'predictors_') and \
                   len(model_eo_wrapper.mitigator.predictors_) > 0:
                    with st.spinner("Calcul SHAP global EO Wrapper..."):
                        first_estimator_eo_glob = model_eo_wrapper.mitigator.predictors_[0]
                        explainer_eo_glob = shap.TreeExplainer(first_estimator_eo_glob)
                        shap_val_eo_glob = explainer_eo_glob.shap_values(X_sample)
                        shap_val_eo_for_plot = shap_val_eo_glob[1] if isinstance(shap_val_eo_glob, list) else shap_val_eo_glob
                    mean_abs_shap_eo = np.abs(shap_val_eo_for_plot).mean(axis=0)
                    df_shap_eo = pd.DataFrame({"Feature": X_sample.columns, "Importance_SHAP": mean_abs_shap_eo})
                    df_shap_eo = df_shap_eo.sort_values("Importance_SHAP", ascending=False).head(20)

                    fig_shap_summary_eo = px.bar(df_shap_eo.sort_values("Importance_SHAP", ascending=True), 
                                                 x="Importance_SHAP", y="Feature", orientation="h",
                                                 title="Top 20 Features (EO - 1er est., mean |SHAP value|)",
                                                 color="Importance_SHAP", color_continuous_scale=px.colors.sequential.Plasma)
                    st.plotly_chart(fig_shap_summary_eo, use_container_width=True)
                    with st.expander("Données SHAP (EO Wrapper)"): st.dataframe(df_shap_eo, use_container_width=True)
                else:
                    st.warning("Structure interne du EO Wrapper non conforme pour SHAP globale.")

                st.subheader("DALEX - Importance par permutation (Baseline, Perte AUC)")
                with st.spinner("Calcul DALEX Baseline..."):
                    exp_b_dalex = dx.Explainer(model_baseline, X_sample, y_sample, label="Baseline", verbose=False)
                    parts_b_dalex = exp_b_dalex.model_parts(loss_function="auc", N=None, random_state=42)
                df_dalex_b = parts_b_dalex.result[parts_b_dalex.result.variable.isin(["_full_model_", "_baseline_"]) == False]
                df_dalex_b = df_dalex_b.sort_values("dropout_loss", ascending=False).head(20)

                fig_dx_b = px.bar(df_dalex_b.sort_values("dropout_loss", ascending=True), 
                                  x="dropout_loss", y="variable", orientation="h",
                                  title="DALEX - Top 20 Features (Baseline, Perte Dropout AUC)",
                                  labels={"variable": "Feature", "dropout_loss": "Perte Dropout (AUC)"},
                                  color="dropout_loss", color_continuous_scale=px.colors.sequential.Viridis)
                st.plotly_chart(fig_dx_b, use_container_width=True)
                with st.expander("Données DALEX (Baseline)"): st.dataframe(parts_b_dalex.result, use_container_width=True)

                st.subheader("DALEX - Importance par permutation (EO Wrapper - 1er est., Perte AUC)")
                if hasattr(model_eo_wrapper, 'mitigator') and hasattr(model_eo_wrapper.mitigator, 'predictors_') and \
                   len(model_eo_wrapper.mitigator.predictors_) > 0:
                    with st.spinner("Calcul DALEX EO Wrapper..."):
                        first_estimator_eo_dalex = model_eo_wrapper.mitigator.predictors_[0]
                        exp_eo_dalex = dx.Explainer(first_estimator_eo_dalex, X_sample, y_sample, label="EO (1er est.)", verbose=False)
                        parts_eo_dalex = exp_eo_dalex.model_parts(loss_function="auc", N=None, random_state=42)
                    df_dalex_eo = parts_eo_dalex.result[parts_eo_dalex.result.variable.isin(["_full_model_", "_baseline_"]) == False]
                    df_dalex_eo = df_dalex_eo.sort_values("dropout_loss", ascending=False).head(20)

                    fig_dx_eo = px.bar(df_dalex_eo.sort_values("dropout_loss", ascending=True), 
                                       x="dropout_loss", y="variable", orientation="h",
                                       title="DALEX - Top 20 Features (EO - 1er est., Perte Dropout AUC)",
                                       labels={"variable": "Feature", "dropout_loss": "Perte Dropout (AUC)"},
                                       color="dropout_loss", color_continuous_scale=px.colors.sequential.Viridis)
                    st.plotly_chart(fig_dx_eo, use_container_width=True)
                    with st.expander("Données DALEX (EO Wrapper)"): st.dataframe(parts_eo_dalex.result, use_container_width=True)
                else:
                    st.warning("Structure interne du EO Wrapper non conforme pour DALEX.")
            except Exception as e:
                st.error(f"Erreur graphes explicabilité globale: {e}")
                st.exception(e)
    else:
        st.warning("Données ou modèles non prêts pour l'explicabilité globale.")

st.markdown("---")
st.caption(f"POC Scoring Crédit Équitable – {pd.Timestamp.now(tz='Europe/Paris').strftime('%d/%m/%Y %H:%M')}")
