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
import plotly.express as px # Réintroduit pour l'EDA et autres graphiques interactifs

# --- Fairness Libraries ---
# ExponentiatedGradient est importé dynamiquement si nécessaire dans ensure_eowrapper_in_main
from fairlearn.metrics import (
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
RAW_DATA_FILENAME: str = "application_train.csv" # Réintroduit pour l'EDA
MODEL_BASELINE_FILENAME: str = "lgbm_baseline.joblib"
BASELINE_THRESHOLD_FILENAME: str = "baseline_threshold.joblib"
MODEL_WRAPPED_EO_FILENAME: str = "eo_wrapper_with_proba.joblib"
WRAPPER_EO_MODULE_FILENAME: str = "wrapper_eo.py"

# Dictionnaire des artefacts à télécharger (réintroduction de RAW_DATA_FILENAME)
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

# -- Streamlit config --
st.set_page_config(
    page_title="POC Scoring Équitable (Simplifié)", # Le titre de la page est important pour l'accessibilité (WCAG)
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
        temp_mod_name = "eowrapper_dyn_simplified_eda" # Nom de module unique
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
def load_parquet_file(path: str) -> Optional[pd.DataFrame]:
    """Charge un fichier Parquet."""
    try:
        return pd.read_parquet(path)
    except Exception as e:
        st.error(f"Erreur de chargement du fichier Parquet {path}: {e}")
        return None

@st.cache_data
def load_csv_sample_for_eda(filename: str, sample_frac: float = 0.1, columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """Charge un échantillon d'un fichier CSV pour l'EDA."""
    try:
        df = pd.read_csv(filename, usecols=columns)
        if 0.0 < sample_frac < 1.0:
            if len(df) * sample_frac >= 1:
                df = df.sample(frac=sample_frac, random_state=42)
        return df
    except FileNotFoundError:
        st.error(f"Fichier EDA non trouvé: {filename}")
        return None
    except Exception as e:
        st.error(f"Erreur de chargement du fichier CSV {filename} pour EDA: {e}")
        return None


@st.cache_resource
def load_model_joblib(path: str) -> Any:
    """Charge un modèle sauvegardé avec joblib."""
    st.info(f"Tentative de chargement du modèle depuis {path}...")
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Erreur de chargement du modèle {path}: {e}")
        st.exception(e)
        return None

def sanitize_feature_names(df_input: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les noms de colonnes."""
    df = df_input.copy()
    cleaned_columns = [re.sub(r"[^a-zA-Z0-9_]", "_", str(col)) for col in df.columns]
    df.columns = cleaned_columns
    return df

# === Chargement des modèles et données ===
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

# Chargement des données pour l'EDA
df_eda_sample = load_csv_sample_for_eda(RAW_DATA_FILENAME, sample_frac=0.05) # Échantillon plus petit pour l'EDA
if df_eda_sample is not None:
    st.sidebar.info("Échantillon de données pour EDA chargé.")


# === Fonctions métriques ===
def compute_classification_metrics(
    y_true: pd.Series, 
    y_pred_hard: np.ndarray, 
    y_pred_proba_positive_class: np.ndarray
) -> Dict[str, float]:
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
        for k in ["AUC", "Accuracy", "Precision (1)", "Recall (1)", "F1 (1)", "Taux de sélection"]:
            metrics.setdefault(k, np.nan)
    return metrics

def compute_fairness_metrics(
    y_true: pd.Series, 
    y_pred_hard: np.ndarray, 
    sensitive_features: pd.Series
) -> Dict[str, float]:
    metrics = {}
    try:
        metrics["DPD"] = demographic_parity_difference(y_true, y_pred_hard, sensitive_features=sensitive_features)
        metrics["EOD"] = equalized_odds_difference(y_true, y_pred_hard, sensitive_features=sensitive_features)
    except Exception as e:
        st.warning(f"Erreur calcul métriques d'équité: {e}")
        metrics.setdefault("DPD", np.nan)
        metrics.setdefault("EOD", np.nan)
    return metrics

# === Sidebar navigation (réintroduit EDA et Prediction Client) ===
st.sidebar.title("📊 POC Scoring Équitable")
page_options: List[str] = [
    "Analyse Exploratoire (EDA)",
    "Résultats & Comparaisons",
    "Prédiction sur Client Sélectionné"
]
default_page_index: int = 0

# Utiliser une clé de session unique pour éviter les conflits si d'autres apps utilisent la même
session_key_page_index = "current_page_index_poc_scoring" 
if session_key_page_index not in st.session_state:
    st.session_state[session_key_page_index] = default_page_index

page: str = st.sidebar.radio(
    "Navigation",
    page_options,
    index=st.session_state[session_key_page_index],
    key="nav_radio_poc_scoring" # Clé de widget unique
)
if page_options.index(page) != st.session_state[session_key_page_index]:
    st.session_state[session_key_page_index] = page_options.index(page)
    st.rerun()


# === Contenu des Pages ===
if page == "Analyse Exploratoire (EDA)":
    st.header("🔎 Analyse Exploratoire des Données (EDA)")
    st.caption(f"Basée sur un échantillon de {len(df_eda_sample) if df_eda_sample is not None else 0} lignes du fichier `{RAW_DATA_FILENAME}`.")

    if df_eda_sample is not None and not df_eda_sample.empty:
        st.subheader("Aperçu des données brutes (échantillon)")
        st.dataframe(df_eda_sample.head(), use_container_width=True)

        st.subheader("Statistiques descriptives (variables numériques)")
        st.dataframe(df_eda_sample.describe(include=np.number).T, use_container_width=True)
        
        # Comptage de la variable cible 'TARGET'
        if "TARGET" in df_eda_sample.columns:
            st.subheader("Distribution de la variable cible 'TARGET'")
            target_counts = df_eda_sample["TARGET"].value_counts()
            target_counts_percent = df_eda_sample["TARGET"].value_counts(normalize=True) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Comptage absolu :")
                st.dataframe(target_counts)
            with col2:
                st.write("Pourcentage :")
                st.dataframe(target_counts_percent.map("{:.2f}%".format))

            # Graphique interactif 1: Histogramme de la variable TARGET
            # WCAG: Titre clair, couleurs contrastées par défaut de Plotly, texte lisible.
            try:
                fig_target_hist = px.histogram(
                    df_eda_sample, 
                    x="TARGET", 
                    color="TARGET", # Utilisation de la couleur pour distinguer les catégories
                    title="Histogramme de la variable cible 'TARGET'",
                    labels={"TARGET": "Classe de défaut (0: Non-défaut, 1: Défaut)"},
                    text_auto=True # Affiche les comptages sur les barres
                )
                fig_target_hist.update_layout(bargap=0.2) # Espace entre les barres
                st.plotly_chart(fig_target_hist, use_container_width=True)
            except Exception as e:
                st.warning(f"Impossible de générer l'histogramme de TARGET: {e}")
        else:
            st.warning("La colonne 'TARGET' n'est pas présente dans l'échantillon de données pour l'EDA.")

        # Graphique interactif 2: Distribution d'une variable numérique (ex: AMT_INCOME_TOTAL)
        # Choix d'une variable numérique pertinente pour l'exemple
        numerical_col_for_eda = 'AMT_INCOME_TOTAL' # Revenu total
        if numerical_col_for_eda in df_eda_sample.columns:
            st.subheader(f"Distribution de '{numerical_col_for_eda}'")
            # WCAG: Titre clair, texte lisible.
            # Filtrer les outliers pour une meilleure visualisation (optionnel, mais souvent utile)
            # Par exemple, limiter au 99ème percentile pour éviter que quelques valeurs extrêmes écrasent le graphique
            income_cap = df_eda_sample[numerical_col_for_eda].quantile(0.99)
            df_filtered_income = df_eda_sample[df_eda_sample[numerical_col_for_eda] < income_cap]

            try:
                fig_income_dist = px.histogram(
                    df_filtered_income, 
                    x=numerical_col_for_eda, 
                    color="TARGET" if "TARGET" in df_filtered_income else None, # Colorer par TARGET si disponible
                    marginal="box", # Ajoute un box plot en marge
                    title=f"Distribution de '{numerical_col_for_eda}' (plafonné à {income_cap:,.0f})",
                    labels={numerical_col_for_eda: "Revenu total", "TARGET": "Classe de défaut"}
                )
                st.plotly_chart(fig_income_dist, use_container_width=True)
            except Exception as e:
                st.warning(f"Impossible de générer l'histogramme de {numerical_col_for_eda}: {e}")
        else:
            st.info(f"La colonne '{numerical_col_for_eda}' n'est pas disponible pour l'EDA.")
            
    else:
        st.error("L'échantillon de données pour l'EDA n'a pas pu être chargé. Vérifiez les logs.")


elif page == "Résultats & Comparaisons":
    st.header("📊 Résultats comparatifs sur jeu de test")
    # Le titre de la page est déjà défini via st.set_page_config, mais un header est bon pour la structure.
    if model_baseline is not None and model_eo_wrapper is not None and \
       X_test is not None and y_test is not None and A_test is not None and \
       optimal_thresh_baseline is not None:
        try:
            # st.header("Comparaison des modèles sur le jeu de test") # Déjà un header de page
            
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
            # WCAG: Les tables de données doivent être correctement structurées. st.dataframe le fait.
            # Le contraste des couleurs du thème par défaut de Streamlit est généralement bon.
            st.dataframe(df_res.set_index("Modèle").style.format("{:.3f}", na_rep="-"), use_container_width=True)
        
        except Exception as e:
            st.error(f"Erreur lors du calcul ou de l'affichage des résultats comparatifs: {e}")
            st.exception(e)
    else:
        st.warning("Des éléments sont manquants pour afficher les résultats. Vérifiez les messages dans la barre latérale.")
        # Messages d'erreur plus spécifiques pour aider au débogage
        if model_baseline is None: st.error("- Modèle baseline non chargé.")
        if model_eo_wrapper is None: st.error("- Modèle EO wrapper non chargé.")
        if X_test is None: st.error("- Données X_test non chargées.")
        if y_test is None: st.error("- Données y_test non chargées.")
        if A_test is None: st.error("- Données A_test (features sensibles) non chargées.")
        if optimal_thresh_baseline is None: st.error("- Seuil optimal baseline non chargé.")

elif page == "Prédiction sur Client Sélectionné":
    st.header("🔍 Prédiction sur un Client Sélectionné du Jeu de Test")

    if X_test is not None and model_baseline is not None and model_eo_wrapper is not None \
       and optimal_thresh_baseline is not None and y_test is not None:
        
        # Sélection du client
        # WCAG: Le widget selectbox est accessible au clavier.
        client_ids = X_test.index.tolist()
        if not client_ids:
            st.warning("Aucun ID client disponible dans le jeu de test.")
        else:
            # Limiter le nombre d'options pour la performance du selectbox si X_test est très grand
            max_clients_in_selectbox = 2000 
            if len(client_ids) > max_clients_in_selectbox:
                st.info(f"Affichage des {max_clients_in_selectbox} premiers IDs clients pour la sélection.")
                client_ids_to_display = client_ids[:max_clients_in_selectbox]
            else:
                client_ids_to_display = client_ids

            selected_client_id_str = st.selectbox(
                "Choisissez un ID client du jeu de test:",
                options=[str(id_val) for id_val in client_ids_to_display] # Convertir en str pour le selectbox
            )
            
            # Convertir l'ID sélectionné (str) au type original de l'index
            selected_client_id: Any
            try:
                if X_test.index.dtype == 'int64' or X_test.index.dtype == 'int32':
                    selected_client_id = int(selected_client_id_str)
                elif X_test.index.dtype == 'float64' or X_test.index.dtype == 'float32':
                     selected_client_id = float(selected_client_id_str)
                else: # Conserver comme string si ce n'est ni int ni float
                    selected_client_id = selected_client_id_str
            except ValueError:
                st.error(f"ID client '{selected_client_id_str}' invalide.")
                st.stop()


            if selected_client_id in X_test.index:
                client_features = X_test.loc[[selected_client_id]] # Doit être un DataFrame pour predict_proba
                client_true_target = y_test.loc[selected_client_id] if selected_client_id in y_test.index else "Inconnue"

                st.subheader(f"Données du client ID: {selected_client_id}")
                st.write(f"Vraie cible (TARGET) : **{client_true_target}**")
                st.dataframe(client_features.T.rename(columns={0: "Valeur"}), use_container_width=True)

                # Prédictions
                try:
                    proba_baseline = model_baseline.predict_proba(client_features)[0, 1]
                    pred_baseline = (proba_baseline >= optimal_thresh_baseline).astype(int)

                    proba_eo = model_eo_wrapper.predict_proba(client_features)[0, 1]
                    pred_eo = model_eo_wrapper.predict(client_features)[0] # predict() du wrapper applique le seuil

                    st.subheader("Résultats de la Prédiction")
                    results_data = {
                        "Métrique": ["Probabilité de défaut (classe 1)", "Prédiction (0 ou 1)"],
                        "Modèle Baseline": [f"{proba_baseline:.4f}", pred_baseline],
                        "Modèle EO Wrapper": [f"{proba_eo:.4f}", pred_eo]
                    }
                    df_pred_results = pd.DataFrame(results_data)
                    st.table(df_pred_results.set_index("Métrique")) # st.table est bon pour l'accessibilité des petites tables

                except Exception as e_pred:
                    st.error(f"Erreur lors de la prédiction pour le client {selected_client_id}: {e_pred}")
            else:
                st.error(f"L'ID client {selected_client_id} n'a pas été trouvé dans le jeu de test après conversion.")
    else:
        st.warning("Des éléments sont manquants pour effectuer une prédiction : modèles ou données de test.")

st.sidebar.markdown("---")
st.sidebar.caption("Version simplifiée avec EDA et prédiction client.")

