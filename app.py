# --------------------------------------------
# fichier : app_streamlit.py
# --------------------------------------------

import os
import sys
import importlib.util
import requests
import streamlit as st
from typing import List, Dict, Any, Optional

# ——————————————————————————————————————————————————————————————
# CONSTANTES ET CHEMINS
# ——————————————————————————————————————————————————————————————
RAW_DATA_FILENAME: str = "application_train.csv"
MODEL_BASELINE_FILENAME: str = "lgbm_baseline.joblib"
BASELINE_THRESHOLD_FILENAME: str = "baseline_threshold.joblib"
MODEL_WRAPPED_EO_FILENAME: str = "eo_wrapper_with_proba.joblib"
WRAPPER_EO_MODULE_FILENAME: str = "wrapper_eo.py"

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

HF_TOKEN: Optional[str] = st.secrets.get("HF_TOKEN", None)
HEADERS: Dict[str, str] = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# ——————————————————————————————————————————————————————————————
# FONCTIONS UTILITAIRES (définies une fois pour tout l’app)
# ——————————————————————————————————————————————————————————————
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


def ensure_eowrapper_in_main(wrapper_file_path: str = WRAPPER_EO_MODULE_FILENAME) -> Optional[type]:
    """Charge dynamiquement EOWrapper et l'injecte dans __main__."""
    try:
        temp_mod_name = "eowrapper_dyn_simplified_roc_conf"
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
def load_parquet_file(path: str) -> Optional["pd.DataFrame"]:
    """Charge un fichier Parquet."""
    try:
        import pandas as pd  # import paresseux
        return pd.read_parquet(path)
    except Exception as e:
        st.error(f"Erreur de chargement du fichier Parquet {path}: {e}")
        return None


@st.cache_data
def load_csv_sample_for_eda(
    filename: str, sample_frac: float = 0.1, columns: Optional[List[str]] = None
) -> Optional["pd.DataFrame"]:
    """Charge un échantillon d'un fichier CSV pour l’EDA."""
    try:
        import pandas as pd  # import paresseux
        df = pd.read_csv(filename, usecols=columns)
        if 0.0 < sample_frac < 1.0 and len(df) * sample_frac >= 1:
            df = df.sample(frac=sample_frac, random_state=42)
        return df
    except FileNotFoundError:
        st.error(f"Fichier EDA non trouvé: {filename}")
        return None
    except Exception as e:
        st.error(f"Erreur de chargement du CSV {filename} pour EDA: {e}")
        return None


@st.cache_resource
def load_model_joblib(path: str) -> Any:
    """Charge un modèle sauvegardé avec joblib."""
    st.info(f"Tentative de chargement du modèle depuis {path}...")
    try:
        import joblib  # import paresseux
        return joblib.load(path)
    except Exception as e:
        st.error(f"Erreur de chargement du modèle {path}: {e}")
        st.exception(e)
        return None


def sanitize_feature_names(df_input: "pd.DataFrame") -> "pd.DataFrame":
    """Nettoie les noms de colonnes."""
    import re  # import paresseux
    df = df_input.copy()
    cleaned_columns = [re.sub(r"[^a-zA-Z0-9_]", "_", str(col)) for col in df.columns]
    df.columns = cleaned_columns
    return df


# ——————————————————————————————————————————————————————————————
# CONFIGURATION GLOBALE DE STREAMLIT
# ——————————————————————————————————————————————————————————————
st.set_page_config(
    page_title="POC Scoring Équitable (Simplifié)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ——————————————————————————————————————————————————————————————
# NAVIGATION (BARRE LATÉRALE)
# ——————————————————————————————————————————————————————————————

st.sidebar.title("📊 POC Scoring Équitable")
page_options: List[str] = [
    "Analyse Exploratoire (EDA)",
    "Résultats & Comparaisons",
    "Prédiction sur Client Sélectionné",
    "Courbes ROC & Probabilités - Baseline",
    "Courbes ROC & Probabilités - EO Wrapper",
]
default_page_index: int = 0
session_key_page_index = "current_page_index_poc_scoring_roc_conf"

# Initialisation si nécessaire
if session_key_page_index not in st.session_state:
    st.session_state[session_key_page_index] = default_page_index

# On affiche le radio, avec la valeur actuelle dans session_state
page: str = st.sidebar.radio(
    "Navigation",
    page_options,
    index=st.session_state[session_key_page_index],
    key="nav_radio_poc_scoring_roc_conf",
)

# Si la sélection a changé, on met simplement à jour session_state.
# Streamlit va recharger/rafraîchir le script automatiquement, donc on n’a pas besoin de st.experimental_rerun().
if page_options.index(page) != st.session_state[session_key_page_index]:
    st.session_state[session_key_page_index] = page_options.index(page)

# ——————————————————————————————————————————————————————————————
# PAGE : Analyse Exploratoire (EDA)
# ——————————————————————————————————————————————————————————————
if page == "Analyse Exploratoire (EDA)":
    # 1) Imports spécifiques à l’EDA
    import pandas as pd
    import numpy as np
    import plotly.express as px

    # 2) Téléchargement du CSV pour EDA
    download_if_missing(RAW_DATA_FILENAME, ARTEFACTS[RAW_DATA_FILENAME])

    # 3) Chargement de l’échantillon
    df_eda_sample = load_csv_sample_for_eda(RAW_DATA_FILENAME, sample_frac=0.05)
    st.header("🔎 Analyse Exploratoire des Données (EDA)")
    st.caption(
        f"Basée sur un échantillon de {len(df_eda_sample) if df_eda_sample is not None else 0} lignes."
    )

    if df_eda_sample is not None and not df_eda_sample.empty:
        st.subheader("Aperçu des données brutes (échantillon)")
        st.dataframe(df_eda_sample.head(), use_container_width=True)

        st.subheader("Statistiques descriptives (variables numériques)")
        st.dataframe(df_eda_sample.describe(include=np.number).T, use_container_width=True)

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

            try:
                fig_target_hist = px.histogram(
                    df_eda_sample,
                    x="TARGET",
                    color="TARGET",
                    title="Histogramme de la variable cible 'TARGET'",
                    labels={"TARGET": "Classe de défaut (0: Non-défaut, 1: Défaut)"},
                    text_auto=True,
                )
                fig_target_hist.update_layout(bargap=0.2)
                st.plotly_chart(fig_target_hist, use_container_width=True)
            except Exception as e:
                st.warning(f"Impossible de générer l'histogramme de TARGET: {e}")
        else:
            st.warning("La colonne 'TARGET' n'est pas présente dans l’échantillon.")

        numerical_col_for_eda = "AMT_INCOME_TOTAL"
        if numerical_col_for_eda in df_eda_sample.columns:
            st.subheader(f"Distribution de '{numerical_col_for_eda}'")
            income_cap = df_eda_sample[numerical_col_for_eda].quantile(0.99)
            df_filtered_income = df_eda_sample[
                df_eda_sample[numerical_col_for_eda] < income_cap
            ]

            try:
                fig_income_dist = px.histogram(
                    df_filtered_income,
                    x=numerical_col_for_eda,
                    color="TARGET" if "TARGET" in df_filtered_income else None,
                    marginal="box",
                    title=f"Distribution de '{numerical_col_for_eda}' (plafonné à {income_cap:,.0f})",
                    labels={numerical_col_for_eda: "Revenu total", "TARGET": "Classe de défaut"},
                )
                st.plotly_chart(fig_income_dist, use_container_width=True)
            except Exception as e:
                st.warning(f"Impossible de générer l'histogramme de {numerical_col_for_eda}: {e}")
        else:
            st.info(f"La colonne '{numerical_col_for_eda}' n'est pas disponible pour l'EDA.")
    else:
        st.error("L’échantillon de données pour l’EDA n’a pas pu être chargé.")

# ——————————————————————————————————————————————————————————————
# PAGE : Résultats & Comparaisons
# ——————————————————————————————————————————————————————————————
elif page == "Résultats & Comparaisons":
    # 1) Imports spécifiques à cette page (modèles, métriques, visualisations)
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import (
        roc_auc_score,
        roc_curve,
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )
    from fairlearn.metrics import (
        MetricFrame,
        selection_rate as fairlearn_selection_rate,
        demographic_parity_difference,
        equalized_odds_difference,
    )
    import plotly.express as px
    import plotly.figure_factory as ff

    # 2) Téléchargement des artefacts nécessaires pour la comparaison
    artefacts_to_download = [
        MODEL_BASELINE_FILENAME,
        BASELINE_THRESHOLD_FILENAME,
        MODEL_WRAPPED_EO_FILENAME,
        "X_test_pre.parquet",
        "y_test.parquet",
        "A_test.parquet",
    ]
    for fname in artefacts_to_download:
        download_if_missing(fname, ARTEFACTS[fname])

    # 3) Chargement des modèles et des données
    model_baseline = load_model_joblib(MODEL_BASELINE_FILENAME)
    optimal_thresh_baseline = load_model_joblib(BASELINE_THRESHOLD_FILENAME)
    if optimal_thresh_baseline is None:
        st.warning(f"Seuil baseline ('{BASELINE_THRESHOLD_FILENAME}') non trouvé. Fallback à 0.5.")
        optimal_thresh_baseline = 0.5
    else:
        st.sidebar.info(f"Seuil baseline : {optimal_thresh_baseline:.3f}")

    # Chargement du wrapper EO
    EOWrapper_class = ensure_eowrapper_in_main()
    model_eo_wrapper = None
    if EOWrapper_class is not None:
        model_eo_wrapper = load_model_joblib(MODEL_WRAPPED_EO_FILENAME)

    if model_baseline:
        st.sidebar.success("Modèle baseline chargé.")
    if model_eo_wrapper:
        st.sidebar.success("Modèle EO Wrapper chargé.")
        if hasattr(model_eo_wrapper, "threshold"):
            st.sidebar.info(f"Seuil EO Wrapper : {model_eo_wrapper.threshold:.4f}")
        else:
            st.sidebar.warning("L’objet EO Wrapper n’a pas d’attribut 'threshold'.")

    # Chargement des données de test
    X_test_raw = load_parquet_file("X_test_pre.parquet")
    y_test = load_parquet_file("y_test.parquet")
    A_test = load_parquet_file("A_test.parquet")

    X_test = None
    if X_test_raw is not None:
        X_test = sanitize_feature_names(X_test_raw)
        st.sidebar.info("Données X_test nettoyées.")
    if y_test is not None:
        y_test = y_test.squeeze()
        st.sidebar.info("Données y_test chargées.")
    if A_test is not None:
        A_test = A_test.squeeze()
        st.sidebar.info("Données A_test chargées.")

    st.header("📊 Résultats comparatifs sur jeu de test")

    if (
        model_baseline is not None
        and model_eo_wrapper is not None
        and X_test is not None
        and y_test is not None
        and A_test is not None
        and optimal_thresh_baseline is not None
    ):
        try:
            # Calcul Baseline
            y_test_proba_b: np.ndarray = model_baseline.predict_proba(X_test)[:, 1]
            y_test_pred_b: np.ndarray = (y_test_proba_b >= optimal_thresh_baseline).astype(int)
            metrics_b: Dict[str, float] = {
                "AUC": roc_auc_score(y_test, y_test_proba_b),
                "Accuracy": accuracy_score(y_test, y_test_pred_b),
                "Precision (1)": precision_score(y_test, y_test_pred_b, pos_label=1, zero_division=0),
                "Recall (1)": recall_score(y_test, y_test_pred_b, pos_label=1, zero_division=0),
                "F1 (1)": f1_score(y_test, y_test_pred_b, pos_label=1, zero_division=0),
                "Taux de sélection global": np.mean(y_test_pred_b),
            }
            fairness_b: Dict[str, float] = {
                "DPD": demographic_parity_difference(y_test, y_test_pred_b, sensitive_features=A_test),
                "EOD": equalized_odds_difference(y_test, y_test_pred_b, sensitive_features=A_test),
            }
            cm_b = confusion_matrix(y_test, y_test_pred_b)

            # Calcul EO Wrapper
            y_test_proba_eo: np.ndarray = model_eo_wrapper.predict_proba(X_test)[:, 1]
            y_test_pred_eo: np.ndarray = model_eo_wrapper.predict(X_test)
            metrics_eo: Dict[str, float] = {
                "AUC": roc_auc_score(y_test, y_test_proba_eo),
                "Accuracy": accuracy_score(y_test, y_test_pred_eo),
                "Precision (1)": precision_score(y_test, y_test_pred_eo, pos_label=1, zero_division=0),
                "Recall (1)": recall_score(y_test, y_test_pred_eo, pos_label=1, zero_division=0),
                "F1 (1)": f1_score(y_test, y_test_pred_eo, pos_label=1, zero_division=0),
                "Taux de sélection global": np.mean(y_test_pred_eo),
            }
            fairness_eo: Dict[str, float] = {
                "DPD": demographic_parity_difference(y_test, y_test_pred_eo, sensitive_features=A_test),
                "EOD": equalized_odds_difference(y_test, y_test_pred_eo, sensitive_features=A_test),
            }
            cm_eo = confusion_matrix(y_test, y_test_pred_eo)

            # Affichage des tableaux de métriques et fairness
            st.subheader("Tableau récapitulatif des métriques")
            df_res = pd.DataFrame(
                [
                    {"Modèle": "Baseline", **metrics_b, **fairness_b},
                    {"Modèle": "EO Wrapper", **metrics_eo, **fairness_eo},
                ]
            )
            st.dataframe(df_res.set_index("Modèle").style.format("{:.3f}", na_rep="-"), use_container_width=True)

            # Matrices de confusion (idem qu’avant)
            st.subheader("Matrices de Confusion")
            col1_cm, col2_cm = st.columns(2)
            labels_cm = ["Non-Défaut (0)", "Défaut (1)"]

            with col1_cm:
                st.markdown("**Modèle Baseline**")
                z_text_b = [[str(y) for y in x] for x in cm_b]
                fig_cm_b = ff.create_annotated_heatmap(
                    cm_b, x=labels_cm, y=labels_cm, annotation_text=z_text_b, colorscale="Blues"
                )
                fig_cm_b.update_layout(
                    title_text="<i>Baseline</i>",
                    xaxis_title="Prédit",
                    yaxis_title="Réel",
                )
                st.plotly_chart(fig_cm_b, use_container_width=True)

            with col2_cm:
                st.markdown("**Modèle EO Wrapper**")
                z_text_eo = [[str(y) for y in x] for x in cm_eo]
                fig_cm_eo = ff.create_annotated_heatmap(
                    cm_eo, x=labels_cm, y=labels_cm, annotation_text=z_text_eo, colorscale="Greens"
                )
                fig_cm_eo.update_layout(
                    title_text="<i>EO Wrapper</i>",
                    xaxis_title="Prédit",
                    yaxis_title="Réel",
                )
                st.plotly_chart(fig_cm_eo, use_container_width=True)

            # Taux de sélection par groupe (MetricFrame)
            st.subheader("Taux de Sélection par Groupe (Feature Sensible)")
            mf_selection_baseline = MetricFrame(
                metrics=fairlearn_selection_rate,
                y_true=y_test,
                y_pred=y_test_pred_b,
                sensitive_features=A_test,
            )
            mf_selection_eo = MetricFrame(
                metrics=fairlearn_selection_rate,
                y_true=y_test,
                y_pred=y_test_pred_eo,
                sensitive_features=A_test,
            )

            df_selection_rates = (
                pd.DataFrame(
                    {
                        "Groupe Sensible": mf_selection_baseline.by_group.index,
                        "Taux Sélection Baseline": mf_selection_baseline.by_group.values,
                        "Taux Sélection EO Wrapper": mf_selection_eo.by_group.values,
                    }
                )
                .set_index("Groupe Sensible")
            )
            st.dataframe(df_selection_rates.style.format("{:.3f}"), use_container_width=True)

            df_selection_plot = df_selection_rates.reset_index().melt(
                id_vars="Groupe Sensible",
                value_vars=["Taux Sélection Baseline", "Taux Sélection EO Wrapper"],
                var_name="Modèle",
                value_name="Taux de Sélection",
            )
            fig_sr = px.bar(
                df_selection_plot,
                x="Groupe Sensible",
                y="Taux de Sélection",
                color="Modèle",
                barmode="group",
                title="Taux de Sélection par Groupe et Modèle",
                labels={"Groupe Sensible": "Groupe (Feature Sensible)", "Taux de Sélection": "Taux d'approbation"},
            )
            st.plotly_chart(fig_sr, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur lors du calcul ou affichage des résultats comparatifs: {e}")
            st.exception(e)
    else:
        st.warning("Des éléments sont manquants pour afficher les résultats.")
        if model_baseline is None:
            st.error("- Modèle baseline non chargé.")
        if model_eo_wrapper is None:
            st.error("- Modèle EO wrapper non chargé.")
        if X_test is None:
            st.error("- Données X_test non chargées.")
        if y_test is None:
            st.error("- Données y_test non chargées.")
        if A_test is None:
            st.error("- Données A_test non chargées.")
        if optimal_thresh_baseline is None:
            st.error("- Seuil optimal baseline non chargé.")

# ——————————————————————————————————————————————————————————————
# (Tu peux appliquer la même logique de « lazy import + download » pour les pages suivantes…)
# ——————————————————————————————————————————————————————————————

elif page == "Prédiction sur Client Sélectionné":
    # 1) Imports spécifiques à la prédiction individuelle
    import pandas as pd
    import numpy as np
    import joblib

    # 2) Télécharger uniquement ce qui est nécessaire
    artefacts_pred = [
        MODEL_BASELINE_FILENAME,
        BASELINE_THRESHOLD_FILENAME,
        MODEL_WRAPPED_EO_FILENAME,
        "X_test_pre.parquet",
        "y_test.parquet",
        "A_test.parquet",
    ]
    for fname in artefacts_pred:
        download_if_missing(fname, ARTEFACTS[fname])

    # 3) Chargement des modèles et des données
    model_baseline = load_model_joblib(MODEL_BASELINE_FILENAME)
    optimal_thresh_baseline = load_model_joblib(BASELINE_THRESHOLD_FILENAME)
    EOWrapper_class = ensure_eowrapper_in_main()
    model_eo_wrapper = None
    if EOWrapper_class is not None:
        model_eo_wrapper = load_model_joblib(MODEL_WRAPPED_EO_FILENAME)

    X_test_raw = load_parquet_file("X_test_pre.parquet")
    y_test = load_parquet_file("y_test.parquet")
    A_test = load_parquet_file("A_test.parquet")

    X_test = None
    if X_test_raw is not None:
        X_test = sanitize_feature_names(X_test_raw)
    if y_test is not None:
        y_test = y_test.squeeze()

    st.header("🔍 Prédiction sur un Client Sélectionné du Jeu de Test")

    if (
        X_test is not None
        and model_baseline is not None
        and model_eo_wrapper is not None
        and optimal_thresh_baseline is not None
        and y_test is not None
    ):
        client_ids = X_test.index.tolist()
        if not client_ids:
            st.warning("Aucun ID client disponible dans le jeu de test.")
        else:
            max_clients_in_selectbox = 2000
            client_ids_to_display = (
                client_ids[:max_clients_in_selectbox]
                if len(client_ids) > max_clients_in_selectbox
                else client_ids
            )
            if len(client_ids) > max_clients_in_selectbox:
                st.info(f"Affichage des {max_clients_in_selectbox} premiers IDs clients.")

            selected_client_id_str = st.selectbox(
                "Choisis un ID client du jeu de test :", options=[str(id_val) for id_val in client_ids_to_display]
            )

            try:
                if X_test.index.dtype in ("int64", "int32"):
                    selected_client_id = int(selected_client_id_str)
                elif X_test.index.dtype in ("float64", "float32"):
                    selected_client_id = float(selected_client_id_str)
                else:
                    selected_client_id = selected_client_id_str
            except ValueError:
                st.error(f"ID client '{selected_client_id_str}' invalide.")
                st.stop()

            if selected_client_id in X_test.index:
                client_features = X_test.loc[[selected_client_id]]
                client_true_target = y_test.loc[selected_client_id] if selected_client_id in y_test.index else "Inconnue"

                st.subheader(f"Données du client ID : {selected_client_id}")
                st.write(f"Vraie cible (TARGET) : **{client_true_target}**")
                st.dataframe(client_features.T.rename(columns={0: "Valeur"}), use_container_width=True)

                try:
                    proba_baseline = model_baseline.predict_proba(client_features)[0, 1]
                    pred_baseline = (proba_baseline >= optimal_thresh_baseline).astype(int)

                    proba_eo = model_eo_wrapper.predict_proba(client_features)[0, 1]
                    pred_eo = model_eo_wrapper.predict(client_features)[0]

                    st.subheader("Résultats de la Prédiction")
                    results_data = {
                        "Métrique": ["Probabilité de défaut (classe 1)", "Prédiction (0 ou 1)"],
                        "Modèle Baseline": [f"{proba_baseline:.4f}", pred_baseline],
                        "Modèle EO Wrapper": [f"{proba_eo:.4f}", pred_eo],
                    }
                    df_pred_results = pd.DataFrame(results_data)
                    st.table(df_pred_results.set_index("Métrique"))
                except Exception as e_pred:
                    st.error(f"Erreur lors de la prédiction pour {selected_client_id} : {e_pred}")
            else:
                st.error(f"L’ID client {selected_client_id} n’a pas été trouvé.")
    else:
        st.warning("Modèle, données de test ou seuil manquant pour faire la prédiction.")

elif page == "Courbes ROC & Probabilités - Baseline":
    # 1) Imports
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import roc_auc_score, roc_curve
    import plotly.graph_objects as go
    import plotly.express as px

    # 2) Téléchargement
    artefacts_roc_b = [
        MODEL_BASELINE_FILENAME,
        BASELINE_THRESHOLD_FILENAME,
        "X_valid_pre.parquet",
        "y_valid.parquet",
    ]
    for fname in artefacts_roc_b:
        download_if_missing(fname, ARTEFACTS[fname])

    # 3) Chargement
    model_baseline = load_model_joblib(MODEL_BASELINE_FILENAME)
    optimal_thresh_baseline = load_model_joblib(BASELINE_THRESHOLD_FILENAME)
    X_valid_raw = load_parquet_file("X_valid_pre.parquet")
    y_valid = load_parquet_file("y_valid.parquet")

    X_valid = sanitize_feature_names(X_valid_raw) if X_valid_raw is not None else None
    if y_valid is not None:
        y_valid = y_valid.squeeze()

    st.header("Courbes ROC & Distribution des Probabilités - Baseline")
    st.caption("Calculé sur le jeu de validation.")

    if model_baseline and X_valid is not None and y_valid is not None and optimal_thresh_baseline is not None:
        try:
            y_val_proba_b: np.ndarray = model_baseline.predict_proba(X_valid)[:, 1]
            fpr, tpr, thresholds_roc = roc_curve(y_valid, y_val_proba_b)
            auc_b_val: float = roc_auc_score(y_valid, y_val_proba_b)

            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"ROC Baseline (AUC={auc_b_val:.3f})",
                    line=dict(color="blue"),
                )
            )
            fig_roc.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash", color="grey"))

            optimal_idx_val = np.argmin(np.abs(thresholds_roc - optimal_thresh_baseline))
            fig_roc.add_trace(
                go.Scatter(
                    x=[fpr[optimal_idx_val]],
                    y=[tpr[optimal_idx_val]],
                    mode="markers",
                    marker=dict(size=10, color="red"),
                    name=f"Seuil Optimal ({optimal_thresh_baseline:.3f})",
                )
            )
            fig_roc.update_layout(
                title_text="Courbe ROC - Modèle Baseline (Validation)",
                xaxis_title="Taux de Faux Positifs (FPR)",
                yaxis_title="Taux de Vrais Positifs (TPR)",
                legend_title_text="Légende",
            )
            st.plotly_chart(fig_roc, use_container_width=True)

            # Distribution des scores
            df_dist_b = pd.DataFrame({"proba_baseline": y_val_proba_b, "y_true": y_valid.astype(str)})
            fig_dist_b = px.histogram(
                df_dist_b,
                x="proba_baseline",
                color="y_true",
                nbins=50,
                barmode="overlay",
                marginal="rug",
                title="Distribution des Scores (Baseline)",
                labels={"proba_baseline": "Score Prédit (Baseline)", "y_true": "Vraie Cible"},
                color_discrete_map={"0": "green", "1": "red"},
            )
            fig_dist_b.add_vline(
                x=optimal_thresh_baseline,
                line_color="black",
                line_dash="dash",
                annotation_text=f"Seuil={optimal_thresh_baseline:.3f}",
                annotation_position="top right",
            )
            st.plotly_chart(fig_dist_b, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur génération graphiques Baseline : {e}")
            st.exception(e)
    else:
        st.warning("Modèle Baseline, données de validation ou seuil manquant.")

elif page == "Courbes ROC & Probabilités - EO Wrapper":
    # 1) Imports
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import roc_auc_score, roc_curve
    import plotly.graph_objects as go
    import plotly.express as px

    # 2) Téléchargement
    artefacts_roc_eo = [
        MODEL_WRAPPED_EO_FILENAME,
        "X_valid_pre.parquet",
        "y_valid.parquet",
    ]
    for fname in artefacts_roc_eo:
        download_if_missing(fname, ARTEFACTS[fname])

    # 3) Chargement
    EOWrapper_class = ensure_eowrapper_in_main()
    model_eo_wrapper = None
    if EOWrapper_class is not None:
        model_eo_wrapper = load_model_joblib(MODEL_WRAPPED_EO_FILENAME)

    X_valid_raw = load_parquet_file("X_valid_pre.parquet")
    y_valid = load_parquet_file("y_valid.parquet")

    X_valid = sanitize_feature_names(X_valid_raw) if X_valid_raw is not None else None
    if y_valid is not None:
        y_valid = y_valid.squeeze()

    st.header("Courbes ROC & Distribution des Probabilités - EO Wrapper")
    st.caption("Calculé sur le jeu de validation.")

    if model_eo_wrapper and hasattr(model_eo_wrapper, "threshold") and X_valid is not None and y_valid is not None:
        try:
            wrapper_threshold = model_eo_wrapper.threshold
            y_val_proba_eo: np.ndarray = model_eo_wrapper.predict_proba(X_valid)[:, 1]
            fpr_eo, tpr_eo, thresholds_roc_eo = roc_curve(y_valid, y_val_proba_eo)
            auc_eo_val: float = roc_auc_score(y_valid, y_val_proba_eo)

            fig_roc_eo = go.Figure()
            fig_roc_eo.add_trace(
                go.Scatter(
                    x=fpr_eo,
                    y=tpr_eo,
                    mode="lines",
                    name=f"ROC EO Wrapper (AUC={auc_eo_val:.3f})",
                    line=dict(color="green"),
                )
            )
            fig_roc_eo.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash", color="grey"))

            optimal_idx_eo_val = np.argmin(np.abs(thresholds_roc_eo - wrapper_threshold))
            fig_roc_eo.add_trace(
                go.Scatter(
                    x=[fpr_eo[optimal_idx_eo_val]],
                    y=[tpr_eo[optimal_idx_eo_val]],
                    mode="markers",
                    marker=dict(size=10, color="orange"),
                    name=f"Seuil du Wrapper ({wrapper_threshold:.3f})",
                )
            )
            fig_roc_eo.update_layout(
                title_text="Courbe ROC - Modèle EO Wrapper (Validation)",
                xaxis_title="Taux de Faux Positifs (FPR)",
                yaxis_title="Taux de Vrais Positifs (TPR)",
                legend_title_text="Légende",
            )
            st.plotly_chart(fig_roc_eo, use_container_width=True)

            # Distribution des scores EO
            df_dist_eo = pd.DataFrame({"proba_eo": y_val_proba_eo, "y_true": y_valid.astype(str)})
            fig_dist_eo = px.histogram(
                df_dist_eo,
                x="proba_eo",
                color="y_true",
                nbins=50,
                barmode="overlay",
                marginal="rug",
                title="Distribution des Scores (EO Wrapper)",
                labels={"proba_eo": "Score Prédit (EO Wrapper)", "y_true": "Vraie Cible"},
                color_discrete_map={"0": "green", "1": "red"},
            )
            fig_dist_eo.add_vline(
                x=wrapper_threshold,
                line_color="black",
                line_dash="dash",
                annotation_text=f"Seuil={wrapper_threshold:.3f}",
                annotation_position="top right",
            )
            st.plotly_chart(fig_dist_eo, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur génération graphiques EO Wrapper : {e}")
            st.exception(e)
    else:
        st.warning("EO Wrapper, son seuil ou données de validation manquantes.")

# Fin du script
