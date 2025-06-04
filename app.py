# --------------------------------------------
# fichier : app_streamlit_dashboard.py
# --------------------------------------------

import os
import re
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import streamlit as st

import plotly.express as px
import plotly.figure_factory as ff

from fairlearn.metrics import (
    MetricFrame,
    selection_rate as fairlearn_selection_rate,
    equalized_odds_difference,
)

# ——————————————————————————————————————————————————————————————
# CONSTANTES ET CHEMINS
# ——————————————————————————————————————————————————————————————
RAW_DATA_FILENAME: str = "application_train.csv"
PREDICTIONS_FILENAME: str = "predictions_validation.parquet"

ARTEFACTS: Dict[str, str] = {
    RAW_DATA_FILENAME: (
        "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/"
        "main/application_train.csv"
    ),
    PREDICTIONS_FILENAME: (
        "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/"
        "main/predictions_validation.parquet"
    ),
}

# ——————————————————————————————————————————————————————————————
# FONCTIONS UTILITAIRES
# ——————————————————————————————————————————————————————————————
def download_if_missing(filename: str, url: str) -> None:
    """
    Télécharge le fichier depuis Hugging Face si absent localement.
    """
    if not os.path.exists(filename):
        st.info(f"Téléchargement de {filename}…")
        try:
            import requests  # import paresseux

            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            st.success(f"{filename} téléchargé.")
        except Exception as e:
            st.error(f"Erreur lors du téléchargement de {filename} : {e}")
            if hasattr(e, "response") and getattr(e, "response", None) is not None:
                st.error(
                    f"Réponse du serveur : "
                    f"{e.response.status_code} – {e.response.text}"
                )
            st.stop()


@st.cache_data
def load_csv_for_eda(path: str, sample_frac: float = 0.05) -> Optional[pd.DataFrame]:
    """
    Charge un échantillon du CSV brut pour l’EDA.
    """
    try:
        df = pd.read_csv(path)
        if 0.0 < sample_frac < 1.0 and len(df) * sample_frac >= 1:
            df = df.sample(frac=sample_frac, random_state=42)
        return df
    except FileNotFoundError:
        st.error(f"Fichier EDA non trouvé : {path}")
        return None
    except Exception as e:
        st.error(f"Erreur de chargement du CSV pour l’EDA {path} : {e}")
        return None


@st.cache_data
def load_parquet_file(path: str) -> Optional[pd.DataFrame]:
    """
    Charge un fichier Parquet.
    """
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        st.error(f"Parquet non trouvé : {path}")
        return None
    except Exception as e:
        st.error(f"Erreur de chargement du Parquet {path} : {e}")
        return None


def sanitize_feature_names(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace les caractères non alphanumériques ou underscore dans les noms de colonnes
    par un underscore.
    """
    df = df_input.copy()
    cleaned_columns = [re.sub(r"[^0-9a-zA-Z_]", "_", str(col)) for col in df.columns]
    df.columns = cleaned_columns
    return df


# ——————————————————————————————————————————————————————————————
# CONFIGURATION DE STREAMLIT
# ——————————————————————————————————————————————————————————————
st.set_page_config(
    page_title="POC Scoring Équitable (Dashboard)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ——————————————————————————————————————————————————————————————
# TÉLÉCHARGEMENT DES ARTEFACTS (raw CSV + prédictions)
# ——————————————————————————————————————————————————————————————
for fname, url in ARTEFACTS.items():
    download_if_missing(fname, url)

# Chargement des données
df_eda_sample = load_csv_for_eda(RAW_DATA_FILENAME, sample_frac=0.05)
df_preds = load_parquet_file(PREDICTIONS_FILENAME)

# Pour intersectionnalité, on fusionne les prédictions avec le CSV complet
df_application: Optional[pd.DataFrame] = None
if df_preds is not None:
    try:
        df_application = pd.read_csv(RAW_DATA_FILENAME, index_col=0)
        df_application = sanitize_feature_names(df_application)
        # On joint en utilisant l’index ; df_preds.index correspond à X_valid.index,
        # qui provient de l’index de application_train initial. 
        df_merged = df_application.join(df_preds, how="inner")
    except Exception as e:
        st.error(f"Erreur lors de la fusion application+predictions : {e}")
        df_merged = None
else:
    df_merged = None

# ——————————————————————————————————————————————————————————————
# NAVIGATION
# ——————————————————————————————————————————————————————————————
st.sidebar.title("📊 POC Scoring Équitable")
page_options: List[str] = [
    "Analyse Exploratoire (EDA)",
    "Résultats & Comparaisons",
    "Prédiction sur Client Sélectionné",
    "Analyse Intersectionnelle",
    "Courbes ROC & Probabilités - Baseline",
    "Courbes ROC & Probabilités - EO Wrapper",
]
session_key = "current_page_index_poc_scoring_dashboard"
if session_key not in st.session_state:
    st.session_state[session_key] = 0

page: str = st.sidebar.radio(
    "Navigation",
    page_options,
    index=st.session_state[session_key],
    key="nav_radio_poc_scoring_dashboard",
)
if page_options.index(page) != st.session_state[session_key]:
    st.session_state[session_key] = page_options.index(page)


# ——————————————————————————————————————————————————————————————
# PAGE : Analyse Exploratoire (EDA)
# ——————————————————————————————————————————————————————————————
if page == "Analyse Exploratoire (EDA)":
    st.header("🔎 Analyse Exploratoire des Données (EDA)")
    st.caption(
        f"Basée sur un échantillon de "
        f"{len(df_eda_sample) if df_eda_sample is not None else 0} lignes."
    )

    if df_eda_sample is not None and not df_eda_sample.empty:
        st.subheader("Aperçu des données brutes (échantillon)")
        st.dataframe(df_eda_sample.head(), use_container_width=True)

        st.subheader("Statistiques descriptives (variables numériques)")
        st.dataframe(
            df_eda_sample.describe(include=np.number).T, use_container_width=True
        )

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
                st.warning(f"Impossible de générer l'histogramme de TARGET : {e}")
        else:
            st.warning("La colonne 'TARGET' n’est pas présente dans l’échantillon.")

        numerical_col = "AMT_INCOME_TOTAL"
        if numerical_col in df_eda_sample.columns:
            st.subheader(f"Distribution de '{numerical_col}'")
            df_positive = df_eda_sample[df_eda_sample[numerical_col] > 0]
            if not df_positive.empty:
                cap = df_positive[numerical_col].quantile(0.99)
                df_filtered = df_positive[df_positive[numerical_col] < cap]
            else:
                df_filtered = df_eda_sample
                cap = df_eda_sample[numerical_col].max() if not df_eda_sample.empty else 0

            try:
                fig_income = px.histogram(
                    df_filtered,
                    x=numerical_col,
                    color="TARGET" if "TARGET" in df_filtered.columns else None,
                    marginal="box",
                    title=f"Distribution de '{numerical_col}' "
                    f"(plafonné à {cap:,.0f} si applicable)",
                    labels={numerical_col: "Revenu total", "TARGET": "Classe de défaut"},
                )
                st.plotly_chart(fig_income, use_container_width=True)
            except Exception as e:
                st.warning(f"Impossible de générer l'histogramme de {numerical_col} : {e}")
        else:
            st.info(f"La colonne '{numerical_col}' n’est pas disponible pour l’EDA.")
    else:
        st.error("L’échantillon pour l’EDA n’a pas pu être chargé.")


# ——————————————————————————————————————————————————————————————
# PAGE : Résultats & Comparaisons
# ——————————————————————————————————————————————————————————————
elif page == "Résultats & Comparaisons":
    st.header("📊 Résultats comparatifs (jeu de validation)")
    if df_preds is not None:
        try:
            # Calcul des métriques globales
            y_true = df_preds["y_true"]
            y_pred_b = df_preds["y_pred_baseline"]
            y_pred_e = df_preds["y_pred_eo"]
            proba_b = df_preds["proba_baseline"]
            proba_e = df_preds["proba_eo"]
            sens = df_preds["sensitive_feature"]

            # Metrics classification
            from sklearn.metrics import (
                roc_auc_score,
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                confusion_matrix,
            )

            metrics_b = {
                "AUC": roc_auc_score(y_true, proba_b),
                "Accuracy": accuracy_score(y_true, y_pred_b),
                "Precision (1)": precision_score(y_true, y_pred_b, pos_label=1, zero_division=0),
                "Recall (1)": recall_score(y_true, y_pred_b, pos_label=1, zero_division=0),
                "F1 (1)": f1_score(y_true, y_pred_b, pos_label=1, zero_division=0),
                "Taux de sélection global": np.mean(y_pred_b),
            }
            metrics_e = {
                "AUC": roc_auc_score(y_true, proba_e),
                "Accuracy": accuracy_score(y_true, y_pred_e),
                "Precision (1)": precision_score(y_true, y_pred_e, pos_label=1, zero_division=0),
                "Recall (1)": recall_score(y_true, y_pred_e, pos_label=1, zero_division=0),
                "F1 (1)": f1_score(y_true, y_pred_e, pos_label=1, zero_division=0),
                "Taux de sélection global": np.mean(y_pred_e),
            }

            df_metrics = pd.DataFrame(
                [
                    {"Modèle": "Baseline", **metrics_b},
                    {"Modèle": "EO Wrapper", **metrics_e},
                ]
            ).set_index("Modèle")
            st.subheader("Métriques de classification")
            st.dataframe(df_metrics.style.format("{:.3f}", na_rep="-"), use_container_width=True)

            # Fairness metrics
            fair_b = {
                "DPD": demographic_parity_difference(y_true, y_pred_b, sensitive_features=sens),
                "EOD": equalized_odds_difference(y_true, y_pred_b, sensitive_features=sens),
            }
            fair_e = {
                "DPD": demographic_parity_difference(y_true, y_pred_e, sensitive_features=sens),
                "EOD": equalized_odds_difference(y_true, y_pred_e, sensitive_features=sens),
            }
            df_fair = pd.DataFrame(
                [
                    {"Modèle": "Baseline", **fair_b},
                    {"Modèle": "EO Wrapper", **fair_e},
                ]
            ).set_index("Modèle")
            st.subheader("Métriques d’équité (global)")
            st.dataframe(df_fair.style.format("{:.3f}", na_rep="-"), use_container_width=True)

            # Matrices de confusion
            st.subheader("Matrices de Confusion")
            cm_b = confusion_matrix(y_true, y_pred_b)
            cm_e = confusion_matrix(y_true, y_pred_e)

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
                z_text_e = [[str(y) for y in x] for x in cm_e]
                fig_cm_e = ff.create_annotated_heatmap(
                    cm_e, x=labels_cm, y=labels_cm, annotation_text=z_text_e, colorscale="Greens"
                )
                fig_cm_e.update_layout(
                    title_text="<i>EO Wrapper</i>",
                    xaxis_title="Prédit",
                    yaxis_title="Réel",
                )
                st.plotly_chart(fig_cm_e, use_container_width=True)

            # Taux de sélection par groupe sensible
            st.subheader("Taux de sélection par groupe sensible")
            mf_b = MetricFrame(
                metrics=fairlearn_selection_rate,
                y_true=y_true,
                y_pred=y_pred_b,
                sensitive_features=sens,
            )
            mf_e = MetricFrame(
                metrics=fairlearn_selection_rate,
                y_true=y_true,
                y_pred=y_pred_e,
                sensitive_features=sens,
            )

            df_sel = pd.DataFrame(
                {
                    "Groupe sensible": mf_b.by_group.index,
                    "Taux sélection Baseline": mf_b.by_group.values,
                    "Taux sélection EO Wrapper": mf_e.by_group.values,
                }
            ).set_index("Groupe sensible")
            st.dataframe(df_sel.style.format("{:.3f}"), use_container_width=True)

            # Barplot des taux de sélection
            df_sel_plot = df_sel.reset_index().melt(
                id_vars="Groupe sensible",
                value_vars=["Taux sélection Baseline", "Taux sélection EO Wrapper"],
                var_name="Modèle",
                value_name="Taux de sélection",
            )
            fig_sel = px.bar(
                df_sel_plot,
                x="Groupe sensible",
                y="Taux de sélection",
                color="Modèle",
                barmode="group",
                title="Taux de sélection par groupe sensible et par modèle",
                labels={"Groupe sensible": "Groupe sensible", "Taux de sélection": "Taux d’approbation"},
            )
            st.plotly_chart(fig_sel, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur lors du calcul/affichage des résultats : {e}")
            st.exception(e)
    else:
        st.warning("Le fichier de prédictions n’a pas pu être chargé.")


# ——————————————————————————————————————————————————————————————
# PAGE : Prédiction sur Client Sélectionné
# ——————————————————————————————————————————————————————————————
elif page == "Prédiction sur Client Sélectionné":
    st.header("🔍 Résultats enregistrés pour un client (validation)")
    if df_preds is not None:
        client_ids = df_preds.index.tolist()
        if not client_ids:
            st.warning("Aucun ID disponible dans le fichier de prédictions.")
        else:
            max_display = 2000
            ids_to_display = client_ids[:max_display] if len(client_ids) > max_display else client_ids
            if len(client_ids) > max_display:
                st.info(f"Affichage des {max_display} premiers IDs seulement.")
            selected_id = st.selectbox(
                "Choisis un ID client (validation) :", options=[str(i) for i in ids_to_display]
            )
            try:
                sel_id = int(selected_id)
            except ValueError:
                sel_id = selected_id

            if sel_id in df_preds.index:
                row = df_preds.loc[sel_id]
                st.subheader(f"Client ID : {sel_id}")
                st.write(f"Vraie cible (TARGET) : **{row['y_true']}**")
                st.write(f"Probabilité Baseline : **{row['proba_baseline']:.4f}**")
                st.write(f"Prédiction Baseline : **{row['y_pred_baseline']}**")
                st.write(f"Probabilité EO : **{row['proba_eo']:.4f}**")
                st.write(f"Prédiction EO : **{row['y_pred_eo']}**")
                st.write(f"Groupe sensible : **{row['sensitive_feature']}**")
            else:
                st.error(f"L’ID {sel_id} n’est pas présent dans le jeu de validation.")
    else:
        st.warning("Le fichier de prédictions n’a pas pu être chargé.")


# ——————————————————————————————————————————————————————————————
# PAGE : Analyse Intersectionnelle
# ——————————————————————————————————————————————————————————————
elif page == "Analyse Intersectionnelle":
    st.header("🔀 Analyse Intersectionnelle")
    st.caption(
        "Choisis une feature catégorielle pour évaluer les métriques "
        "de sélection et d’équité selon ses modalités."
    )

    if df_merged is not None:
        # Récupérer uniquement les colonnes catégorielles
        categorical_cols = df_merged.select_dtypes(include=["object", "category"]).columns.tolist()
        if not categorical_cols:
            st.warning("Aucune colonne catégorielle n’a été trouvée.")
        else:
            chosen_col = st.selectbox("Choisis une colonne catégorielle :", categorical_cols)
            # Filtrer les modalités non-nulles
            modalities = df_merged[chosen_col].dropna().unique().tolist()
            if not modalities:
                st.error(f"Aucune modalité valide pour {chosen_col}.")
            else:
                # Calculer, pour chaque modalité, le taux de sélection et l’EOD
                results = []
                for mod in modalities:
                    subset = df_merged[df_merged[chosen_col] == mod]
                    if subset.empty:
                        continue

                    # Sélection rate pour Baseline et EO
                    sel_base = subset["y_pred_baseline"].mean()
                    sel_eo = subset["y_pred_eo"].mean()

                    # EOD (différence d’égalisation des odds) dans la modalité
                    try:
                        eod_mod = equalized_odds_difference(
                            subset["y_true"],
                            subset["y_pred_baseline"],
                            sensitive_features=subset["sensitive_feature"],
                        )
                    except Exception:
                        eod_mod = np.nan

                    results.append(
                        {
                            "Modalité": mod,
                            "SelRate_Baseline": sel_base,
                            "SelRate_EO": sel_eo,
                            "EOD_Baseline": eod_mod,
                        }
                    )

                df_inter = pd.DataFrame(results).set_index("Modalité")
                st.subheader(f"Métriques par modalité de '{chosen_col}'")
                st.dataframe(df_inter.style.format({col: "{:.3f}" for col in df_inter.columns}), use_container_width=True)

                # Graphique comparatif des taux de sélection
                fig_inter_sel = px.bar(
                    df_inter.reset_index().melt(
                        id_vars="Modalité",
                        value_vars=["SelRate_Baseline", "SelRate_EO"],
                        var_name="Modèle",
                        value_name="Taux de sélection",
                    ),
                    x="Modalité",
                    y="Taux de sélection",
                    color="Modèle",
                    barmode="group",
                    title=f"Taux de sélection par modalités de '{chosen_col}'",
                    labels={"Modalité": chosen_col},
                )
                st.plotly_chart(fig_inter_sel, use_container_width=True)

                # Graphique EOD
                fig_inter_eod = px.bar(
                    df_inter.reset_index(),
                    x="Modalité",
                    y="EOD_Baseline",
                    title=f"EOD (Baseline) par modalités de '{chosen_col}'",
                    labels={"EOD_Baseline": "Equalized Odds Diff"},
                )
                st.plotly_chart(fig_inter_eod, use_container_width=True)
    else:
        st.warning("Fusion des données application + prédictions impossible.")


# ——————————————————————————————————————————————————————————————
# PAGE : Courbes ROC & Probabilités - Baseline
# ——————————————————————————————————————————————————————————————
elif page == "Courbes ROC & Probabilités - Baseline":
    st.header("Courbes ROC & Distribution des Probabilités - Baseline")
    st.caption("Basé sur le jeu de validation enregistré dans 'predictions_validation.parquet'.")

    if df_preds is not None:
        try:
            from sklearn.metrics import roc_auc_score, roc_curve

            y_true = df_preds["y_true"]
            proba_b = df_preds["proba_baseline"]

            fpr, tpr, thresholds = roc_curve(y_true, proba_b)
            auc_val = roc_auc_score(y_true, proba_b)

            fig_roc = px.line(
                x=fpr,
                y=tpr,
                labels={"x": "Taux de faux positifs (FPR)", "y": "Taux de vrais positifs (TPR)"},
                title=f"ROC Baseline (AUC = {auc_val:.3f})",
            )
            fig_roc.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash", color="gray"))
            st.plotly_chart(fig_roc, use_container_width=True)

            # Distribution des probabilités
            df_dist = pd.DataFrame({"proba_baseline": proba_b, "y_true": y_true.astype(str)})
            fig_dist = px.histogram(
                df_dist,
                x="proba_baseline",
                color="y_true",
                nbins=50,
                barmode="overlay",
                marginal="rug",
                title="Distribution des scores (Baseline)",
                labels={"proba_baseline": "Score Baseline", "y_true": "Vraie cible"},
                color_discrete_map={"0": "green", "1": "red"},
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur lors de la génération des graphiques : {e}")
            st.exception(e)
    else:
        st.warning("Le fichier de prédictions n’a pas pu être chargé.")


# ——————————————————————————————————————————————————————————————
# PAGE : Courbes ROC & Probabilités - EO Wrapper
# ——————————————————————————————————————————————————————————————
elif page == "Courbes ROC & Probabilités - EO Wrapper":
    st.header("Courbes ROC & Distribution des Probabilités - EO Wrapper")
    st.caption("Basé sur le jeu de validation enregistré dans 'predictions_validation.parquet'.")

    if df_preds is not None:
        try:
            from sklearn.metrics import roc_auc_score, roc_curve

            y_true = df_preds["y_true"]
            proba_e = df_preds["proba_eo"]

            fpr_e, tpr_e, thresholds_e = roc_curve(y_true, proba_e)
            auc_e_val = roc_auc_score(y_true, proba_e)

            fig_roc_e = px.line(
                x=fpr_e,
                y=tpr_e,
                labels={"x": "Taux de faux positifs (FPR)", "y": "Taux de vrais positifs (TPR)"},
                title=f"ROC EO Wrapper (AUC = {auc_e_val:.3f})",
            )
            fig_roc_e.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash", color="gray"))
            st.plotly_chart(fig_roc_e, use_container_width=True)

            # Distribution des probabilités EO
            df_dist_e = pd.DataFrame({"proba_eo": proba_e, "y_true": y_true.astype(str)})
            fig_dist_e = px.histogram(
                df_dist_e,
                x="proba_eo",
                color="y_true",
                nbins=50,
                barmode="overlay",
                marginal="rug",
                title="Distribution des scores (EO Wrapper)",
                labels={"proba_eo": "Score EO", "y_true": "Vraie cible"},
                color_discrete_map={"0": "green", "1": "red"},
            )
            st.plotly_chart(fig_dist_e, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur lors de la génération des graphiques : {e}")
            st.exception(e)
    else:
        st.warning("Le fichier de prédictions n’a pas pu être chargé.")


# ——————————————————————————————————————————————————————————————
# FIN
# ——————————————————————————————————————————————————————————————
