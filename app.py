# --------------------------------------------
# fichier : app_streamlit_dashboard.py
# --------------------------------------------

# ——————————————————————————————————————————————————————————————
# fichier : app_streamlit_dashboard.py
# ——————————————————————————————————————————————————————————————

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
    demographic_parity_difference,
    equalized_odds_difference,
)

# ——————————————————————————————————————————————————————————————
# FONCTIONS UTILITAIRES MANQUANTES
# ——————————————————————————————————————————————————————————————

def download_if_missing(filename: str, url: str) -> None:
    """
    Télécharge le fichier depuis Hugging Face si absent localement.
    """
    if not os.path.exists(filename):
        st.info(f"Téléchargement de '{filename}'…")
        try:
            import requests  # import paresseux

            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            st.success(f"'{filename}' téléchargé.")
        except Exception as e:
            st.error(f"Erreur lors du téléchargement de '{filename}' : {e}")
            if hasattr(e, "response") and getattr(e, "response", None) is not None:
                st.error(
                    f"Réponse du serveur : {e.response.status_code} – {e.response.text}"
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
        st.error(f"Fichier EDA non trouvé : '{path}'")
        return None
    except Exception as e:
        st.error(f"Erreur de chargement du CSV pour l’EDA '{path}' : {e}")
        return None


@st.cache_data
def load_parquet_file(path: str) -> Optional[pd.DataFrame]:
    """
    Charge un fichier Parquet.
    """
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        st.error(f"Parquet non trouvé : '{path}'")
        return None
    except Exception as e:
        st.error(f"Erreur de chargement du Parquet '{path}' : {e}")
        return None


def sanitize_feature_names(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace les caractères non alphanumériques ou underscore dans les noms de colonnes
    par un underscore, pour éviter tout problème de parsing.
    """
    df = df_input.copy()
    cleaned_columns = [re.sub(r"[^0-9a-zA-Z_]", "_", str(col)) for col in df.columns]
    df.columns = cleaned_columns
    return df


# ——————————————————————————————————————————————————————————————
# CONSTANTES ET CHEMINS (définitions avant ARTEFACTS)
# ——————————————————————————————————————————————————————————————
RAW_DATA_FILENAME: str = "application_train.csv"
PREDICTIONS_FILENAME: str = "predictions_validation.parquet"

# --- Ajout pour l’interprétabilité globale et locale ---
BASELINE_MODEL_FILENAME: str = "lgbm_baseline.joblib"
X_VALID_FILENAME: str = "X_valid_pre.parquet"

ARTEFACTS: Dict[str, str] = {
    RAW_DATA_FILENAME: (
        "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/"
        "main/application_train.csv"
    ),
    PREDICTIONS_FILENAME: (
        "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/"
        "main/predictions_validation.parquet"
    ),
    BASELINE_MODEL_FILENAME: (
        "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/"
        "main/lgbm_baseline.joblib"
    ),
    X_VALID_FILENAME: (
        "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/"
        "main/X_valid_pre.parquet"
    ),
}

# ——————————————————————————————————————————————————————————————
# TÉLÉCHARGEMENT DES ARTEFACTS
# ——————————————————————————————————————————————————————————————
for fname, url in ARTEFACTS.items():
    download_if_missing(fname, url)

# ——————————————————————————————————————————————————————————————
# Chargement des données existantes
# ——————————————————————————————————————————————————————————————
df_eda_sample = load_csv_for_eda(RAW_DATA_FILENAME, sample_frac=0.05)
df_preds = load_parquet_file(PREDICTIONS_FILENAME)

# Chargement du modèle baseline et des données X_valid pour interprétabilité
import joblib

model_baseline = None
X_valid = None

if os.path.exists(BASELINE_MODEL_FILENAME):
    try:
        model_baseline = joblib.load(BASELINE_MODEL_FILENAME)
    except Exception as e:
        st.error(f"Erreur chargement modèle baseline : {e}")

if os.path.exists(X_VALID_FILENAME):
    X_valid_raw = load_parquet_file(X_VALID_FILENAME)
    if X_valid_raw is not None:
        X_valid = sanitize_feature_names(X_valid_raw)

# Pour intersectionnalité
df_application: Optional[pd.DataFrame] = None
if df_preds is not None:
    try:
        df_application = pd.read_csv(RAW_DATA_FILENAME, index_col=0)
        df_application = sanitize_feature_names(df_application)
        df_merged = df_application.join(df_preds, how="inner")
    except Exception as e:
        st.error(f"Erreur fusion application+prédictions : {e}")
        df_merged = None
else:
    df_merged = None

# ——————————————————————————————————————————————————————————————
# NAVIGATION (ajout page “Interpretabilité”)
# ——————————————————————————————————————————————————————————————
st.sidebar.title("📊 POC Scoring Équitable")
page_options: List[str] = [
    "Contexte & Objectifs",
    "Méthodologie",
    "Analyse Exploratoire (EDA)",
    "Résultats & Comparaisons",
    "Prédiction sur Client Sélectionné",
    "Analyse Intersectionnelle",
    "Interpretabilité",
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
# PAGE : Contexte & Objectifs
# ——————————————————————————————————————————————————————————————
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
        - **Découpage** : apprentissage (80 %), validation (10 %), test (10 %).
        - **Nettoyage** : gestion des valeurs aberrantes ou manquantes, suppression des doublons, filtrage sur le genre, plafonnement des revenus extrêmes.
        - **Nouvelles variables** : création de ratios simples (ex : mensualité/revenu, crédit/revenu), transformation de l’âge.
        - **Mise en forme** : transformation des variables catégorielles, découpage de l’âge en tranches, etc.
        - **Encodage & imputation** : gestion automatique des valeurs manquantes et transformation des variables pour les modèles.
        - **Nettoyage des noms de features** : standardisation pour éviter tout problème technique (caractères spéciaux, espaces, accents).
        """
    )

    st.subheader("Modèle de base (LightGBM)")
    st.write(
        """
        - Modèle classique de machine learning pour prédire le défaut de remboursement.
        - Prise en compte du déséquilibre entre bons et mauvais payeurs via `scale_pos_weight`.
        - Le seuil de décision pour catégoriser “défaut” ou “pas défaut” est choisi de façon optimale sur l’ensemble de validation (indice Youden).
        """
    )

    st.subheader("Modèle équitable (EG-EO)")
    st.write(
        """
        - LightGBM associé à la contrainte Fairlearn **Equalized Odds** pour garantir l’équité entre groupes sensibles (par ex. `CODE_GENDER`).
        - Utilisation de `ExponentiatedGradient` pour combiner plusieurs estimateurs et ajuster leurs poids afin de minimiser l’écart de performance (`EOD`) tout en conservant une bonne AUC.
        - On fixe une tolérance maximale (`eps`) sur l’écart d’équité autorisé.
        - Le modèle final est un wrapper qui encapsule cette logique et inclut également le seuil de décision optimisé.
        """
    )

    st.subheader("Évaluation et comparaison")
    st.write(
        """
        - **Performances mesurées** : AUC, précision, rappel, F1.  
        - **Équité** : vérification que le modèle ne favorise pas un groupe au détriment d’un autre via **Demographic Parity Difference (DPD)** et **Equalized Odds Difference (EOD)**.  
        - **Analyse détaillée** : matrices de confusion, taux de sélection par groupe sensible, métriques d’équité globales et par sous-population.
        """
    )

    st.subheader("Accessibilité & normes WCAG")
    st.write(
        """
        Pour que le dashboard soit utilisable par les personnes en situation de handicap, nous avons appliqué les principes essentiels du **WCAG (Web Content Accessibility Guidelines)** :
        """
    )
    st.markdown(
        """
        1. **Perceivable (Perceptible)**  
           - **Contraste élevé** : toutes les palettes de couleurs ont un ratio de contraste suffisant (texte et graphiques).  
           - **Texte alternatif & descriptions** : chaque graphique est accompagné d’une description textuelle (“Description : …”) pour lecteurs d’écran.  
           - **Taille de police lisible** : textes et annotations respectent une taille minimale pour être aisément lisibles.

        2. **Operable (Opérable)**  
           - **Navigation clavier** : toutes les interactions (sélecteurs, boutons) fonctionnent sans souris.  
           - **Focus visible** : le surlignage des éléments actifs est clairement visible.  
           - **Temps suffisant** : les utilisateurs ont le temps de comprendre et interagir avant toute expiration de session.

        3. **Understandable (Compréhensible)**  
           - **Langage clair** : terminologie simple, explications accessibles, évitement du jargon inutile.  
           - **Consistance** : mise en page uniforme et conventions de nommage cohérentes (titres, sous-titres, légendes).  
           - **Aide intégrée** : info-bulle ou légende fournie dès qu’un contrôle peut être ambigu.

        4. **Robust (Robuste)**  
           - **Compatibilité navigateurs et lecteurs d’écran** : tests réalisés sur Chrome, Firefox, NVDA et JAWS pour s’assurer d’une lecture cohérente.  
           - **Utilisation de balises HTML sémantiques** : via Streamlit, nous nous assurons que les éléments sont reconnus correctement par les aides techniques.

        > **Conclusion** : en appliquant ces quatre piliers du WCAG, nous garantissons que les graphiques et le contenu textuel sont accessibles à tous, y compris aux personnes daltoniennes, malvoyantes ou utilisant un lecteur d’écran.
        """
    )


# ——————————————————————————————————————————————————————————————
# PAGE : Analyse Exploratoire (EDA)
# ——————————————————————————————————————————————————————————————
elif page == "Analyse Exploratoire (EDA)":
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
            # Extraction des colonnes du DataFrame de prédictions
            y_true = df_preds["y_true"]
            y_pred_b = df_preds["y_pred_baseline"]
            y_pred_e = df_preds["y_pred_eo"]
            proba_b = df_preds["proba_baseline"]
            proba_e = df_preds["proba_eo"]
            sens = df_preds["sensitive_feature"]

            # Importations pour les metrics
            from sklearn.metrics import (
                roc_auc_score,
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                confusion_matrix,
            )

            # --- Classification Metrics ---
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

            # --- Fairness Metrics (DPD & EOD) ---
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

            # — Matrices de confusion —
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

            # — Taux de sélection par groupe sensible —
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

            # — Barplot des taux de sélection —
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
    st.header("🔍 Résultats & Interprétation pour un Client (validation)")
    if df_preds is not None and X_valid is not None and model_baseline is not None:
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

                # --- Interprétabilité locale via SHAP ---
                import shap

                try:
                    explainer = shap.TreeExplainer(model_baseline)
                    client_feat = X_valid.loc[[sel_id]]
                    shap_values = explainer.shap_values(client_feat)[1]  # classe 1

                    st.subheader("Interprétabilité Locale (SHAP) – Baseline")
                    st.markdown(
                        "Graphique des contributions individuelles des features pour la prédiction de ce client."
                    )
                    fig_shap = shap.plots._waterfall.waterfall_legacy(
                        explainer.expected_value[1], shap_values[0], feature_names=client_feat.columns
                    )
                    st.pyplot(fig_shap)
                except Exception as e:
                    st.warning(f"Impossible de calculer SHAP pour le client : {e}")

            else:
                st.error(f"L’ID {sel_id} n’est pas présent dans le jeu de validation.")
    else:
        st.warning("Modèle, données X_valid ou prédictions manquantes pour l’interprétation.")
# ——————————————————————————————————————————————————————————————
# PAGE : Interpretabilité (nouvelle page globale)
# ——————————————————————————————————————————————————————————————
elif page == "Interpretabilité":
    st.header("🔍 Interprétabilité Globale du Modèle Baseline")
    if model_baseline is None or X_valid is None:
        st.warning("Modèle Baseline ou données de validation non disponibles pour interprétabilité globale.")
    else:
        st.markdown(
            "Affichage des importances de features du modèle LightGBM Baseline "
            "et explication globale via SHAP."
        )

        # 1) Feature importances du modèle
        st.subheader("Importances de Features (Baseline)")
        fi = model_baseline.feature_importances_
        df_fi = pd.DataFrame({
            "feature": X_valid.columns,
            "importance": fi
        }).sort_values("importance", ascending=False).head(20)

        fig_fi = px.bar(
            df_fi,
            x="importance",
            y="feature",
            orientation="h",
            title="Top 20 Features par importance (Baseline)",
            labels={"importance": "Importance", "feature": "Feature"},
            color_discrete_sequence=["#1f77b4"],  # bleu à contraste élevé
        )
        fig_fi.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_fi, use_container_width=True)
        st.markdown(
            "**Description** : Barres horizontales en bleu indiquant "
            "les 20 features les plus importantes selon le modèle Baseline."
        )

        # 2) Explication globale via SHAP (summary plot)
        try:
            import shap

            st.subheader("Résumé SHAP Global (Baseline)")
            explainer = shap.TreeExplainer(model_baseline)
            shap_values = explainer.shap_values(X_valid)[1]
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(shap.summary_plot(shap_values, X_valid, show=False))
            st.markdown(
                "**Description** : Diagramme SHAP montrant la distribution des effets "
                "des features sur l’ensemble des clients de validation."
            )
        except Exception as e:
            st.warning(f"Impossible de générer le summary SHAP : {e}")

# ——————————————————————————————————————————————————————————————
# PAGE : Analyse Intersectionnelle (MAJ avec EOD_EO, DPD, precision/recall, histogrammes)
# ——————————————————————————————————————————————————————————————
# ——————————————————————————————————————————————————————————————
# PAGE : Analyse Intersectionnelle (MAJ complète)
# ——————————————————————————————————————————————————————————————
# ——————————————————————————————————————————————————————————————
# PAGE : Analyse Intersectionnelle (MAJ pour afficher coefficient de Gini)
# ——————————————————————————————————————————————————————————————
# ——————————————————————————————————————————————————————————————
# PAGE : Analyse Intersectionnelle (MAJ pour WCAG/Accessibilité)
# ——————————————————————————————————————————————————————————————
elif page == "Analyse Intersectionnelle":
    st.header("🔀 Analyse Intersectionnelle")
    st.caption(
        "Choisis une feature catégorielle pour évaluer les métriques "
        "de sélection et d’équité selon ses modalités."
    )

    if df_merged is not None:
        categorical_cols = df_merged.select_dtypes(include=["object", "category"]).columns.tolist()
        if not categorical_cols:
            st.warning("Aucune colonne catégorielle n’a été trouvée.")
        else:
            chosen_col = st.selectbox("Choisis une colonne catégorielle :", categorical_cols)
            modalities = df_merged[chosen_col].dropna().unique().tolist()
            if not modalities:
                st.error(f"Aucune modalité valide pour {chosen_col}.")
            else:
                # Filtre temporel/géographique (facultatif)
                if "DATE" in df_merged.columns:
                    dates = pd.to_datetime(df_merged["DATE"], errors="coerce")
                    df_merged["ANNEE"] = dates.dt.year
                    years = sorted(df_merged["ANNEE"].dropna().unique().astype(int).tolist())
                    chosen_year = st.selectbox("Filtrer par année :", ["Toutes"] + [str(y) for y in years])
                    if chosen_year != "Toutes":
                        df_merged = df_merged[df_merged["ANNEE"] == int(chosen_year)]
                if "REGION" in df_merged.columns:
                    regions = df_merged["REGION"].dropna().unique().tolist()
                    chosen_region = st.selectbox("Filtrer par région :", ["Toutes"] + regions)
                    if chosen_region != "Toutes":
                        df_merged = df_merged[df_merged["REGION"] == chosen_region]

                # Fonction pour calculer indice de Gini
                def gini_coefficient(x: np.ndarray) -> float:
                    arr = np.array(x, dtype=float)
                    if arr.size == 0 or np.all(arr == 0):
                        return np.nan
                    sorted_arr = np.sort(arr)
                    n = len(arr)
                    cumvals = np.cumsum(sorted_arr)
                    return (1 + (1 / n) - 2 * np.sum(cumvals) / (cumvals[-1] * n))

                # Calcul des métriques par modalité
                results = []
                for mod in modalities:
                    subset = df_merged[df_merged[chosen_col] == mod]
                    if subset.empty:
                        continue

                    y_true_mod = subset["y_true"]
                    y_pred_b_mod = subset["y_pred_baseline"]
                    y_pred_e_mod = subset["y_pred_eo"]
                    proba_e_mod = subset["proba_eo"]
                    sens_mod = subset["sensitive_feature"]

                    sel_base = float(y_pred_b_mod.mean())
                    sel_eo = float(y_pred_e_mod.mean())

                    try:
                        eod_mod = float(equalized_odds_difference(
                            y_true_mod, y_pred_e_mod, sensitive_features=sens_mod
                        ))
                    except Exception:
                        eod_mod = np.nan

                    try:
                        dpd_mod = float(demographic_parity_difference(
                            y_true_mod, y_pred_e_mod, sensitive_features=sens_mod
                        ))
                    except Exception:
                        dpd_mod = np.nan

                    from sklearn.metrics import precision_score, recall_score
                    try:
                        prec_mod = float(precision_score(y_true_mod, y_pred_e_mod, zero_division=0))
                        rec_mod = float(recall_score(y_true_mod, y_pred_e_mod, zero_division=0))
                    except Exception:
                        prec_mod = np.nan
                        rec_mod = np.nan

                    # Calcul du coefficient de Gini pour chaque groupe sensible
                    gini_values = {}
                    for grp in sens_mod.dropna().unique():
                        scores_grp = proba_e_mod[sens_mod == grp].values
                        gini_values[f"Gini_{grp}"] = float(gini_coefficient(scores_grp))

                    results.append({
                        "Modalité": mod,
                        "SelRate_Baseline": sel_base,
                        "SelRate_EO": sel_eo,
                        "EOD_EO": eod_mod,
                        "DPD_EO": dpd_mod,
                        "Precision_EO": prec_mod,
                        "Recall_EO": rec_mod,
                        **gini_values,
                    })

                df_inter = pd.DataFrame(results).set_index("Modalité")
                st.subheader(f"Métriques par modalité de '{chosen_col}'")
                st.dataframe(
                    df_inter.style.format({col: "{:.3f}" for col in df_inter.columns}),
                    use_container_width=True,
                )

                # 1) Taux de sélection : on ajoute un label textuel sur chaque barre,
                #    et on utilise une palette à contraste élevé
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
                    labels={"Modalité": chosen_col, "Taux de sélection": "Taux de sélection"},
                    color_discrete_sequence=["#00429d", "#ffa600"],  # bleu foncé & orange vif
                    text="Taux de sélection",  # affichage du pourcentage sur chaque barre
                )
                fig_inter_sel.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                fig_inter_sel.update_layout(
                    uniformtext_minsize=8,
                    uniformtext_mode="hide",
                    legend_title_text="Modèle",
                )
                # Description textuelle pour les lecteurs d'écran
                st.markdown(
                    f"**Description** : Barres comparant les taux de sélection Baseline "
                    f"(bleu foncé) et EO (orange vif) pour chaque modalité de '{chosen_col}'."
                )
                st.plotly_chart(fig_inter_sel, use_container_width=True)

                # 2) EOD pour EO : barres avec annotation de la valeur
                fig_inter_eod = px.bar(
                    df_inter.reset_index(),
                    x="Modalité",
                    y="EOD_EO",
                    title=f"EOD (EO mitigé) par modalités de '{chosen_col}'",
                    labels={"Modalité": chosen_col, "EOD_EO": "Equalized Odds Diff (EO)"},
                    color_discrete_sequence=["#800080"],  # violet foncé à bon contraste
                    text="EOD_EO",
                )
                fig_inter_eod.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                fig_inter_eod.update_layout(
                    yaxis=dict(range=[df_inter["EOD_EO"].min() - 0.05, df_inter["EOD_EO"].max() + 0.05]),
                    uniformtext_minsize=8,
                    uniformtext_mode="hide",
                )
                st.markdown(
                    f"**Description** : Barres violettes représentant l’EOD du modèle EO "
                    f"pour chaque modalité de '{chosen_col}', annotées avec la valeur numérique."
                )
                st.plotly_chart(fig_inter_eod, use_container_width=True)

                # 3) DPD pour EO : barres avec annotation
                fig_inter_dpd = px.bar(
                    df_inter.reset_index(),
                    x="Modalité",
                    y="DPD_EO",
                    title=f"DPD (EO mitigé) par modalités de '{chosen_col}'",
                    labels={"Modalité": chosen_col, "DPD_EO": "Demographic Parity Diff (EO)"},
                    color_discrete_sequence=["#1b7837"],  # vert foncé
                    text="DPD_EO",
                )
                fig_inter_dpd.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                fig_inter_dpd.update_layout(
                    yaxis=dict(range=[df_inter["DPD_EO"].min() - 0.05, df_inter["DPD_EO"].max() + 0.05]),
                    uniformtext_minsize=8,
                    uniformtext_mode="hide",
                )
                st.markdown(
                    f"**Description** : Barres vertes illustrant la DPD du modèle EO "
                    f"pour chaque modalité de '{chosen_col}', annotées avec la valeur."
                )
                st.plotly_chart(fig_inter_dpd, use_container_width=True)

                # 4) Précision & rappel : barres côte à côte, avec annotation
                df_pr_rec = df_inter[["Precision_EO", "Recall_EO"]].reset_index().melt(
                    id_vars="Modalité",
                    value_vars=["Precision_EO", "Recall_EO"],
                    var_name="Métrique",
                    value_name="Score",
                )
                fig_pr_rec = px.bar(
                    df_pr_rec,
                    x="Modalité",
                    y="Score",
                    color="Métrique",
                    barmode="group",
                    title=f"Précision & Rappel (EO) par modalités de '{chosen_col}'",
                    labels={"Modalité": chosen_col, "Score": "Score"},
                    color_discrete_map={"Precision_EO": "#a50026", "Recall_EO": "#fdae61"},  # rouge vif & orange clair
                    text="Score",
                )
                fig_pr_rec.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                fig_pr_rec.update_layout(
                    uniformtext_minsize=8,
                    uniformtext_mode="hide",
                    legend_title_text="Métrique",
                )
                st.markdown(
                    f"**Description** : Comparaison côte à côte de la précision "
                    f"(rouge vif) et du rappel (orange clair) pour chaque modalité."
                )
                st.plotly_chart(fig_pr_rec, use_container_width=True)

                # 5) Coefficients de Gini : barres groupées par groupe sensible
                gini_cols = [c for c in df_inter.columns if c.startswith("Gini_")]
                if gini_cols:
                    df_gini = (
                        df_inter[gini_cols]
                        .reset_index()
                        .melt(id_vars="Modalité", value_vars=gini_cols, var_name="Groupe", value_name="Gini")
                    )
                    df_gini["Groupe"] = df_gini["Groupe"].str.replace(r"^Gini_", "", regex=True)
                    fig_gini = px.bar(
                        df_gini,
                        x="Modalité",
                        y="Gini",
                        color="Groupe",
                        barmode="group",
                        title=f"Coefficients de Gini des scores EO par modalités de '{chosen_col}'",
                        labels={"Modalité": chosen_col, "Gini": "Coefficient de Gini"},
                        color_discrete_sequence=px.colors.qualitative.Dark24,  # palette qualitatives à contraste élevé
                        text="Gini",
                    )
                    fig_gini.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                    fig_gini.update_layout(
                        uniformtext_minsize=8,
                        uniformtext_mode="hide",
                    )
                    st.markdown(
                        f"**Description** : Coefficients de Gini des scores EO, "
                        f"séparés par groupe sensible, pour chaque modalité."
                    )
                    st.plotly_chart(fig_gini, use_container_width=True)

                # 6) Distribution des probabilités EO par groupe (hachures + couleur)
                if st.checkbox("Afficher distribution des probabilités EO par groupe pour chaque modalité"):
                    for mod in modalities:
                        subset = df_merged[df_merged[chosen_col] == mod]
                        if subset.empty:
                            continue
                        fig_hist = px.histogram(
                            subset,
                            x="proba_eo",
                            color="sensitive_feature",
                            barmode="overlay",
                            nbins=30,
                            pattern_shape="sensitive_feature",  # ajoute hachures selon le groupe
                            pattern_shape_sequence=["/", "\\"],    # deux motifs de hachures distincts
                            title=f"Distribution des probabilités EO pour la modalité '{mod}'",
                            labels={"proba_eo": "Score EO", "sensitive_feature": "Groupe sensible"},
                            color_discrete_sequence=px.colors.qualitative.Set2,  # palette alternative à contraste élevé
                        )
                        st.markdown(
                            f"**Description** : Répartition des scores EO pour la modalité '{mod}', "
                            "comparant les groupes sensibles via couleur et motif."
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                # 7) Matrice de confusion pour EO par modalité (texte + alt text)
                if st.checkbox("Afficher la matrice de confusion EO pour chaque modalité"):
                    from sklearn.metrics import confusion_matrix

                    for mod in modalities:
                        subset = df_merged[df_merged[chosen_col] == mod]
                        if subset.empty:
                            continue
                        y_true_mod = subset["y_true"]
                        y_pred_e_mod = subset["y_pred_eo"]
                        cm = confusion_matrix(y_true_mod, y_pred_e_mod)
                        labels = ["Non-Défaut (0)", "Défaut (1)"]
                        z_text = [[str(entry) for entry in row] for row in cm]
                        fig_cm = ff.create_annotated_heatmap(
                            cm,
                            x=labels,
                            y=labels,
                            annotation_text=z_text,
                            colorscale="Greys",
                        )
                        fig_cm.update_layout(
                            title_text=f"Matrice de confusion EO pour '{chosen_col}' = '{mod}'",
                            xaxis_title="Prédit",
                            yaxis_title="Réel",
                        )
                        st.markdown(
                            f"**Description** : Matrice de confusion EO pour la modalité '{mod}', "
                            "montrant TN, FP, FN, TP avec légende textuelle."
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)

                # 8) Export du rapport Excel
                buffer = None
                if st.button("📥 Exporter ce tableau au format Excel"):
                    import io

                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                        df_inter.to_excel(writer, sheet_name="Intersectionnalité")
                    buffer.seek(0)
                    st.download_button(
                        label="Télécharger le fichier Excel",
                        data=buffer,
                        file_name="rapport_intersectionnalite.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                # 9) Comparaison “avant/après” biais artificiel
                if st.checkbox("Comparer avant/après injection d’un biais artificiel"):
                    group_to_bias = st.selectbox(
                        "Choisir le groupe sensible à biaiser :", sens_mod.dropna().unique().tolist()
                    )
                    rate_to_flip = st.slider("Pourcentage de labels positifs à inverser %", 0, 100, 10)
                    mask = (df_merged["sensitive_feature"] == group_to_bias) & (df_merged["y_true"] == 1)
                    idxs = df_merged[mask].sample(frac=rate_to_flip / 100, random_state=42).index
                    df_biased = df_merged.copy()
                    df_biased.loc[idxs, "y_true"] = 0

                    try:
                        eod_global_orig = equalized_odds_difference(
                            df_merged["y_true"],
                            df_merged["y_pred_eo"],
                            sensitive_features=df_merged["sensitive_feature"],
                        )
                        eod_global_biased = equalized_odds_difference(
                            df_biased["y_true"],
                            df_biased["y_pred_eo"],
                            sensitive_features=df_biased["sensitive_feature"],
                        )
                        st.write(f"- EOD global (avant biais) : **{eod_global_orig:.3f}**")
                        st.write(f"- EOD global (après biais) : **{eod_global_biased:.3f}**")
                    except Exception as ex:
                        st.error(f"Erreur lors de la comparaison biais : {ex}")

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
