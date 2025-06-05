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
    demographic_parity_difference,
    equalized_odds_difference,
)

# ——————————————————————————————————————————————————————————————
# FONCTIONS UTILITAIRES
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
# CONSTANTES ET CHEMINS
# ——————————————————————————————————————————————————————————————
RAW_DATA_FILENAME: str = "application_train.csv"
PREDICTIONS_FILENAME: str = "predictions_test.parquet"

ARTEFACTS: Dict[str, str] = {
    RAW_DATA_FILENAME: (
        "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/"
        "main/application_train.csv"
    ),
    PREDICTIONS_FILENAME: (
        "https://huggingface.co/cantalapiedra/poc_scoring_fair/resolve/"
        "main/predictions_test.parquet"
    ),
}

# ——————————————————————————————————————————————————————————————
# TÉLÉCHARGEMENT DES ARTEFACTS
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
        df_merged = df_application.join(df_preds, how="inner")
    except Exception as e:
        st.error(f"Erreur lors de la fusion application+prédictions : {e}")
        df_merged = None
else:
    df_merged = None

# ——————————————————————————————————————————————————————————————
# NAVIGATION
# ——————————————————————————————————————————————————————————————
st.sidebar.title("📊 POC Scoring Équitable")
page_options: List[str] = [
    "Contexte & Objectifs",
    "Méthodologie",
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
    st.subheader("Fairlearn")
    st.markdown(
        """
        **Fairlearn** est une librairie open source développée par Microsoft pour évaluer et améliorer l’équité des modèles de Machine Learning.  
        Elle fournit :
        - Des métriques d’équité (DPD, EOD, etc.) pour détecter d’éventuels biais.  
        - Un module de réduction (« reductions ») qui ajuste plusieurs estimateurs afin de satisfaire des contraintes d’équité, comme **Equalized Odds** ou **Demographic Parity**.  
        - Des outils de visualisation pour comparer différentes stratégies de mitigation.  

        Pour en savoir plus, consultez la documentation officielle de Fairlearn :  
        [https://fairlearn.org](https://fairlearn.org)  
        (ou le dépôt GitHub)  
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



# ——————————————————————————————————————————————————————————————
# PAGE : Méthodologie
# ——————————————————————————————————————————————————————————————
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
           - **Contraste élevé** : palettes de couleurs à ratio de contraste suffisant (texte et graphiques).  
           - **Texte alternatif & descriptions** : chaque graphique a une description textuelle (“Description : …”) pour lecteurs d’écran.  
           - **Taille de police lisible** : textes et annotations respectent une taille minimale.

        2. **Operable (Opérable)**  
           - **Navigation clavier** : toutes les interactions (sélecteurs, boutons) fonctionnent sans souris.  
           - **Focus visible** : le surlignage des éléments actifs est clairement visible.  
           - **Temps suffisant** : l’utilisateur dispose de temps pour comprendre et interagir avant toute expiration de session.

        3. **Understandable (Compréhensible)**  
           - **Langage clair** : terminologie simple, explications accessibles, évitement du jargon inutile.  
           - **Consistance** : mise en page uniforme et conventions de nommage cohérentes (titres, sous-titres, légendes).  
           - **Aide intégrée** : info-bulle ou légende fournie dès qu’un contrôle peut être ambigu.

        4. **Robust (Robuste)**  
           - **Compatibilité navigateurs et lecteurs d’écran** : tests effectués sur Chrome, Firefox, NVDA et JAWS.  
           - **Balises HTML sémantiques** : via Streamlit, on s’assure que les éléments sont correctement reconnus par les aides techniques.

        > En appliquant ces quatre piliers WCAG, nous garantissons que le contenu textuel et graphique reste accessible, y compris aux personnes daltoniennes, malvoyantes ou utilisant un lecteur d’écran.
        """
    )


# ——————————————————————————————————————————————————————————————
# PAGE : Analyse Exploratoire (EDA) – Distribution selon CODE_GENDER
# ——————————————————————————————————————————————————————————————
elif page == "Analyse Exploratoire (EDA)":
    st.header("🔎 Analyse Exploratoire des Données (EDA)")
    st.caption(
        f"Basée sur un échantillon de "
        f"{len(df_eda_sample) if df_eda_sample is not None else 0} lignes."
    )

    if df_eda_sample is not None and not df_eda_sample.empty:
        # … (vos blocs précédents) …

        # ── Répartition du genre (CODE_GENDER) ──
        if "CODE_GENDER" in df_eda_sample.columns:
            st.subheader("Répartition par genre ('CODE_GENDER')")
            gender_counts = df_eda_sample["CODE_GENDER"].value_counts()
            gender_percent = df_eda_sample["CODE_GENDER"].value_counts(normalize=True) * 100

            col1, col2 = st.columns(2)
            with col1:
                st.write("Comptage absolu :")
                st.dataframe(gender_counts)
            with col2:
                st.write("Pourcentage :")
                st.dataframe(gender_percent.map("{:.2f}%".format))

            try:
                fig_gender_pie = px.pie(
                    names=gender_counts.index,
                    values=gender_counts.values,
                    title="Répartition du genre dans l’échantillon",
                    color_discrete_sequence=["#0A0A23", "#FF6600", "#2CA02C"],
                )
                fig_gender_pie.update_traces(textinfo="label+percent", hole=0.4)
                st.plotly_chart(fig_gender_pie, use_container_width=True)
            except Exception as e:
                st.warning(f"Impossible de générer le camembert de genre : {e}")
        else:
            st.info("La colonne 'CODE_GENDER' n’est pas présente pour afficher la répartition par genre.")

        # ── Distribution de 'TARGET' ──
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
                    labels={"TARGET": "Classe de défaut (0 : Non-défaut, 1 : Défaut)"},
                    text_auto=True,
                    color_discrete_sequence=["#1F77B4", "#FF7F0E"],
                )
                fig_target_hist.update_layout(bargap=0.2)
                st.plotly_chart(fig_target_hist, use_container_width=True)
            except Exception as e:
                st.warning(f"Impossible de générer l'histogramme de TARGET : {e}")
        else:
            st.warning("La colonne 'TARGET' n’est pas présente dans l’échantillon.")

        # ── Distribution de 'AMT_INCOME_TOTAL' ──
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
                    title=f"Distribution de '{numerical_col}' (plafonné à {cap:,.0f} si applicable)",
                    labels={numerical_col: "Revenu total", "TARGET": "Classe de défaut"},
                    color_discrete_sequence=["#1F77B4", "#FF7F0E"],
                )
                st.plotly_chart(fig_income, use_container_width=True)
            except Exception as e:
                st.warning(f"Impossible de générer l'histogramme de {numerical_col} : {e}")
        else:
            st.info(f"La colonne '{numerical_col}' n’est pas disponible pour l’EDA.")

        # ── Menu déroulant : distribution d’une feature selon CODE_GENDER ──
        if "CODE_GENDER" in df_eda_sample.columns:
            st.subheader("Distribution d’une feature selon CODE_GENDER")
            potential_cols = df_eda_sample.select_dtypes(include=[np.number]).columns.tolist()
            # On retire TARGET et AMT_INCOME_TOTAL
            potential_cols = [c for c in potential_cols if c not in ["TARGET", "AMT_INCOME_TOTAL"]]

            chosen_feature = st.selectbox(
                "Choisissez une colonne numérique :", [""] + potential_cols
            )
            if chosen_feature:
                st.markdown(f"**Distribution de '{chosen_feature}' par genre**")
                try:
                    fig_feat_gender = px.histogram(
                        df_eda_sample,
                        x=chosen_feature,
                        color="CODE_GENDER",
                        nbins=30,
                        barmode="group",  # barres côte à côte
                        opacity=0.8,
                        title=f"Distribution de '{chosen_feature}' par CODE_GENDER",
                        labels={chosen_feature: chosen_feature, "CODE_GENDER": "Genre"},
                        color_discrete_map={"M": "#0A0A23", "F": "#FF6600", "XNA": "#2CA02C"},
                    )
                    fig_feat_gender.update_layout(bargap=0.1)
                    # Ajout d'une ligne verticale pour la moyenne globale
                    moyenne_globale = df_eda_sample[chosen_feature].mean()
                    fig_feat_gender.add_vline(
                        x=moyenne_globale,
                        line_color="black",
                        line_dash="dash",
                        annotation_text="Moyenne globale",
                        annotation_position="top right"
                    )
                    st.plotly_chart(fig_feat_gender, use_container_width=True)
                except Exception as e:
                    st.warning(f"Impossible de générer la distribution pour '{chosen_feature}' : {e}")
        # ── Fin EDA ──

    else:
        st.error("L’échantillon pour l’EDA n’a pas pu être chargé.")

# ——————————————————————————————————————————————————————————————
# PAGE : Résultats & Comparaisons
# ——————————————————————————————————————————————————————————————
elif page == "Résultats & Comparaisons":
    st.header("📊 Résultats comparatifs (jeu de validation)")
    if df_preds is not None:
        try:
            # **Extraction des colonnes du DataFrame de prédictions**
            y_true   = df_preds["y_true"]
            y_pred_b = df_preds["y_pred_baseline"]  # 0 = accord, 1 = refus
            y_pred_e = df_preds["y_pred_eo"]        # 0 = accord, 1 = refus
            proba_b  = df_preds["proba_baseline"]
            proba_e  = df_preds["proba_eo"]
            sens     = df_preds["sensitive_feature"]

            # --- Classification Metrics + taux de refus / d’acceptation global ---
            from sklearn.metrics import (
                roc_auc_score,
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
            )

            metrics_b = {
                "AUC":                         roc_auc_score(y_true, proba_b),
                "Accuracy":                    accuracy_score(y_true, y_pred_b),
                "Precision (1)":               precision_score(y_true, y_pred_b, pos_label=1, zero_division=0),
                "Recall (1)":                  recall_score(y_true, y_pred_b, pos_label=1, zero_division=0),
                "F1 (1)":                      f1_score(y_true, y_pred_b, pos_label=1, zero_division=0),
                "Taux de refus global":        float(np.mean(y_pred_b)),        # selection_rate
                "Taux d’acceptation global":   float(1.0 - np.mean(y_pred_b)),
            }
            metrics_e = {
                "AUC":                         roc_auc_score(y_true, proba_e),
                "Accuracy":                    accuracy_score(y_true, y_pred_e),
                "Precision (1)":               precision_score(y_true, y_pred_e, pos_label=1, zero_division=0),
                "Recall (1)":                  recall_score(y_true, y_pred_e, pos_label=1, zero_division=0),
                "F1 (1)":                      f1_score(y_true, y_pred_e, pos_label=1, zero_division=0),
                "Taux de refus global":        float(np.mean(y_pred_e)),
                "Taux d’acceptation global":   float(1.0 - np.mean(y_pred_e)),
            }

            df_metrics = pd.DataFrame(
                [
                    {"Modèle": "Baseline", **metrics_b},
                    {"Modèle": "EO Wrapper", **metrics_e},
                ]
            ).set_index("Modèle")

            st.subheader("Métriques de classification")
            st.dataframe(df_metrics.style.format("{:.3f}", na_rep="-"), use_container_width=True)

            # --- Fairness Metrics + taux de refus / d’acceptation par groupe sensible ---
            mf_b = MetricFrame(
                metrics={"refusal_rate": fairlearn_selection_rate},  # 1 = refus
                y_true=y_true,
                y_pred=y_pred_b,
                sensitive_features=sens,
            )
            mf_e = MetricFrame(
                metrics={"refusal_rate": fairlearn_selection_rate},
                y_true=y_true,
                y_pred=y_pred_e,
                sensitive_features=sens,
            )

            df_sel = pd.DataFrame({
                "Groupe sensible":                  mf_b.by_group.index,
                "Taux de refus Baseline":            mf_b.by_group["refusal_rate"].values,
                "Taux d’acceptation Baseline":       1.0 - mf_b.by_group["refusal_rate"].values,
                "Taux de refus EO Wrapper":          mf_e.by_group["refusal_rate"].values,
                "Taux d’acceptation EO Wrapper":     1.0 - mf_e.by_group["refusal_rate"].values,
            }).set_index("Groupe sensible")

            st.subheader("Taux de refus / d’acceptation par groupe sensible")
            st.dataframe(df_sel.style.format("{:.3f}", na_rep="-"), use_container_width=True)

            # — Barplot du taux d’acceptation —
            df_sel_plot = df_sel.reset_index().melt(
                id_vars="Groupe sensible",
                value_vars=["Taux d’acceptation Baseline", "Taux d’acceptation EO Wrapper"],
                var_name="Modèle",
                value_name="Taux d’acceptation",
            )
            fig_sel = px.bar(
                df_sel_plot,
                x="Groupe sensible",
                y="Taux d’acceptation",
                color="Modèle",
                barmode="group",
                title="Taux d’acceptation par groupe sensible et par modèle",
                labels={"Groupe sensible": "Groupe sensible", "Taux d’acceptation": "Taux d’acceptation"},
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
        "Choisis une feature (catégorielle ou numérique) pour évaluer "
        "les métriques de sélection et d’équité selon ses modalités / bins."
    )

    if df_merged is not None:
        # 1. Lister les colonnes catégorielles et numériques
        categorical_cols = df_merged.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols     = df_merged.select_dtypes(include=[np.number]).columns.tolist()

        # Retirer les colonnes de labels/prédictions pour ne pas les proposer
        for excl in ["y_true", "y_pred_baseline", "y_pred_eo", "proba_baseline", "proba_eo", "sensitive_feature"]:
            if excl in categorical_cols:
                categorical_cols.remove(excl)
            if excl in numeric_cols:
                numeric_cols.remove(excl)

        # 2. Choix du type de feature
        chosen_type = st.radio("Type de feature à intersectionnaliser :", ["Catégorielle", "Numérique"])
        if chosen_type == "Catégorielle":
            chosen_col = st.selectbox("Choisis une colonne catégorielle :", [""] + categorical_cols)
            if not chosen_col:
                st.info("Sélectionne une colonne pour commencer.")
                st.stop()

            modalities = df_merged[chosen_col].dropna().unique().tolist()
            df_merged["INTERSECTION"] = df_merged[chosen_col].astype(str)
            bin_labels = modalities

        else:
            chosen_col = st.selectbox("Choisis une colonne numérique :", [""] + numeric_cols)
            if not chosen_col:
                st.info("Sélectionne une colonne pour commencer.")
                st.stop()

            # 2.1. Nombre de bins souhaités
            n_bins = st.slider(
                "Nombre de bins à créer pour cette variable :", min_value=2, max_value=10, value=5
            )
            # 2.2. Afficher min / max
            col_min = float(df_merged[chosen_col].min())
            col_max = float(df_merged[chosen_col].max())
            st.write(f"Valeurs comprises entre {col_min:.2f} et {col_max:.2f}")

            # 2.3. Créer les bins via pd.cut
            try:
                # On crée d’abord les bins, en récupérant leurs bornes
                series_cut, bins = pd.cut(
                    df_merged[chosen_col],
                    bins=n_bins,
                    retbins=True,
                    labels=[f"{round(bins[i],2)}–{round(bins[i+1],2)}" for i in range(n_bins)],
                    include_lowest=True,
                )
                df_merged["INTERSECTION"] = series_cut
                bin_labels = df_merged["INTERSECTION"].dropna().unique().tolist()
            except Exception as e:
                st.error(f"Erreur lors de la création des bins : {e}")
                st.stop()

            st.write(f"Bins calculés ({n_bins}) pour '{chosen_col}':")
            st.write(bins)

        # 3. (Optionnel) Filtrage temporal/géographique
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

        # 4. Fonction utilitaire pour Gini
        def gini_coefficient(x: np.ndarray) -> float:
            """Calcule le coefficient de Gini (x doit être ≥ 0)."""
            arr = np.array(x, dtype=float)
            if arr.size == 0 or np.all(arr == 0):
                return np.nan
            sorted_arr = np.sort(arr)
            n = len(arr)
            cumvals = np.cumsum(sorted_arr)
            return (1 + (1 / n) - 2 * np.sum(cumvals) / (cumvals[-1] * n))

        # 5. Boucle par bin/ modalité
        results = []
        for modal in bin_labels:
            subset = df_merged[df_merged["INTERSECTION"] == modal]
            if subset.empty:
                continue

            y_true_mod   = subset["y_true"]
            y_pred_b_mod = subset["y_pred_baseline"]
            y_pred_e_mod = subset["y_pred_eo"]
            proba_e_mod  = subset["proba_eo"]
            sens_mod     = subset["sensitive_feature"]

            # Taux de refus / d’acceptation
            ref_base = float(np.mean(y_pred_b_mod))
            acc_base = float(1.0 - ref_base)
            ref_eo   = float(np.mean(y_pred_e_mod))
            acc_eo   = float(1.0 - ref_eo)

            # EOD / DPD pour EO
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

            # Précision & recall pour EO
            from sklearn.metrics import precision_score, recall_score
            try:
                prec_mod = float(precision_score(y_true_mod, y_pred_e_mod, zero_division=0))
                rec_mod  = float(recall_score(y_true_mod, y_pred_e_mod, zero_division=0))
            except Exception:
                prec_mod = np.nan
                rec_mod  = np.nan

            # Gini des scores EO par groupe
            gini_values = {}
            for grp in sens_mod.dropna().unique():
                scores_grp = proba_e_mod[sens_mod == grp].values
                gini_values[f"Gini_{grp}"] = float(gini_coefficient(scores_grp))

            results.append({
                "Modalité/Bin":                     modal,
                "Support":                          len(subset),
                "Taux de refus Baseline":            ref_base,
                "Taux d’acceptation Baseline":       acc_base,
                "Taux de refus EO":                  ref_eo,
                "Taux d’acceptation EO":             acc_eo,
                "EOD EO":                           eod_mod,
                "DPD EO":                           dpd_mod,
                "Precision EO":                     prec_mod,
                "Recall EO":                        rec_mod,
                **gini_values
            })

        # 6. DataFrame de synthèse
        df_inter = pd.DataFrame(results).set_index("Modalité/Bin")
        st.subheader(f"Métriques par modalité/bin de '{chosen_col}'")
        st.dataframe(df_inter.style.format({col: "{:.3f}" for col in df_inter.columns}), use_container_width=True)

        # 7. Barplot du taux d’acceptation
        df_sel_plot = df_inter.reset_index().melt(
            id_vars=["Modalité/Bin"],
            value_vars=["Taux d’acceptation Baseline", "Taux d’acceptation EO"],
            var_name="Modèle",
            value_name="Taux d’acceptation",
        )
        fig_sel = px.bar(
            df_sel_plot,
            x="Modalité/Bin",
            y="Taux d’acceptation",
            color="Modèle",
            barmode="group",
            title=f"Taux d’acceptation par modalité/bin de '{chosen_col}'",
            labels={"Modalité/Bin": chosen_col},
        )
        st.plotly_chart(fig_sel, use_container_width=True)

        # 8. Barplot EOD
        fig_eod = px.bar(
            df_inter.reset_index(),
            x="Modalité/Bin",
            y="EOD EO",
            title=f"EOD (EO mitigé) par modalité/bin de '{chosen_col}'",
            labels={"EOD EO": "Equalized Odds Diff (EO)"},
        )
        st.plotly_chart(fig_eod, use_container_width=True)

        # 9. Barplot DPD
        fig_dpd = px.bar(
            df_inter.reset_index(),
            x="Modalité/Bin",
            y="DPD EO",
            title=f"DPD (EO mitigé) par modalité/bin de '{chosen_col}'",
            labels={"DPD EO": "Demographic Parity Diff (EO)"},
        )
        st.plotly_chart(fig_dpd, use_container_width=True)

        # 10. Barplot Precision & Recall pour EO
        df_prrec = df_inter[["Precision EO", "Recall EO"]].reset_index().melt(
            id_vars=["Modalité/Bin"],
            value_vars=["Precision EO", "Recall EO"],
            var_name="Métrique",
            value_name="Score",
        )
        fig_prrec = px.bar(
            df_prrec,
            x="Modalité/Bin",
            y="Score",
            color="Métrique",
            barmode="group",
            title=f"Precision & Recall (EO) par modalité/bin de '{chosen_col}'",
            labels={"Modalité/Bin": chosen_col, "Score": "Valeur"},
        )
        st.plotly_chart(fig_prrec, use_container_width=True)

        # 11. Distribution des probabilités EO par groupe (optionnel)
        if st.checkbox("Afficher distribution des probabilités EO par groupe pour chaque modalité/bin"):
            for modal in df_inter.index:
                subset = df_merged[df_merged["INTERSECTION"] == modal]
                if subset.empty:
                    continue
                fig_hist = px.histogram(
                    subset,
                    x="proba_eo",
                    color="sensitive_feature",
                    nbins=30,
                    barmode="overlay",
                    title=f"Distribution des scores EO pour '{chosen_col}' = '{modal}'",
                    labels={"proba_eo": "Score EO", "sensitive_feature": "Groupe sensible"},
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        # 12. Matrice de confusion EO par modalité/bin (optionnel)
        if st.checkbox("Afficher la matrice de confusion EO pour chaque modalité/bin"):
            from sklearn.metrics import confusion_matrix
            for modal in df_inter.index:
                subset = df_merged[df_merged["INTERSECTION"] == modal]
                if subset.empty:
                    continue
                y_true_mod   = subset["y_true"]
                y_pred_e_mod = subset["y_pred_eo"]
                cm = confusion_matrix(y_true_mod, y_pred_e_mod)
                labels_cm = ["Non-Défaut (0)", "Défaut (1)"]
                z_text = [[str(val) for val in row] for row in cm]
                fig_cm = ff.create_annotated_heatmap(
                    cm, x=labels_cm, y=labels_cm, annotation_text=z_text, colorscale="Purples"
                )
                fig_cm.update_layout(
                    title_text=f"Matrice de confusion EO pour '{chosen_col}' = '{modal}'",
                    xaxis_title="Prédit",
                    yaxis_title="Réel",
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        # 13. Export Excel
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

        # 14. Injection de biais artificiel (inchangée)
        # … (le code existant sans modification) …

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
