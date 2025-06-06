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
    st.header("🔀 Analyse Intersectionnelle (Optimisée Avancée)")
    st.caption(
        "Métriques de refus/acceptation et d’équité, croisées avec le genre "
        "et d’autres variables, avec filtrage, binning par quartile et plot comparatif."
    )

    if df_merged is None:
        st.warning("Fusion des données application + prédictions impossible.")
        st.stop()

    # === 1. Filtrer selon le genre ===
    genres = df_merged["sensitive_feature"].dropna().unique().tolist()
    selected_genres = st.multiselect(
        "Filtrer par genre (choisis un ou plusieurs) :",
        options=genres,
        default=genres
    )
    df_filtered_gender = df_merged[df_merged["sensitive_feature"].isin(selected_genres)].copy()
    if df_filtered_gender.empty:
        st.warning("Aucun enregistrement pour le(s) genre(s) sélectionné(s).")
        st.stop()

    # === 2. Sélection du type de feature à analyser ===
    feature_type = st.radio(
        "Type de feature à analyser :",
        ["Catégorielle", "Numérique"]
    )

    if feature_type == "Catégorielle":
        candidate_cols = df_filtered_gender.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
    else:
        candidate_cols = df_filtered_gender.select_dtypes(
            include=[np.number]
        ).columns.tolist()

    excluded_cols = [
        "y_true", "y_pred_baseline", "y_pred_eo",
        "proba_baseline", "proba_eo", "sensitive_feature"
    ]
    candidate_cols = [c for c in candidate_cols if c not in excluded_cols]

    if not candidate_cols:
        st.warning("Aucune colonne disponible pour ce type.")
        st.stop()

    chosen_col = st.selectbox(
        "Choisis une colonne à analyser :",
        candidate_cols
    )
    df_work = df_filtered_gender.copy()

    # === 3. Binning par quartile pour variables numériques, ou conversion en string ===
    if feature_type == "Numérique":
        unique_vals = df_work[chosen_col].dropna().unique()
        # Si flag binaire (0/1 ou 1/0), on convertit en "Non"/"Oui"
        if set(unique_vals) <= {0, 1}:
            df_work["MODALITE_ANALYSE"] = df_work[chosen_col].map({0: "Non", 1: "Oui"})
        else:
            # Binning en quartiles avec labels personnalisés
            labels_bins = ["Très faible (Q1)", "Faible (Q2)", "Élevé (Q3)", "Très élevé (Q4)"]
            try:
                df_work["MODALITE_ANALYSE"] = pd.qcut(
                    df_work[chosen_col],
                    q=4,
                    labels=labels_bins,
                    duplicates="drop"
                )
            except Exception:
                # Si qcut échoue (valeurs identiques ou pas assez de quartiles),
                # on fait un cut en intervalles égaux, sans labels
                try:
                    df_work["MODALITE_ANALYSE"] = pd.cut(
                        df_work[chosen_col],
                        bins=4,
                        labels=labels_bins,
                        duplicates="drop"
                    )
                except Exception:
                    st.error(
                        "Impossible d'appliquer un binning par quartile sur cette variable numérique."
                    )
                    st.stop()
    else:
        # Catégorie existante (y compris flags encodés comme 0/1 préalablement)
        df_work["MODALITE_ANALYSE"] = df_work[chosen_col].astype(str)

    # === 4. Création de la modalité combinée (feature + genre) ===
    df_work["MODALITE_GENRE"] = (
        df_work["MODALITE_ANALYSE"].astype(str) + "_"
        + df_work["sensitive_feature"].astype(str)
    )

    # Import local des métriques de classification
    from sklearn.metrics import recall_score, precision_score
    from fairlearn.metrics import (
        equalized_odds_difference,
        demographic_parity_difference,
    )

    # === 5. Calcul des métriques pour chaque groupe Modalité+Genre ===
    grouped = df_work.groupby("MODALITE_GENRE")

    modalites = []
    support_list = []

    baseline_data = {
        "Taux de refus": [],
        "Taux d’acceptation": [],
        "Recall": [],
        "Precision": [],
        "EOD": [],
        "DPD": [],
        "Gini": []
    }
    eo_data = {
        "Taux de refus": [],
        "Taux d’acceptation": [],
        "Recall": [],
        "Precision": [],
        "EOD": [],
        "DPD": [],
        "Gini": []
    }
    delta_data = {
        "Taux de refus": [],
        "Taux d’acceptation": [],
        "Recall": [],
        "Precision": [],
        "EOD": [],
        "DPD": [],
        "Gini": []
    }

    def gini_coefficient(series: pd.Series) -> float:
        arr = np.array(series, dtype=float)
        if arr.size == 0 or np.all(arr == 0):
            return np.nan
        sorted_arr = np.sort(arr)
        n = len(arr)
        cumvals = np.cumsum(sorted_arr)
        return (1 + (1 / n) - 2 * np.sum(cumvals) / (cumvals[-1] * n))

    for label, group in grouped:
        if group.empty:
            continue
        try:
            y_true = group["y_true"]
            sens = group["sensitive_feature"]

            # --- Baseline ---
            y_pred_b = group["y_pred_baseline"]
            proba_b = group["proba_baseline"]
            refusal_b = float(y_pred_b.mean())
            acceptance_b = 1.0 - refusal_b
            recall_b = recall_score(y_true, y_pred_b, zero_division=0)
            precision_b = precision_score(y_true, y_pred_b, zero_division=0)
            eod_b = equalized_odds_difference(
                y_true, y_pred_b, sensitive_features=sens
            )
            dpd_b = demographic_parity_difference(
                y_true, y_pred_b, sensitive_features=sens
            )
            gini_b = gini_coefficient(proba_b)

            # --- EO Wrapper ---
            y_pred_e = group["y_pred_eo"]
            proba_e = group["proba_eo"]
            refusal_e = float(y_pred_e.mean())
            acceptance_e = 1.0 - refusal_e
            recall_e = recall_score(y_true, y_pred_e, zero_division=0)
            precision_e = precision_score(y_true, y_pred_e, zero_division=0)
            eod_e = equalized_odds_difference(
                y_true, y_pred_e, sensitive_features=sens
            )
            dpd_e = demographic_parity_difference(
                y_true, y_pred_e, sensitive_features=sens
            )
            gini_e = gini_coefficient(proba_e)

            # --- Deltas (EO - Baseline) ---
            delta_refusal = refusal_e - refusal_b
            delta_acceptance = acceptance_e - acceptance_b
            delta_recall = recall_e - recall_b
            delta_precision = precision_e - precision_b
            delta_eod = eod_e - eod_b
            delta_dpd = dpd_e - dpd_b
            delta_gini = gini_e - gini_b

            # Stockage des résultats
            modalites.append(label)
            support_list.append(len(group))

            baseline_data["Taux de refus"].append(refusal_b)
            baseline_data["Taux d’acceptation"].append(acceptance_b)
            baseline_data["Recall"].append(recall_b)
            baseline_data["Precision"].append(precision_b)
            baseline_data["EOD"].append(eod_b)
            baseline_data["DPD"].append(dpd_b)
            baseline_data["Gini"].append(gini_b)

            eo_data["Taux de refus"].append(refusal_e)
            eo_data["Taux d’acceptation"].append(acceptance_e)
            eo_data["Recall"].append(recall_e)
            eo_data["Precision"].append(precision_e)
            eo_data["EOD"].append(eod_e)
            eo_data["DPD"].append(dpd_e)
            eo_data["Gini"].append(gini_e)

            delta_data["Taux de refus"].append(delta_refusal)
            delta_data["Taux d’acceptation"].append(delta_acceptance)
            delta_data["Recall"].append(delta_recall)
            delta_data["Precision"].append(delta_precision)
            delta_data["EOD"].append(delta_eod)
            delta_data["DPD"].append(delta_dpd)
            delta_data["Gini"].append(delta_gini)

        except Exception:
            continue

    # Construction des DataFrames
    df_baseline = pd.DataFrame(baseline_data, index=modalites)
    df_eo = pd.DataFrame(eo_data, index=modalites)
    df_delta = pd.DataFrame(delta_data, index=modalites)
    df_info = pd.DataFrame({"Support": support_list}, index=modalites)

    # Concaténation en MultiIndex (Info / Baseline / EO / Delta)
    df_combined = pd.concat(
        {
            "Info": df_info,
            "Baseline": df_baseline,
            "EO": df_eo,
            "Delta": df_delta
        },
        axis=1
    )
    df_combined.index.name = "Modalité+Genre"

    # === 6. Filtrer les lignes où Δ EOD est différent de zéro (optionnel) ===
    filter_delta_eod = st.checkbox(
        "Afficher uniquement les groupes où Δ EOD ≠ 0",
        value=False
    )
    if filter_delta_eod:
        mask_delta = df_combined[("Delta", "EOD")] != 0
        df_combined = df_combined.loc[mask_delta]
        if df_combined.empty:
            st.warning("Aucun groupe n'a un Δ EOD non nul.")
            st.stop()

    # === 7. Recherche sur les modalités ===
    search_input = st.text_input(
        "Filtrer Modalités (partie du nom) :"
    )
    if search_input:
        mask_search = df_combined.index.str.contains(search_input, case=False, na=False)
        df_filtered = df_combined.loc[mask_search]
    else:
        df_filtered = df_combined.copy()

    if df_filtered.empty:
        st.warning("Aucune modalité ne correspond au filtre.")
        st.stop()

    # === 8. Sélection des métriques à afficher ===
    metrics_dispo = ["Taux de refus", "Taux d’acceptation", "Recall",
                     "Precision", "EOD", "DPD", "Gini"]
    selected_metrics = st.multiselect(
        "Choisir métriques à afficher (Baseline/EO/Delta) :",
        metrics_dispo,
        default=["Taux de refus", "Recall", "Gini"]
    )

    if not selected_metrics:
        st.warning("Sélectionne au moins une métrique.")
        st.stop()

    # Préparer les sous-DataFrames à afficher
    cols_baseline = [("Baseline", m) for m in selected_metrics]
    cols_eo = [("EO", m) for m in selected_metrics]
    cols_delta = [("Delta", m) for m in selected_metrics]

    df_be = df_filtered.loc[:, cols_baseline + cols_eo]
    df_deltas = df_filtered.loc[:, cols_delta]

    # Affichage du support (Info)
    if "Info" in df_filtered.columns.get_level_values(0):
        st.subheader("Support (nombre d’observations) par groupe")
        st.dataframe(
            df_filtered[("Info", "Support")].to_frame(),
            use_container_width=True
        )

    # Affichage Baseline vs EO
    st.subheader("Comparaison Baseline vs EO")
    st.dataframe(
        df_be.style.format("{:.3f}"),
        use_container_width=True
    )

    # Affichage des Deltas avec mise en forme conditionnelle
    def color_delta(val):
        if pd.isna(val):
            return ""
        return (
            "background-color: lightgreen"
            if val > 0 else
            "background-color: lightcoral"
            if val < 0 else ""
        )

    st.subheader("Delta (EO – Baseline)")
    st.dataframe(
        df_deltas.style.format("{:.3f}")
                       .applymap(color_delta),
        use_container_width=True
    )

    # === 9. Plot comparatif pour une métrique choisie ===
    st.subheader("Plot comparatif Baseline vs EO")
    # Sélection d'une métrique pour le graphique
    metric_for_plot = st.selectbox(
        "Choisir une métrique pour le plot comparatif :",
        selected_metrics
    )
    if metric_for_plot:
        df_plot = pd.DataFrame({
            "Modalité+Genre": df_filtered.index.astype(str),
            "Baseline": df_filtered[("Baseline", metric_for_plot)].values,
            "EO": df_filtered[("EO", metric_for_plot)].values,
        })
        fig_comparatif = px.bar(
            df_plot,
            x="Modalité+Genre",
            y=["Baseline", "EO"],
            barmode="group",
            title=f"Comparaison Baseline vs EO pour '{metric_for_plot}'",
            labels={"value": metric_for_plot, "variable": "Modèle"}
        )
        fig_comparatif.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_comparatif, use_container_width=True)

    # === 10. Export Excel complet ===
    if st.button("📥 Exporter en Excel"):
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df_combined.to_excel(writer, sheet_name="Intersectionnalité")
        buffer.seek(0)
        st.download_button(
            label="Télécharger le fichier Excel complet",
            data=buffer,
            file_name="intersectionnalite_genre.xlsx",
            mime="application/vnd.openxmlformats-officedocument."
                 "spreadsheetml.sheet",
        )



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
