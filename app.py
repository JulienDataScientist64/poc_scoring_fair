import os
import re
import requests
import importlib.util

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import shap
import dalex as dx

from fairlearn.reductions import ExponentiatedGradient
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

# === Chemins & artefacts Hugging Face ===
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
    """Télécharge le fichier depuis Hugging Face si absent localement."""
    if not os.path.exists(filename):
        st.info(f"Téléchargement de {filename} depuis Hugging Face...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            st.success(f"{filename} téléchargé.")
        except Exception as e:
            st.error(f"Erreur lors du téléchargement de {filename}: {e}")
            st.stop()

# --- Download all needed artefacts from Hugging Face ---
for fname, url in ARTEFACTS.items():
    download_if_missing(fname, url)

# === Import dynamique de la classe EOWrapper ===
spec = importlib.util.spec_from_file_location("wrapper_eo", "wrapper_eo.py")
wrapper_eo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper_eo)
EOWrapper = wrapper_eo.EOWrapper

# === Définition des chemins ===
RAW_DATA_FILENAME = "application_train.csv"
MODEL_BASELINE_FILENAME = "lgbm_baseline.joblib"
BASELINE_THRESHOLD_FILENAME = "baseline_threshold.joblib"
MODEL_WRAPPED_EO_FILENAME = "eo_wrapper_with_proba.joblib"
SPLITS_DIR = ""  # splits à la racine

# === Page config Streamlit ===
st.set_page_config(
    page_title="POC Scoring Crédit Équitable",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
def load_model(path):
    return joblib.load(path)

def sanitize_feature_names(df_input):
    df = df_input.copy()
    new_columns = []
    counts = {}
    for col in df.columns:
        new_col = str(col)
        new_col = re.sub(r"[^A-Za-z0-9_.-]+", "_", new_col)
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
try:
    model_baseline = load_model(MODEL_BASELINE_FILENAME)
    st.sidebar.success("Modèle baseline chargé !")
except Exception as e:
    st.error(f"Erreur de chargement du modèle baseline : {e}")
    st.stop()
try:
    optimal_thresh_baseline = joblib.load(BASELINE_THRESHOLD_FILENAME)
    st.sidebar.info(f"Seuil optimal baseline : {optimal_thresh_baseline:.3f}")
except Exception as e:
    st.warning(f"Seuil baseline non trouvé, fallback 0.5 ({e})")
    optimal_thresh_baseline = 0.5
try:
    model_eo_wrapper = load_model(MODEL_WRAPPED_EO_FILENAME)
    assert isinstance(model_eo_wrapper, EOWrapper)
    st.sidebar.success("EO Wrapper chargé !")
except Exception as e:
    st.error(f"Erreur de chargement du modèle EO Wrapper : {e}")
    st.stop()

try:
    X_valid = load_parquet_file("X_valid_pre.parquet")
    y_valid = load_parquet_file("y_valid.parquet").squeeze()
    A_valid = load_parquet_file("A_valid.parquet").squeeze()
    X_test = load_parquet_file("X_test_pre.parquet")
    y_test = load_parquet_file("y_test.parquet").squeeze()
    A_test = load_parquet_file("A_test.parquet").squeeze()
except Exception as e:
    st.error(f"Erreur de chargement des splits de données : {e}")
    st.stop()

X_valid = sanitize_feature_names(X_valid)
X_test = sanitize_feature_names(X_test)

# EDA brute pour analyse
try:
    df_eda_raw_sample = load_csv_sample(RAW_DATA_FILENAME, sample_frac=0.3)
except Exception:
    df_eda_raw_sample = None

# === Fonctions métriques ===
def compute_classification_metrics(y_true, y_pred_hard, y_pred_proba_positive_class):
    return {
        "AUC": roc_auc_score(y_true, y_pred_proba_positive_class),
        "Accuracy": accuracy_score(y_true, y_pred_hard),
        "Precision (1)": precision_score(y_true, y_pred_hard, zero_division=0),
        "Recall (1)": recall_score(y_true, y_pred_hard, zero_division=0),
        "F1 (1)": f1_score(y_true, y_pred_hard, zero_division=0),
        "Taux de sélection": np.mean(y_pred_hard),
    }

def compute_fairness_metrics(y_true, y_pred_hard, sensitive_features):
    try:
        dpd = demographic_parity_difference(y_true, y_pred_hard, sensitive_features=sensitive_features)
        eod = equalized_odds_difference(y_true, y_pred_hard, sensitive_features=sensitive_features)
        return {"DPD": dpd, "EOD": eod}
    except Exception:
        return {"DPD": np.nan, "EOD": np.nan}

# === Sidebar navigation ===
st.sidebar.title("📊 POC Scoring Équitable")
page_options = [
    "Contexte & Objectifs", "Méthodologie", "Analyse Exploratoire", "Résultats & Comparaisons",
    "ROC/Proba - Baseline", "ROC/Proba - EO Wrapper",
    "Intersectionnalité", "Explicabilité Locale", "Explicabilité Globale"
]
page = st.sidebar.radio("Navigation", page_options, index=0)
st.sidebar.markdown("---")
st.sidebar.info(f"Seuil Baseline : {optimal_thresh_baseline:.4f}")
st.sidebar.info(f"Seuil EO Wrapper : {model_eo_wrapper.threshold:.4f}")

# === Page Content ===

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
        Obtenir un modèle performant mais qui reste juste entre les différents groupes (ex : hommes/femmes).
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
          > Mesure la différence de taux d’attribution positive du crédit entre groupes (idéal : zéro différence).
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
        - **Nouvelles variables** : création de ratios simples (ex : mensualité/revenu, crédit/revenu), transformation de l’âge.
        - **Mise en forme** : transformation des variables catégorielles, découpage de l’âge en tranches, etc.
        - **Encodage & imputation** : gestion automatique des valeurs manquantes et transformation des variables pour les modèles.
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
        - Modèle LightGBM ajusté avec Fairlearn pour garantir l’équité entre hommes et femmes.
        - La méthode combine plusieurs modèles et ajuste leurs poids pour minimiser les écarts de traitement selon le genre.
        - On fixe une tolérance maximale sur l’écart d’équité autorisé.
        """
    )

    st.subheader("Évaluation et comparaison")
    st.write(
        """
        - **Performances mesurées** : capacité à bien trier les clients (AUC, précision, rappel, F1).
        - **Équité** : on vérifie que le modèle ne favorise pas un groupe par rapport à l’autre, via des métriques spécifiques.
        - **Analyse détaillée** : matrice de confusion et taux de sélection par groupe.
        """
    )

elif page == "Analyse Exploratoire":
    st.header("🔎 Analyse exploratoire (EDA)")
    if df_eda_raw_sample is not None:
        st.subheader("Aperçu échantillon")
        st.dataframe(df_eda_raw_sample.head(20), use_container_width=True)
        st.subheader("Statistiques descriptives")
        st.dataframe(df_eda_raw_sample.describe(include=np.number).T, use_container_width=True)
        if "TARGET" in df_eda_raw_sample:
            st.subheader("Distribution de la cible (TARGET)")
            tab = pd.DataFrame(df_eda_raw_sample["TARGET"].value_counts()).T
            st.dataframe(tab, use_container_width=True)
            fig = px.histogram(df_eda_raw_sample, x="TARGET", color="TARGET",
                               color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig, use_container_width=True)

elif page == "Résultats & Comparaisons":
    st.header("📊 Résultats comparatifs sur jeu test")
    # Baseline
    y_test_proba_b = model_baseline.predict_proba(X_test)[:, 1]
    y_test_pred_b = (y_test_proba_b >= optimal_thresh_baseline).astype(int)
    metrics_b = compute_classification_metrics(y_test, y_test_pred_b, y_test_proba_b)
    fairness_b = compute_fairness_metrics(y_test, y_test_pred_b, A_test)
    # EO
    y_test_proba_eo = model_eo_wrapper.predict_proba(X_test)[:, 1]
    y_test_pred_eo = model_eo_wrapper.predict(X_test)
    metrics_eo = compute_classification_metrics(y_test, y_test_pred_eo, y_test_proba_eo)
    fairness_eo = compute_fairness_metrics(y_test, y_test_pred_eo, A_test)
    df_res = pd.DataFrame([
        {"Modèle": "Baseline", **metrics_b, **fairness_b},
        {"Modèle": "EO Wrapper", **metrics_eo, **fairness_eo}
    ])
    st.dataframe(df_res.style.format("{:.3f}"), use_container_width=True)

elif page == "ROC/Proba - Baseline":
    st.header("Courbe ROC & Distribution de probas - Baseline")
    y_val_proba = model_baseline.predict_proba(X_valid)[:, 1]
    fpr, tpr, _ = roc_curve(y_valid, y_val_proba)
    auc_b = roc_auc_score(y_valid, y_val_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={auc_b:.3f})'))
    fig_roc.add_shape(type='line', x0=0, x1=1, y0=0, y1=1, line=dict(dash='dash'))
    fig_roc.update_layout(title="Courbe ROC (Validation, Baseline)",
                          xaxis_title="Faux positifs (FPR)",
                          yaxis_title="Vrais positifs (TPR)")
    st.plotly_chart(fig_roc, use_container_width=True)
    # Probas
    st.subheader("Distribution des probabilités (Baseline)")
    df_dist = pd.DataFrame({"proba": y_val_proba, "y": y_valid})
    fig_dist = px.histogram(df_dist, x="proba", color="y", nbins=50, barmode='overlay',
                            color_discrete_sequence=px.colors.qualitative.Safe,
                            labels={"y": "Cible", "proba": "Score Baseline"})
    fig_dist.add_vline(x=optimal_thresh_baseline, line_color="red", line_dash="dash")
    st.plotly_chart(fig_dist, use_container_width=True)

elif page == "ROC/Proba - EO Wrapper":
    st.header("Courbe ROC & Distribution de probas - EO Wrapper")
    y_val_proba = model_eo_wrapper.predict_proba(X_valid)[:, 1]
    fpr, tpr, _ = roc_curve(y_valid, y_val_proba)
    auc_eo = roc_auc_score(y_valid, y_val_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={auc_eo:.3f})'))
    fig_roc.add_shape(type='line', x0=0, x1=1, y0=0, y1=1, line=dict(dash='dash'))
    fig_roc.update_layout(title="Courbe ROC (Validation, EO Wrapper)",
                          xaxis_title="Faux positifs (FPR)",
                          yaxis_title="Vrais positifs (TPR)")
    st.plotly_chart(fig_roc, use_container_width=True)
    # Probas
    st.subheader("Distribution des probabilités (EO Wrapper)")
    df_dist = pd.DataFrame({"proba": y_val_proba, "y": y_valid})
    fig_dist = px.histogram(df_dist, x="proba", color="y", nbins=50, barmode='overlay',
                            color_discrete_sequence=px.colors.qualitative.Safe,
                            labels={"y": "Cible", "proba": "Score EO"})
    fig_dist.add_vline(x=model_eo_wrapper.threshold, line_color="red", line_dash="dash")
    st.plotly_chart(fig_dist, use_container_width=True)

elif page == "Intersectionnalité":
    st.header("Équité intersectionnelle")
    # Attribut sensible croisé genre + âge bin
    df_test_raw = load_csv_sample(RAW_DATA_FILENAME, sample_frac=1.0)
    if "DAYS_BIRTH" in df_test_raw:
        df_test_raw = df_test_raw.assign(AGE_BIN=pd.cut(
            -df_test_raw["DAYS_BIRTH"]/365,
            bins=[17, 25, 35, 45, 55, 65, 100],
            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
        ))
    if "CODE_GENDER" in df_test_raw and "AGE_BIN" in df_test_raw:
        # Match index avec X_test
        idx_test = X_test.index.intersection(df_test_raw.index)
        group = df_test_raw.loc[idx_test, "CODE_GENDER"].astype(str) + " | " + df_test_raw.loc[idx_test, "AGE_BIN"].astype(str)
        y_pred_b = (model_baseline.predict_proba(X_test.loc[idx_test])[:, 1] >= optimal_thresh_baseline).astype(int)
        y_pred_eo = model_eo_wrapper.predict(X_test.loc[idx_test])
        metric_inter = MetricFrame(
            metrics={
                "Selection Rate": fairlearn_selection_rate,
                "Recall": lambda y, y_p: recall_score(y, y_p, pos_label=1, zero_division=0),
            },
            y_true=y_test.loc[idx_test],
            y_pred=y_pred_b,
            sensitive_features=group,
        )
        metric_inter_eo = MetricFrame(
            metrics={
                "Selection Rate": fairlearn_selection_rate,
                "Recall": lambda y, y_p: recall_score(y, y_p, pos_label=1, zero_division=0),
            },
            y_true=y_test.loc[idx_test],
            y_pred=y_pred_eo,
            sensitive_features=group,
        )
        # Tableaux
        df_plot = pd.DataFrame({
            "Groupe": metric_inter.by_group.index,
            "Selection Rate - Baseline": metric_inter.by_group["Selection Rate"].values,
            "Recall - Baseline": metric_inter.by_group["Recall"].values,
            "Selection Rate - EO": metric_inter_eo.by_group["Selection Rate"].values,
            "Recall - EO": metric_inter_eo.by_group["Recall"].values
        })
        st.dataframe(df_plot, use_container_width=True)
        # Barplots
        fig = px.bar(df_plot, y="Groupe", x=["Selection Rate - Baseline", "Selection Rate - EO"], barmode="group",
                     title="Taux de sélection par groupe (Baseline vs EO)",
                     color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.bar(df_plot, y="Groupe", x=["Recall - Baseline", "Recall - EO"], barmode="group",
                      title="Recall par groupe (Baseline vs EO)",
                      color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Impossible de construire les groupes intersectionnels (DAYS_BIRTH ou CODE_GENDER manquant)")

elif page == "Explicabilité Locale":
    st.header("Explicabilité locale (SHAP Force Plot)")
    st.caption("Affiche l'explication d'une prédiction pour un client sélectionné.")
    idx_options = X_test.index.astype(str).tolist()[:1000]
    idx_selected = st.selectbox("ID Client à expliquer :", idx_options, index=0)
    idx_selected = int(idx_selected)
    client_feat = X_test.loc[[idx_selected]]
    # Baseline
    explainer_b = shap.TreeExplainer(model_baseline)
    shap_val_b = explainer_b.shap_values(client_feat)
    expected_val_b = explainer_b.expected_value
    st.markdown("**Force Plot – Modèle Baseline**")
    plot_html = shap.force_plot(
        expected_val_b[1] if isinstance(expected_val_b, list) else expected_val_b,
        shap_val_b[1][0] if isinstance(shap_val_b, list) else shap_val_b[0],
        client_feat, matplotlib=False
    ).html()
    st.components.v1.html(f"<head>{shap.getjs()}</head><body>{plot_html}</body>", height=220, scrolling=True)
    # EO Wrapper (premier estimateur)
    explainer_eo = shap.TreeExplainer(model_eo_wrapper.mitigator.predictors_[0])
    shap_val_eo = explainer_eo.shap_values(client_feat)
    expected_val_eo = explainer_eo.expected_value
    st.markdown("**Force Plot – EO Wrapper (1er estimateur)**")
    plot_html_eo = shap.force_plot(
        expected_val_eo[1] if isinstance(expected_val_eo, list) else expected_val_eo,
        shap_val_eo[1][0] if isinstance(shap_val_eo, list) else shap_val_eo[0],
        client_feat, matplotlib=False
    ).html()
    st.components.v1.html(f"<head>{shap.getjs()}</head><body>{plot_html_eo}</body>", height=220, scrolling=True)

elif page == "Explicabilité Globale":
    st.header("Explicabilité globale (SHAP - DALEX)")
    # SHAP global baseline
    st.subheader("SHAP - Importance globale Baseline")
    sample_size = min(500, X_valid.shape[0])
    X_sample = X_valid.sample(n=sample_size, random_state=42)
    explainer_b = shap.TreeExplainer(model_baseline)
    shap_val_b = explainer_b.shap_values(X_sample)
    feature_importances = np.abs(shap_val_b[1] if isinstance(shap_val_b, list) else shap_val_b).mean(axis=0)
    df_shap_b = pd.DataFrame({
        "Feature": X_sample.columns,
        "Importance (mean|SHAP|)": feature_importances
    }).sort_values("Importance (mean|SHAP|)", ascending=False).head(20)
    st.dataframe(df_shap_b, use_container_width=True)
    fig = px.bar(df_shap_b, x="Importance (mean|SHAP|)", y="Feature", orientation="h",
                     title="Importance globale Baseline (SHAP)", color="Importance (mean|SHAP|)",
                     color_continuous_scale=px.colors.sequential.Plasma)
    st.plotly_chart(fig, use_container_width=True)
    # SHAP global EO (1er estimateur)
    st.subheader("SHAP - Importance globale EO Wrapper (1er estimateur)")
    explainer_eo = shap.TreeExplainer(model_eo_wrapper.mitigator.predictors_[0])
    shap_val_eo = explainer_eo.shap_values(X_sample)
    feature_importances_eo = np.abs(shap_val_eo[1] if isinstance(shap_val_eo, list) else shap_val_eo).mean(axis=0)
    df_shap_eo = pd.DataFrame({
        "Feature": X_sample.columns,
        "Importance (mean|SHAP|)": feature_importances_eo
    }).sort_values("Importance (mean|SHAP|)", ascending=False).head(20)
    st.dataframe(df_shap_eo, use_container_width=True)
    fig2 = px.bar(df_shap_eo, x="Importance (mean|SHAP|)", y="Feature", orientation="h",
                      title="Importance globale EO Wrapper (SHAP)", color="Importance (mean|SHAP|)",
                      color_continuous_scale=px.colors.sequential.Plasma)
    st.plotly_chart(fig2, use_container_width=True)

    # DALEX
    st.subheader("DALEX - Dropout Loss (AUC)")
    exp_b = dx.Explainer(model_baseline, X_sample, y_valid.loc[X_sample.index], label="Baseline", verbose=False)
    loss_b = exp_b.model_parts(loss_function="auc")
    exp_eo = dx.Explainer(model_eo_wrapper.mitigator.predictors_[0], X_sample, y_valid.loc[X_sample.index], label="EO Wrapper", verbose=False)
    loss_eo = exp_eo.model_parts(loss_function="auc")
    df_dalex = pd.DataFrame({
        "Feature": loss_b.result.variable,
        "Dropout_loss Baseline": loss_b.result.dropout_loss,
        "Dropout_loss EO": loss_eo.result.dropout_loss
    }).query("Feature != '_full_model_'").sort_values("Dropout_loss Baseline", ascending=False).head(20)
    st.dataframe(df_dalex, use_container_width=True)
    fig_dx = px.bar(df_dalex, x="Dropout_loss Baseline", y="Feature", orientation="h",
                     title="DALEX Dropout Loss (Baseline)",
                     color="Dropout_loss Baseline", color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig_dx, use_container_width=True)
    fig_dx2 = px.bar(df_dalex, x="Dropout_loss EO", y="Feature", orientation="h",
                      title="DALEX Dropout Loss (EO Wrapper)",
                      color="Dropout_loss EO", color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig_dx2, use_container_width=True)

st.markdown("---")
st.caption(f"POC Scoring Crédit Équitable – {pd.Timestamp.now(tz='Europe/Paris').strftime('%d/%m/%Y %H:%M')}")