"""
Feature Selection Comparison Pipeline
=======================================
Dataset: soil_pollution_diseases.csv
Target: Case_Resolved (binary, string type)

Methods compared: RF, XGB, K-Means (cluster-variance), HVGS (variance),
                   PCA, Spearman correlation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_TOP_FEATURES_FULL = 8
N_TOP_FEATURES_REDUCED = 7
CV_FOLDS = 5

# ---------------------------------------------------------------
# STEP 1: Load dataset
# ---------------------------------------------------------------
print("=" * 70)
print("STEP 1: Loading dataset...")
print("=" * 70)

df = pd.read_csv("soil_pollution_diseases.csv")
target_col = "Case_Resolved"

print(f"Shape of dataset (raw): {df.shape}")
print(f"Target distribution ({target_col}):")
print(df[target_col].value_counts())
print(f"Target distribution (proportion):")
print(df[target_col].value_counts(normalize=True).round(3))

# ---------------------------------------------------------------
# STEP 2: Drop first two variables (columns)
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Dropping first two variables (columns)...")
print("=" * 70)

dropped_cols = df.columns[:2].tolist()
df = df.drop(columns=dropped_cols)
print(f"Dropped columns: {dropped_cols}")
print(f"Shape after dropping: {df.shape}")

# ---------------------------------------------------------------
# STEP 3: Encode string columns to numeric based on dtype
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Encoding string/object columns to numeric...")
print("=" * 70)

label_encoders = {}
for col in df.columns:
    if df[col].dtype == object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded column: {col} -> classes: {list(le.classes_)}")

print(f"\nShape of dataset (after encoding): {df.shape}")

# Separate features/target AFTER encoding (target was string, now numeric)
y = df[target_col]
X = df.drop(columns=[target_col])

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"Target distribution (post-encoding):")
print(y.value_counts())
print(f"Target distribution (proportion):")
print(y.value_counts(normalize=True).round(3))

feature_names_full = X.columns.tolist()

# ---------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------

def cv_accuracy_rf(X_sub, y, folds=CV_FOLDS):
    """Cross-validate using RandomForest (default settings)."""
    clf = RandomForestClassifier(random_state=RANDOM_STATE)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X_sub, y, cv=skf, scoring="accuracy")
    return scores.mean()

def cv_accuracy_xgb(X_sub, y, folds=CV_FOLDS):
    """Cross-validate using XGBoost (default settings)."""
    clf = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", use_label_encoder=False)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X_sub, y, cv=skf, scoring="accuracy")
    return scores.mean()


# ---------------------------------------------------------------
# FEATURE SELECTION METHODS
# ---------------------------------------------------------------

def select_rf(X, y, n_features):
    """Random Forest feature importance."""
    clf = RandomForestClassifier(random_state=RANDOM_STATE)
    clf.fit(X, y)
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(n_features).index.tolist()
    return top_features

def select_xgb(X, y, n_features):
    """XGBoost feature importance."""
    clf = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", use_label_encoder=False)
    clf.fit(X, y)
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(n_features).index.tolist()
    return top_features

def select_kmeans(X, n_features):
    """
    K-Means based on cluster-variance.
    Number of selected features = number of clusters.
    For each cluster, compute variance per feature, then pick
    top features across all clusters based on variance.
    """
    n_clusters = n_features  # clusters == number of features to select
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)

    # For each cluster, compute variance of each feature within that cluster
    cluster_variances = pd.DataFrame(index=X.columns)
    for c in range(n_clusters):
        cluster_data = X[labels == c]
        if len(cluster_data) > 1:
            cluster_variances[f"cluster_{c}"] = cluster_data.var()
        else:
            cluster_variances[f"cluster_{c}"] = 0

    # For each cluster, get the feature with the max variance (top feature per cluster)
    top_feature_per_cluster = cluster_variances.idxmax(axis=0)
    # Collect unique top features across clusters, ranked by their variance value
    candidates = []
    for c in range(n_clusters):
        feat = top_feature_per_cluster[f"cluster_{c}"]
        var_val = cluster_variances.loc[feat, f"cluster_{c}"]
        candidates.append((feat, var_val))

    # Sort candidates by variance descending, take unique top n_features
    candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
    selected = []
    for feat, _ in candidates_sorted:
        if feat not in selected:
            selected.append(feat)
        if len(selected) == n_features:
            break

    # If fewer unique features than needed (duplicates across clusters), fill from overall variance
    if len(selected) < n_features:
        overall_var = X.var().sort_values(ascending=False)
        for feat in overall_var.index:
            if feat not in selected:
                selected.append(feat)
            if len(selected) == n_features:
                break

    return selected

def select_hvgs(X, n_features):
    """Highly Variable Gene Selection style: rank by variance."""
    variances = X.var().sort_values(ascending=False)
    top_features = variances.head(n_features).index.tolist()
    return top_features

def select_pca(X, n_features):
    """
    PCA-based feature selection: use loadings of PC1
    to rank original features by absolute contribution.
    """
    pca = PCA(n_components=min(n_features, X.shape[1]), random_state=RANDOM_STATE)
    pca.fit(X)
    # Sum absolute loadings across selected components, weighted by explained variance
    loadings = np.abs(pca.components_)
    weighted_loadings = (loadings.T * pca.explained_variance_ratio_).sum(axis=1)
    loading_series = pd.Series(weighted_loadings, index=X.columns)
    top_features = loading_series.sort_values(ascending=False).head(n_features).index.tolist()
    return top_features

def select_spearman(X, y, n_features):
    """Spearman correlation with target, using scipy.stats.spearmanr."""
    correlations = {}
    for col in X.columns:
        corr, _ = spearmanr(X[col], y)
        correlations[col] = abs(corr) if not np.isnan(corr) else 0
    corr_series = pd.Series(correlations)
    top_features = corr_series.sort_values(ascending=False).head(n_features).index.tolist()
    return top_features


# ---------------------------------------------------------------
# STEP 4: Feature selection - TOP 8 from full feature set
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Selecting TOP 8 features (full feature set) per method...")
print("=" * 70)

methods_top8 = {}

print("-> Running RF selection...")
methods_top8["RF"] = select_rf(X, y, N_TOP_FEATURES_FULL)
print(f"   RF top8: {methods_top8['RF']}")

print("-> Running XGB selection...")
methods_top8["XGB"] = select_xgb(X, y, N_TOP_FEATURES_FULL)
print(f"   XGB top8: {methods_top8['XGB']}")

print("-> Running K-Means (cluster-variance) selection...")
methods_top8["KMeans"] = select_kmeans(X, N_TOP_FEATURES_FULL)
print(f"   KMeans top8: {methods_top8['KMeans']}")

print("-> Running HVGS (variance) selection...")
methods_top8["HVGS"] = select_hvgs(X, N_TOP_FEATURES_FULL)
print(f"   HVGS top8: {methods_top8['HVGS']}")

print("-> Running PCA selection...")
methods_top8["PCA"] = select_pca(X, N_TOP_FEATURES_FULL)
print(f"   PCA top8: {methods_top8['PCA']}")

print("-> Running Spearman selection...")
methods_top8["Spearman"] = select_spearman(X, y, N_TOP_FEATURES_FULL)
print(f"   Spearman top8: {methods_top8['Spearman']}")

# ---------------------------------------------------------------
# STEP 5: Cross-validate with top 8 features per method
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: Cross-validating with TOP 8 selected features...")
print("=" * 70)

cv8_results = {}

for method, feats in methods_top8.items():
    X_sub = X[feats]
    if method == "XGB":
        acc = cv_accuracy_xgb(X_sub, y)
    else:
        acc = cv_accuracy_rf(X_sub, y)
    cv8_results[method] = acc
    print(f"   {method}: CV8 Accuracy = {acc:.3f}")

# ---------------------------------------------------------------
# STEP 6: Remove the highest-ranked feature (per method) -> reduced dataset
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: Removing top-ranked feature per method to build reduced dataset...")
print("=" * 70)

methods_top7 = {}

for method, feats in methods_top8.items():
    highest_feature = feats[0]  # first in ranked list = highest
    reduced_features_pool = [f for f in feature_names_full if f != highest_feature]
    X_reduced = X[reduced_features_pool]

    print(f"-> {method}: removed '{highest_feature}', re-selecting top7 from reduced set...")

    if method == "RF":
        top7 = select_rf(X_reduced, y, N_TOP_FEATURES_REDUCED)
    elif method == "XGB":
        top7 = select_xgb(X_reduced, y, N_TOP_FEATURES_REDUCED)
    elif method == "KMeans":
        top7 = select_kmeans(X_reduced, N_TOP_FEATURES_REDUCED)
    elif method == "HVGS":
        top7 = select_hvgs(X_reduced, N_TOP_FEATURES_REDUCED)
    elif method == "PCA":
        top7 = select_pca(X_reduced, N_TOP_FEATURES_REDUCED)
    elif method == "Spearman":
        top7 = select_spearman(X_reduced, y, N_TOP_FEATURES_REDUCED)

    methods_top7[method] = top7
    print(f"   {method} top7 (reduced): {top7}")

# ---------------------------------------------------------------
# STEP 7: Build summary table & save as result.csv
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: Building summary table...")
print("=" * 70)

summary_rows = []
for method in methods_top8.keys():
    summary_rows.append({
        "Method": method,
        "CV8_Accuracy": round(cv8_results[method], 3),
        "Top8_Features": ", ".join(methods_top8[method]),
        "Top7_Features_Reduced": ", ".join(methods_top7[method])
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("result.csv", index=False)

print("\nSummary Table:")
print(summary_df.to_string(index=False))

print("\n" + "=" * 70)
print("DONE. Results saved to 'result.csv'")
print("=" * 70)