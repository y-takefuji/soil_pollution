import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, KFold

from xgboost import XGBRegressor

# ------------------------------------------------------------------
# 0. Load dataset
# ------------------------------------------------------------------
df = pd.read_csv("Soil_microbe_dataset.csv")

target_col = "β_Glucosidase (µmol/g/h)"

# Show shape and target distribution (BEFORE dropping first column)
print("=== Before dropping first variable ===")
print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df[target_col].describe())

# ------------------------------------------------------------------
# 1. Drop the first variable (first column)
# ------------------------------------------------------------------
df = df.drop(df.columns[0], axis=1)

# ------------------------------------------------------------------
# 2. Parse Soil_Depth_cm (e.g., "10–20" -> midpoint numeric value)
# ------------------------------------------------------------------
def parse_depth(x):
    if isinstance(x, str):
        x = x.replace("–", "-").strip()
        parts = x.split("-")
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2
    return float(x)

df["Soil_Depth_cm"] = df["Soil_Depth_cm"].apply(parse_depth)

# Show shape and target distribution (AFTER dropping first column)
print("\n=== After dropping first variable ===")
print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df[target_col].describe())

# ------------------------------------------------------------------
# 3. Randomly select 1000 rows and save as 1000.csv
# ------------------------------------------------------------------
df_1000 = df.sample(n=1000, random_state=42).reset_index(drop=True)
df_1000.to_csv("1000.csv", index=False)

print("\n=== 1000-row sample ===")
print("Sampled dataset shape:", df_1000.shape)
print("\nTarget distribution (sampled):")
print(df_1000[target_col].describe())

# ------------------------------------------------------------------
# 4. Encode categorical variable: Land_Use_Type (string -> numeric via one-hot)
#    This is required numeric encoding only, NOT scaling/normalization.
# ------------------------------------------------------------------
df_1000 = pd.get_dummies(df_1000, columns=["Land_Use_Type"], drop_first=False)

# ------------------------------------------------------------------
# 5. Split features and target
# ------------------------------------------------------------------
y = df_1000[target_col].values
X = df_1000.drop(columns=[target_col])

feature_names = X.columns.tolist()
X_values = X.values

print("\nFeature count:", len(feature_names))
print(feature_names)

# ------------------------------------------------------------------
# Common function: CV (R2) using RF for evaluating a feature subset
# ------------------------------------------------------------------
def cv_r2_with_rf(X_sub, y, n_splits=5, random_state=0):
    model = RandomForestRegressor(random_state=random_state)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X_sub, y, cv=kf, scoring="r2")
    return scores.mean()

def cv_r2_with_xgb(X_sub, y, n_splits=5, random_state=0):
    model = XGBRegressor(random_state=random_state)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X_sub, y, cv=kf, scoring="r2")
    return scores.mean()

def sig3(x):
    """Format value to 3 significant digits"""
    if x == 0:
        return 0.0
    return float(f"{x:.3g}")

# ==================================================================
# 1. Random Forest (RF) feature selection
# ==================================================================
rf_model = RandomForestRegressor(random_state=0)
rf_model.fit(X_values, y)
rf_importances = rf_model.feature_importances_

rf_order = np.argsort(rf_importances)[::-1]
rf_top6_idx = rf_order[:6]
rf_top6 = [feature_names[i] for i in rf_top6_idx]

rf_cv6_r2 = cv_r2_with_rf(X_values[:, rf_top6_idx], y)

# Remove the highest-importance feature to create reduced dataset
rf_highest_feature = feature_names[rf_order[0]]
reduced_features_rf = [f for f in feature_names if f != rf_highest_feature]
reduced_idx_rf = [feature_names.index(f) for f in reduced_features_rf]
X_reduced_rf = X_values[:, reduced_idx_rf]

rf_model_reduced = RandomForestRegressor(random_state=0)
rf_model_reduced.fit(X_reduced_rf, y)
rf_importances_reduced = rf_model_reduced.feature_importances_
rf_order_reduced = np.argsort(rf_importances_reduced)[::-1]
rf_top5_idx = rf_order_reduced[:5]
rf_top5 = [reduced_features_rf[i] for i in rf_top5_idx]

# ==================================================================
# 2. XGBoost (XGB) feature selection
# ==================================================================
xgb_model = XGBRegressor(random_state=0)
xgb_model.fit(X_values, y)
xgb_importances = xgb_model.feature_importances_

xgb_order = np.argsort(xgb_importances)[::-1]
xgb_top6_idx = xgb_order[:6]
xgb_top6 = [feature_names[i] for i in xgb_top6_idx]

xgb_cv6_r2 = cv_r2_with_xgb(X_values[:, xgb_top6_idx], y)

xgb_highest_feature = feature_names[xgb_order[0]]
reduced_features_xgb = [f for f in feature_names if f != xgb_highest_feature]
reduced_idx_xgb = [feature_names.index(f) for f in reduced_features_xgb]
X_reduced_xgb = X_values[:, reduced_idx_xgb]

xgb_model_reduced = XGBRegressor(random_state=0)
xgb_model_reduced.fit(X_reduced_xgb, y)
xgb_importances_reduced = xgb_model_reduced.feature_importances_
xgb_order_reduced = np.argsort(xgb_importances_reduced)[::-1]
xgb_top5_idx = xgb_order_reduced[:5]
xgb_top5 = [reduced_features_xgb[i] for i in xgb_top5_idx]

# ==================================================================
# 3. KMeans feature selection
#    Importance = between-cluster variance / overall variance per feature
# ==================================================================
def kmeans_feature_importance(X, n_clusters=3, random_state=0):
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(X)
    importances = []
    for j in range(X.shape[1]):
        col = X[:, j]
        overall_var = np.var(col)
        if overall_var == 0:
            importances.append(0.0)
            continue
        weighted_between = 0.0
        overall_mean = np.mean(col)
        for c in np.unique(labels):
            cluster_vals = col[labels == c]
            weighted_between += len(cluster_vals) * (np.mean(cluster_vals) - overall_mean) ** 2
        weighted_between /= len(col)
        importances.append(weighted_between / overall_var)
    return np.array(importances)

kmeans_importances = kmeans_feature_importance(X_values)
kmeans_order = np.argsort(kmeans_importances)[::-1]
kmeans_top6_idx = kmeans_order[:6]
kmeans_top6 = [feature_names[i] for i in kmeans_top6_idx]

kmeans_cv6_r2 = cv_r2_with_rf(X_values[:, kmeans_top6_idx], y)

kmeans_highest_feature = feature_names[kmeans_order[0]]
reduced_features_kmeans = [f for f in feature_names if f != kmeans_highest_feature]
reduced_idx_kmeans = [feature_names.index(f) for f in reduced_features_kmeans]
X_reduced_kmeans = X_values[:, reduced_idx_kmeans]

kmeans_importances_reduced = kmeans_feature_importance(X_reduced_kmeans)
kmeans_order_reduced = np.argsort(kmeans_importances_reduced)[::-1]
kmeans_top5_idx = kmeans_order_reduced[:5]
kmeans_top5 = [reduced_features_kmeans[i] for i in kmeans_top5_idx]

# ==================================================================
# 4. Highly Variable Gene Selection (HVGS) - variance-based selection
# ==================================================================
def hvgs_importance(X):
    return np.var(X, axis=0)

hvgs_importances = hvgs_importance(X_values)
hvgs_order = np.argsort(hvgs_importances)[::-1]
hvgs_top6_idx = hvgs_order[:6]
hvgs_top6 = [feature_names[i] for i in hvgs_top6_idx]

hvgs_cv6_r2 = cv_r2_with_rf(X_values[:, hvgs_top6_idx], y)

hvgs_highest_feature = feature_names[hvgs_order[0]]
reduced_features_hvgs = [f for f in feature_names if f != hvgs_highest_feature]
reduced_idx_hvgs = [feature_names.index(f) for f in reduced_features_hvgs]
X_reduced_hvgs = X_values[:, reduced_idx_hvgs]

hvgs_importances_reduced = hvgs_importance(X_reduced_hvgs)
hvgs_order_reduced = np.argsort(hvgs_importances_reduced)[::-1]
hvgs_top5_idx = hvgs_order_reduced[:5]
hvgs_top5 = [reduced_features_hvgs[i] for i in hvgs_top5_idx]

# ==================================================================
# 5. PCA feature selection
#    Importance = absolute loading on PC1 (default PCA settings)
# ==================================================================
def pca_importance(X):
    pca = PCA()  # default settings
    pca.fit(X)
    loadings = np.abs(pca.components_[0])  # PC1 loadings
    return loadings

pca_importances = pca_importance(X_values)
pca_order = np.argsort(pca_importances)[::-1]
pca_top6_idx = pca_order[:6]
pca_top6 = [feature_names[i] for i in pca_top6_idx]

pca_cv6_r2 = cv_r2_with_rf(X_values[:, pca_top6_idx], y)

pca_highest_feature = feature_names[pca_order[0]]
reduced_features_pca = [f for f in feature_names if f != pca_highest_feature]
reduced_idx_pca = [feature_names.index(f) for f in reduced_features_pca]
X_reduced_pca = X_values[:, reduced_idx_pca]

pca_importances_reduced = pca_importance(X_reduced_pca)
pca_order_reduced = np.argsort(pca_importances_reduced)[::-1]
pca_top5_idx = pca_order_reduced[:5]
pca_top5 = [reduced_features_pca[i] for i in pca_top5_idx]

# ==================================================================
# 6. Spearman correlation feature selection
# ==================================================================
def spearman_importance(X, y):
    importances = []
    for j in range(X.shape[1]):
        corr, _ = spearmanr(X[:, j], y)
        if np.isnan(corr):
            corr = 0.0
        importances.append(abs(corr))
    return np.array(importances)

spearman_importances = spearman_importance(X_values, y)
spearman_order = np.argsort(spearman_importances)[::-1]
spearman_top6_idx = spearman_order[:6]
spearman_top6 = [feature_names[i] for i in spearman_top6_idx]

spearman_cv6_r2 = cv_r2_with_rf(X_values[:, spearman_top6_idx], y)

spearman_highest_feature = feature_names[spearman_order[0]]
reduced_features_spearman = [f for f in feature_names if f != spearman_highest_feature]
reduced_idx_spearman = [feature_names.index(f) for f in reduced_features_spearman]
X_reduced_spearman = X_values[:, reduced_idx_spearman]

spearman_importances_reduced = spearman_importance(X_reduced_spearman, y)
spearman_order_reduced = np.argsort(spearman_importances_reduced)[::-1]
spearman_top5_idx = spearman_order_reduced[:5]
spearman_top5 = [reduced_features_spearman[i] for i in spearman_top5_idx]

# ==================================================================
# 7. Build summary table and save as result.csv
# ==================================================================
summary_data = [
    {
        "method": "RF",
        "CV6_R2": sig3(rf_cv6_r2),
        "top6_features": ", ".join(rf_top6),
        "top5_features": ", ".join(rf_top5),
    },
    {
        "method": "XGB",
        "CV6_R2": sig3(xgb_cv6_r2),
        "top6_features": ", ".join(xgb_top6),
        "top5_features": ", ".join(xgb_top5),
    },
    {
        "method": "KMeans",
        "CV6_R2": sig3(kmeans_cv6_r2),
        "top6_features": ", ".join(kmeans_top6),
        "top5_features": ", ".join(kmeans_top5),
    },
    {
        "method": "HVGS",
        "CV6_R2": sig3(hvgs_cv6_r2),
        "top6_features": ", ".join(hvgs_top6),
        "top5_features": ", ".join(hvgs_top5),
    },
    {
        "method": "PCA",
        "CV6_R2": sig3(pca_cv6_r2),
        "top6_features": ", ".join(pca_top6),
        "top5_features": ", ".join(pca_top5),
    },
    {
        "method": "Spearman",
        "CV6_R2": sig3(spearman_cv6_r2),
        "top6_features": ", ".join(spearman_top6),
        "top5_features": ", ".join(spearman_top5),
    },
]

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("result.csv", index=False)

print("\n=== Summary Table ===")
print(summary_df)