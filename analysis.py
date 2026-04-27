"""
Modules d'analyse statistique et machine learning
INF232 — DataCollect Pro
"""

import numpy as np
import pandas as pd
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (classification_report, confusion_matrix,
                              silhouette_score, accuracy_score)
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img


def _get_numeric(df):
    cols = ['age', 'poids', 'taille', 'tension_sys', 'tension_dia',
            'glycemie', 'cholesterol', 'imc']
    return df[[c for c in cols if c in df.columns]].dropna()


# ── 1. Régression linéaire simple ────────────────────────────────────────────

def regression_simple(df, x_col, y_col):
    data = df[[x_col, y_col]].dropna()
    X = sm.add_constant(data[x_col].values)
    y = data[y_col].values
    model = sm.OLS(y, X).fit()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(data[x_col], data[y_col], alpha=0.5, color='#378ADD', s=30, label='Observations')
    x_range = np.linspace(data[x_col].min(), data[x_col].max(), 200)
    y_pred = model.params[0] + model.params[1] * x_range
    ax.plot(x_range, y_pred, color='#D85A30', linewidth=2, label='Droite de régression')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'Régression linéaire simple : {y_col} ~ {x_col}')
    ax.legend()
    plt.tight_layout()
    img = fig_to_b64(fig)

    return {
        'image': img,
        'equation': f'{y_col} = {model.params[0]:.4f} + {model.params[1]:.4f} × {x_col}',
        'r2': round(model.rsquared, 4),
        'r2_adj': round(model.rsquared_adj, 4),
        'pvalue': round(float(model.f_pvalue), 6),
        'coef': round(float(model.params[1]), 4),
        'intercept': round(float(model.params[0]), 4),
        'n': len(data),
        'interpretation': _interpret_r2(model.rsquared)
    }


def _interpret_r2(r2):
    if r2 >= 0.8:
        return 'Excellente relation linéaire (R² ≥ 0.80)'
    elif r2 >= 0.6:
        return 'Bonne relation linéaire (0.60 ≤ R² < 0.80)'
    elif r2 >= 0.4:
        return 'Relation modérée (0.40 ≤ R² < 0.60)'
    elif r2 >= 0.2:
        return 'Relation faible (0.20 ≤ R² < 0.40)'
    else:
        return 'Relation très faible ou inexistante (R² < 0.20)'


# ── 2. Régression linéaire multiple ──────────────────────────────────────────

def regression_multiple(df, x_cols, y_col):
    valid_x = [c for c in x_cols if c in df.columns]
    data = df[valid_x + [y_col]].dropna()
    X = sm.add_constant(data[valid_x].values)
    y = data[y_col].values
    model = sm.OLS(y, X).fit()

    # Graphique des coefficients
    coefs = dict(zip(['const'] + valid_x, model.params))
    pvals = dict(zip(['const'] + valid_x, model.pvalues))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Coefficients
    names = valid_x
    vals = [model.params[i+1] for i in range(len(names))]
    colors = ['#1D9E75' if v > 0 else '#D85A30' for v in vals]
    axes[0].barh(names, vals, color=colors)
    axes[0].axvline(0, color='gray', linewidth=0.8, linestyle='--')
    axes[0].set_title('Coefficients de régression')
    axes[0].set_xlabel('Valeur du coefficient')

    # Résidus
    y_pred = model.fittedvalues
    residuals = y - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.4, color='#534AB7', s=20)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Valeurs ajustées')
    axes[1].set_ylabel('Résidus')
    axes[1].set_title('Graphique des résidus')
    plt.tight_layout()
    img = fig_to_b64(fig)

    coef_table = []
    for i, name in enumerate(['const'] + valid_x):
        coef_table.append({
            'variable': name,
            'coef': round(float(model.params[i]), 4),
            'std_err': round(float(model.bse[i]), 4),
            'p_value': round(float(model.pvalues[i]), 4),
            'significatif': '***' if model.pvalues[i] < 0.001 else
                           '**' if model.pvalues[i] < 0.01 else
                           '*' if model.pvalues[i] < 0.05 else 'ns'
        })

    return {
        'image': img,
        'r2': round(model.rsquared, 4),
        'r2_adj': round(model.rsquared_adj, 4),
        'f_stat': round(float(model.fvalue), 4),
        'f_pvalue': round(float(model.f_pvalue), 6),
        'aic': round(float(model.aic), 2),
        'bic': round(float(model.bic), 2),
        'n': len(data),
        'coef_table': coef_table,
        'interpretation': _interpret_r2(model.rsquared)
    }


# ── 3. Réduction de dimensionnalité ──────────────────────────────────────────

def pca_analysis(df, n_components=2):
    X = _get_numeric(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter PCA
    if n_components >= 2:
        axes[0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, color='#378ADD', s=30)
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        axes[0].set_title('Projection ACP (PC1 vs PC2)')
    else:
        axes[0].bar(range(len(X_pca)), X_pca[:, 0], color='#378ADD', alpha=0.6)
        axes[0].set_title('PC1')

    # Variance expliquée cumulée
    pca_full = PCA(n_components=min(X_scaled.shape[1], 8))
    pca_full.fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    axes[1].bar(range(1, len(cum_var)+1), pca_full.explained_variance_ratio_*100,
                color='#9FE1CB', label='Individuelle')
    axes[1].plot(range(1, len(cum_var)+1), cum_var*100, 'ro-', linewidth=2,
                 label='Cumulée', markersize=4)
    axes[1].axhline(80, color='gray', linestyle='--', linewidth=0.8, label='Seuil 80%')
    axes[1].set_xlabel('Composante principale')
    axes[1].set_ylabel('Variance expliquée (%)')
    axes[1].set_title('Éboulis des valeurs propres')
    axes[1].legend()
    plt.tight_layout()
    img = fig_to_b64(fig)

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=X.columns
    )

    return {
        'image': img,
        'n_components': pca.n_components_,
        'variance_ratio': [round(v*100, 2) for v in pca.explained_variance_ratio_],
        'variance_cumulee': round(float(np.sum(pca.explained_variance_ratio_))*100, 2),
        'loadings': loadings.round(3).to_dict(),
        'n_samples': len(X)
    }


def tsne_analysis(df):
    X = _get_numeric(df)
    if len(X) < 30:
        return {'error': 'Insuffisant (min 30 observations)'}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    perp = min(30, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=500)
    X_tsne = tsne.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                         c=range(len(X_tsne)), cmap='viridis', alpha=0.6, s=25)
    plt.colorbar(scatter, ax=ax, label='Index')
    ax.set_title('t-SNE — Réduction 2D')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    plt.tight_layout()
    img = fig_to_b64(fig)
    return {'image': img, 'n_samples': len(X)}


def lda_analysis(df, target='maladie'):
    X = _get_numeric(df)
    if target not in df.columns:
        return {'error': f'Colonne cible {target} introuvable'}
    y_raw = df.loc[X.index, target].fillna('Inconnue')
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    n_classes = len(np.unique(y))
    n_comp = min(n_classes - 1, X.shape[1], 2)
    if n_comp < 1:
        return {'error': 'Pas assez de classes'}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    X_lda = lda.fit_transform(X_scaled, y)

    fig, ax = plt.subplots(figsize=(9, 6))
    classes = le.classes_
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    for i, (cls, col) in enumerate(zip(classes, colors)):
        mask = y == i
        if n_comp >= 2:
            ax.scatter(X_lda[mask, 0], X_lda[mask, 1], label=cls, color=col, alpha=0.6, s=30)
        else:
            ax.scatter(X_lda[mask, 0], np.zeros(mask.sum()), label=cls, color=col, alpha=0.6, s=30)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.set_title(f'LDA — Projection par {target}')
    ax.set_xlabel('LD1')
    if n_comp >= 2:
        ax.set_ylabel('LD2')
    plt.tight_layout()
    img = fig_to_b64(fig)
    return {
        'image': img,
        'variance_ratio': [round(v*100, 2) for v in lda.explained_variance_ratio_],
        'n_classes': n_classes
    }


# ── 4. Classification supervisée ─────────────────────────────────────────────

def classification_supervisee(df, method='knn', target='statut'):
    X = _get_numeric(df)
    if target not in df.columns:
        return {'error': f'Colonne {target} introuvable'}
    y_raw = df.loc[X.index, target].fillna('Inconnu')
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    models = {
        'knn': KNeighborsClassifier(n_neighbors=5),
        'svm': SVC(kernel='rbf', probability=True, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'decision_tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'logistic': LogisticRegression(max_iter=500, random_state=42)
    }

    clf = models.get(method, models['knn'])
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')

    # Matrice de confusion
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f'Matrice de confusion — {method.upper()}')
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Réel')
    plt.tight_layout()
    img = fig_to_b64(fig)

    # Feature importance (RF seulement)
    feat_img = None
    if method == 'random_forest':
        importances = clf.feature_importances_
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        sorted_idx = np.argsort(importances)
        ax2.barh(X.columns[sorted_idx], importances[sorted_idx], color='#1D9E75')
        ax2.set_title('Importance des variables (Random Forest)')
        plt.tight_layout()
        feat_img = fig_to_b64(fig2)

    report = classification_report(y_te, y_pred, target_names=le.classes_, output_dict=True)
    return {
        'image': img,
        'feat_img': feat_img,
        'method': method,
        'accuracy': round(acc, 4),
        'cv_mean': round(float(cv_scores.mean()), 4),
        'cv_std': round(float(cv_scores.std()), 4),
        'classes': list(le.classes_),
        'report': report,
        'n_train': len(X_tr),
        'n_test': len(X_te)
    }


# ── 5. Classification non-supervisée ─────────────────────────────────────────

def classification_non_supervisee(df, method='kmeans', k=3):
    X = _get_numeric(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == 'kmeans':
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
    elif method == 'dbscan':
        model = DBSCAN(eps=1.2, min_samples=5)
        labels = model.fit_predict(X_scaled)
        k = len(set(labels)) - (1 if -1 in labels else 0)
    elif method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X_scaled)
    else:
        return {'error': 'Méthode inconnue'}

    # Silhouette score
    unique_labels = set(labels)
    valid = [l for l in labels if l != -1]
    sil = silhouette_score(X_scaled[[l != -1 for l in labels]], valid) if len(set(valid)) >= 2 else 0

    # Visualisation PCA 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scatter = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=labels,
                               cmap='tab10', alpha=0.6, s=25)
    axes[0].set_title(f'Clusters — {method.upper()} (k={k})')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[0])

    # Distribution des clusters
    unique, counts = np.unique(labels, return_counts=True)
    axes[1].bar([str(u) for u in unique], counts,
                color=plt.cm.tab10(np.linspace(0, 1, len(unique))))
    axes[1].set_title('Distribution des clusters')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Nombre d\'observations')
    plt.tight_layout()
    img = fig_to_b64(fig)

    # Elbow pour K-Means
    elbow_img = None
    if method == 'kmeans':
        inertias = []
        k_range = range(2, min(10, len(X) // 5 + 2))
        for ki in k_range:
            km = KMeans(n_clusters=ki, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=5)
        ax2.set_xlabel('Nombre de clusters (k)')
        ax2.set_ylabel('Inertie')
        ax2.set_title('Méthode du coude')
        plt.tight_layout()
        elbow_img = fig_to_b64(fig2)

    cluster_stats = []
    for cl in unique:
        mask = labels == cl
        cluster_stats.append({
            'cluster': int(cl),
            'taille': int(counts[list(unique).index(cl)]),
            'pourcentage': round(float(counts[list(unique).index(cl)] / len(labels) * 100), 1)
        })

    return {
        'image': img,
        'elbow_img': elbow_img,
        'method': method,
        'k': k,
        'silhouette': round(float(sil), 4),
        'cluster_stats': cluster_stats,
        'n': len(X)
    }


# ── Analyse descriptive ───────────────────────────────────────────────────────

def analyse_descriptive(df):
    num_cols = ['age', 'poids', 'taille', 'tension_sys', 'tension_dia',
                'glycemie', 'cholesterol', 'imc']
    num_cols = [c for c in num_cols if c in df.columns]

    stats = {}
    for col in num_cols:
        data = df[col].dropna()
        stats[col] = {
            'n': int(len(data)),
            'mean': round(float(data.mean()), 2),
            'median': round(float(data.median()), 2),
            'std': round(float(data.std()), 2),
            'min': round(float(data.min()), 2),
            'max': round(float(data.max()), 2),
            'q25': round(float(data.quantile(0.25)), 2),
            'q75': round(float(data.quantile(0.75)), 2),
        }

    # Distributions catégorielles
    cat_stats = {}
    for col in ['sexe', 'region', 'maladie', 'statut']:
        if col in df.columns:
            cat_stats[col] = df[col].value_counts().to_dict()

    return {'numerical': stats, 'categorical': cat_stats}
