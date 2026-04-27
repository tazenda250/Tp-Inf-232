# DataCollect Pro — INF232 EC2
## Application de collecte et analyse de données — Santé Publique Cameroun

---

## Fonctionnalités

### Collecte de données
- Formulaire de saisie clinique complet (patients)
- Import CSV / Excel (drag & drop)
- Base SQLite persistante
- Export CSV des données

### Analyses statistiques (5 modules)
1. **Régression linéaire simple** — OLS, R², p-value, graphique
2. **Régression linéaire multiple** — coefficients, résidus, AIC/BIC
3. **Réduction de dimensionnalité** — ACP, LDA, t-SNE
4. **Classification supervisée** — KNN, SVM, Random Forest, Arbre, Régression logistique
5. **Clustering non-supervisé** — K-Means, DBSCAN, Hiérarchique

---

## Lancement local

```bash
# 1. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application
python app.py
# → http://localhost:5000
```

---

## Déploiement en ligne (Render.com — GRATUIT)

### Étape 1 : GitHub
```bash
git init
git add .
git commit -m "DataCollect Pro INF232"
git remote add origin https://github.com/votre-username/datacollect-pro.git
git push -u origin main
```

### Étape 2 : Render.com
1. Créer un compte sur https://render.com
2. "New" → "Web Service"
3. Connecter votre repository GitHub
4. Configuration :
   - **Build Command** : `pip install -r requirements.txt`
   - **Start Command** : `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Environment** : Python 3.11
5. Cliquer "Deploy" → URL générée automatiquement

### Alternatives gratuites
- **Railway** : https://railway.app
- **PythonAnywhere** : https://www.pythonanywhere.com
- **Koyeb** : https://www.koyeb.com

---

## Structure du projet

```
datacollect_pro/
├── app.py                 ← Application Flask principale
├── requirements.txt       ← Dépendances Python
├── Procfile               ← Configuration déploiement
├── datacollect.db         ← Base SQLite (auto-créée)
├── utils/
│   ├── __init__.py
│   └── analysis.py        ← Modules ML & stats
├── templates/
│   ├── base.html          ← Layout principal
│   ├── index.html         ← Tableau de bord
│   ├── collecte.html      ← Formulaire saisie
│   ├── donnees.html       ← Liste patients
│   ├── import.html        ← Import fichiers
│   └── analyse.html       ← Toutes les analyses
└── static/
    └── uploads/           ← Fichiers importés
```

---

## Technologies utilisées
- **Backend** : Python 3.11, Flask 3.0
- **ML/Stats** : scikit-learn, statsmodels, pandas, numpy
- **Visualisation** : matplotlib, seaborn
- **Frontend** : Bootstrap 5.3, Chart.js
- **Base de données** : SQLite
- **Déploiement** : Gunicorn + Render/Railway

---

## Critères de notation INF232

| Critère | Réalisation |
|---------|-------------|
| **Créativité** | Secteur santé publique Cameroun, données épidémiologiques réalistes, dashboard interactif |
| **Robustesse** | Validation des entrées, gestion d'erreurs, pagination, seed données, try/catch |
| **Efficacité** | API REST JSON, calculs asynchrones côté client, cache SQLite, exports rapides |
| **Fiabilité** | Tests croisés CV-5-fold, métriques multiples (R², AIC, silhouette), seed=42 |
