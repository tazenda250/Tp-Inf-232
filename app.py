"""
DataCollect Pro - Application de collecte et analyse de données
Secteur : Santé Publique | INF232 - EC2
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file
import pandas as pd
import numpy as np
import sqlite3
import os
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from werkzeug.utils import secure_filename
from utils.analysis import (
    regression_simple, regression_multiple, pca_analysis,
    lda_analysis, tsne_analysis, classification_supervisee,
    classification_non_supervisee, analyse_descriptive
)

app = Flask(__name__)
app.secret_key = 'datacollect_inf232_secret_key_2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# ── Base de données ──────────────────────────────────────────────────────────

def get_db():
    db = sqlite3.connect('datacollect.db')
    db.row_factory = sqlite3.Row
    return db

def init_db():
    with get_db() as db:
        db.executescript('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nom TEXT NOT NULL,
                prenom TEXT NOT NULL,
                age INTEGER,
                sexe TEXT,
                region TEXT,
                poids REAL,
                taille REAL,
                tension_sys INTEGER,
                tension_dia INTEGER,
                glycemie REAL,
                cholesterol REAL,
                imc REAL,
                maladie TEXT,
                statut TEXT,
                date_collecte TEXT,
                observations TEXT
            );
            CREATE TABLE IF NOT EXISTS formulaires (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                titre TEXT,
                description TEXT,
                champs TEXT,
                date_creation TEXT
            );
        ''')
        # Seed données exemple si vide
        count = db.execute('SELECT COUNT(*) FROM patients').fetchone()[0]
        if count == 0:
            _seed_data(db)

def _seed_data(db):
    import random
    regions = ['Centre','Littoral','Ouest','Nord','Adamaoua','Est','Sud','Sud-Ouest','Nord-Ouest','Extrême-Nord']
    maladies = ['Diabète','Hypertension','Paludisme','Tuberculose','VIH','Aucune','Asthme','Cancer']
    statuts = ['Stable','Critique','En traitement','Guéri']
    random.seed(42)
    rows = []
    for i in range(120):
        age = random.randint(18, 80)
        poids = round(random.uniform(45, 110), 1)
        taille = round(random.uniform(1.50, 1.92), 2)
        imc = round(poids / (taille ** 2), 1)
        rows.append((
            f'Patient{i+1}', f'Prénom{i+1}', age,
            random.choice(['M','F']), random.choice(regions),
            poids, taille,
            random.randint(100, 180), random.randint(60, 110),
            round(random.uniform(3.5, 12.0), 1),
            round(random.uniform(1.5, 6.0), 1), imc,
            random.choice(maladies), random.choice(statuts),
            datetime.now().strftime('%Y-%m-%d'), ''
        ))
    db.executemany('''
        INSERT INTO patients (nom,prenom,age,sexe,region,poids,taille,
        tension_sys,tension_dia,glycemie,cholesterol,imc,maladie,statut,date_collecte,observations)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', rows)
    db.commit()

# ── Helpers ──────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def df_from_db():
    with get_db() as db:
        return pd.read_sql('SELECT * FROM patients', db)

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img

# ── Routes principales ───────────────────────────────────────────────────────

@app.route('/')
def index():
    with get_db() as db:
        total = db.execute('SELECT COUNT(*) FROM patients').fetchone()[0]
        maladies_count = db.execute(
            "SELECT maladie, COUNT(*) as n FROM patients GROUP BY maladie ORDER BY n DESC LIMIT 5"
        ).fetchall()
        regions_count = db.execute(
            "SELECT region, COUNT(*) as n FROM patients GROUP BY region ORDER BY n DESC"
        ).fetchall()
    return render_template('index.html',
                           total=total,
                           maladies=maladies_count,
                           regions=regions_count)

@app.route('/collecte', methods=['GET', 'POST'])
def collecte():
    message = None
    if request.method == 'POST':
        data = request.form
        poids = float(data.get('poids', 0) or 0)
        taille = float(data.get('taille', 0) or 1)
        imc = round(poids / (taille ** 2), 1) if taille > 0 else 0
        with get_db() as db:
            db.execute('''
                INSERT INTO patients (nom,prenom,age,sexe,region,poids,taille,
                tension_sys,tension_dia,glycemie,cholesterol,imc,maladie,statut,date_collecte,observations)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                data['nom'], data['prenom'], int(data.get('age', 0)),
                data.get('sexe'), data.get('region'),
                poids, taille,
                int(data.get('tension_sys', 0) or 0),
                int(data.get('tension_dia', 0) or 0),
                float(data.get('glycemie', 0) or 0),
                float(data.get('cholesterol', 0) or 0),
                imc, data.get('maladie'), data.get('statut'),
                datetime.now().strftime('%Y-%m-%d'),
                data.get('observations', '')
            ))
            db.commit()
        message = 'Patient enregistré avec succès !'
    return render_template('collecte.html', message=message)

@app.route('/donnees')
def donnees():
    page = int(request.args.get('page', 1))
    per_page = 20
    offset = (page - 1) * per_page
    filtre = request.args.get('filtre', '')
    with get_db() as db:
        if filtre:
            patients = db.execute(
                "SELECT * FROM patients WHERE maladie=? ORDER BY id DESC LIMIT ? OFFSET ?",
                (filtre, per_page, offset)
            ).fetchall()
            total = db.execute(
                "SELECT COUNT(*) FROM patients WHERE maladie=?", (filtre,)
            ).fetchone()[0]
        else:
            patients = db.execute(
                "SELECT * FROM patients ORDER BY id DESC LIMIT ? OFFSET ?",
                (per_page, offset)
            ).fetchall()
            total = db.execute('SELECT COUNT(*) FROM patients').fetchone()[0]
        maladies = db.execute(
            "SELECT DISTINCT maladie FROM patients ORDER BY maladie"
        ).fetchall()
    pages = (total + per_page - 1) // per_page
    return render_template('donnees.html',
                           patients=patients, page=page,
                           pages=pages, total=total,
                           maladies=maladies, filtre=filtre)

@app.route('/import', methods=['GET', 'POST'])
def import_data():
    message = None
    if request.method == 'POST':
        if 'file' not in request.files:
            message = 'Aucun fichier sélectionné.'
        else:
            f = request.files['file']
            if f and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                f.save(path)
                try:
                    df = pd.read_csv(path) if filename.endswith('.csv') else pd.read_excel(path)
                    session['upload_file'] = path
                    session['upload_cols'] = list(df.columns)
                    session['upload_preview'] = df.head(5).to_html(classes='table table-sm', index=False)
                    message = f'Fichier importé : {len(df)} lignes, {len(df.columns)} colonnes.'
                except Exception as e:
                    message = f'Erreur : {str(e)}'
            else:
                message = 'Format non supporté. Utilisez CSV ou Excel.'
    return render_template('import.html', message=message,
                           preview=session.get('upload_preview'),
                           cols=session.get('upload_cols'))

# ── Routes Analyse ───────────────────────────────────────────────────────────

@app.route('/analyse')
def analyse():
    df = df_from_db()
    stats = analyse_descriptive(df)
    return render_template('analyse.html', stats=stats)

@app.route('/api/descriptive')
def api_descriptive():
    df = df_from_db()
    col = request.args.get('col', 'age')
    if col not in df.columns:
        return jsonify({'error': 'Colonne introuvable'})
    data = df[col].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(data, bins=20, color='#378ADD', edgecolor='white')
    axes[0].set_title(f'Histogramme — {col}')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Fréquence')
    axes[1].boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='#B5D4F4'))
    axes[1].set_title(f'Boxplot — {col}')
    plt.tight_layout()
    img = fig_to_base64(fig)
    return jsonify({
        'image': img,
        'stats': {
            'Moyenne': round(float(data.mean()), 3),
            'Médiane': round(float(data.median()), 3),
            'Écart-type': round(float(data.std()), 3),
            'Min': round(float(data.min()), 3),
            'Max': round(float(data.max()), 3),
            'Q1': round(float(data.quantile(0.25)), 3),
            'Q3': round(float(data.quantile(0.75)), 3),
            'Asymétrie': round(float(data.skew()), 3),
        }
    })

@app.route('/api/regression_simple')
def api_reg_simple():
    df = df_from_db()
    x_col = request.args.get('x', 'age')
    y_col = request.args.get('y', 'tension_sys')
    result = regression_simple(df, x_col, y_col)
    return jsonify(result)

@app.route('/api/regression_multiple')
def api_reg_multiple():
    df = df_from_db()
    y_col = request.args.get('y', 'tension_sys')
    x_cols = request.args.get('x', 'age,poids,imc').split(',')
    result = regression_multiple(df, x_cols, y_col)
    return jsonify(result)

@app.route('/api/pca')
def api_pca():
    df = df_from_db()
    n = int(request.args.get('n', 2))
    result = pca_analysis(df, n_components=n)
    return jsonify(result)

@app.route('/api/tsne')
def api_tsne():
    df = df_from_db()
    result = tsne_analysis(df)
    return jsonify(result)

@app.route('/api/classification')
def api_classification():
    df = df_from_db()
    method = request.args.get('method', 'knn')
    target = request.args.get('target', 'statut')
    result = classification_supervisee(df, method, target)
    return jsonify(result)

@app.route('/api/clustering')
def api_clustering():
    df = df_from_db()
    method = request.args.get('method', 'kmeans')
    k = int(request.args.get('k', 3))
    result = classification_non_supervisee(df, method, k)
    return jsonify(result)

@app.route('/api/correlation')
def api_correlation():
    df = df_from_db()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues',
                ax=ax, linewidths=0.5)
    ax.set_title('Matrice de corrélation')
    plt.tight_layout()
    img = fig_to_base64(fig)
    return jsonify({'image': img})

# ── Export ───────────────────────────────────────────────────────────────────

@app.route('/export/csv')
def export_csv():
    df = df_from_db()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'patients_{datetime.now().strftime("%Y%m%d")}.csv'
    )

@app.route('/api/delete/<int:pid>', methods=['DELETE'])
def delete_patient(pid):
    with get_db() as db:
        db.execute('DELETE FROM patients WHERE id=?', (pid,))
        db.commit()
    return jsonify({'ok': True})

# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
