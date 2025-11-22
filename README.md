# Insider Threat Detection System

Sistema di rilevamento insider threat basato su machine learning, utilizzando il dataset CERT Insider Threat Test Dataset v4.2.

## Panoramica

Il sistema combina tre tecniche di anomaly detection per identificare comportamenti sospetti:

1. **K-Means Clustering**: Raggruppa utenti con comportamenti simili e calcola distanze dai centroidi
2. **Collaborative Filtering**: Identifica utenti isolati tramite matrice di similarità coseno
3. **Graph Ranking**: Usa PageRank su grafo di similarità per trovare utenti periferici

Gli score vengono normalizzati e combinati in un ranking finale degli utenti più anomali.

## Struttura Progetto

```
project/
├── main.py                          # Pipeline principale
├── config.py                        # Configurazione parametri
├── requirements.txt                 # Dipendenze Python
├── README.md                        # Documentazione
│
├── data/
│   ├── raw/
│   │   ├── r4.2/                   # Dataset CERT ridotto
│   │   └── user_features.csv       # Feature estratte (generato)
│   ├── processed/                   # Dati processati
│   └── results/
│       ├── anomalies.csv           # Risultati anomaly detection
│       └── anomalies_extended.csv  # Risultati con tutte le feature
│
├── src/
│   ├── data_analysis/
│   │   ├── loader.py               # Caricamento CSV
│   │   └── normalizer.py           # Normalizzazione feature
│   │
│   ├── user_modeling/
│   │   └── clustering.py           # K-Means clustering
│   │
│   ├── anomaly_detection/
│   │   ├── collaborative_filtering.py  # CF anomaly detection
│   │   ├── graph_ranking.py            # Graph-based detection
│   │   └── scorer.py                   # Combinazione score
│   │
│   └── visualization/
│       ├── plots.py                # Funzioni plotting Plotly
│       └── dashboard.py            # Dashboard Streamlit
│
└── feature-extraction-for-CERT-insider-threat-test-datasets/
    └── r4.2/
        ├── feature_extraction.py   # Script estrazione feature
        └── ExtractedData/          # Output feature estratte
```

## Quick Start

### 1. Installazione Dipendenze

```bash
pip install -r requirements.txt
```

### 2. Estrazione Feature (se non già fatto)

```bash
cd feature-extraction-for-CERT-insider-threat-test-datasets/r4.2/
python feature_extraction.py
```

Questo genera `user_features.csv` con ~300 feature comportamentali.

### 3. Copia Feature nel Path Corretto

```bash
cp feature-extraction-for-CERT-insider-threat-test-datasets/r4.2/ExtractedData/sessionr4.2.csv \
   data/raw/user_features.csv
```

### 4. Esegui Pipeline Completa

```bash
python main.py --find-optimal-k --top-k 20
```

### 5. Visualizza Risultati nella Dashboard

```bash
streamlit run src/visualization/dashboard.py
```

Apri browser su `http://localhost:8501`

## Uso Dettagliato

### Pipeline Principale (`main.py`)

```bash
# Uso base con parametri default
python main.py

# Ricerca automatica numero ottimale cluster
python main.py --find-optimal-k

# Specifica numero cluster manualmente
python main.py --n-clusters 7

# Configura soglia similarità per grafo
python main.py --similarity-threshold 0.75

# Cambia pesi delle tecniche (devono sommare a 1)
python main.py --weight-cluster 0.4 --weight-cf 0.3 --weight-graph 0.3

# Output custom
python main.py --input data/raw/features.csv --output results/out.csv

# Mostra top 50 anomalie
python main.py --top-k 50
```

**Parametri disponibili:**

- `--input`: Path CSV input con feature (default: `data/raw/user_features.csv`)
- `--output`: Path CSV output anomalie (default: `data/results/anomalies.csv`)
- `--n-clusters`: Numero cluster K-Means (default: auto)
- `--find-optimal-k`: Cerca k ottimale con silhouette score
- `--similarity-threshold`: Soglia per archi nel grafo (default: 0.7)
- `--weight-cluster`: Peso cluster distance (default: 0.3)
- `--weight-cf`: Peso CF score (default: 0.4)
- `--weight-graph`: Peso graph score (default: 0.3)
- `--top-k`: Numero top anomalie da mostrare (default: 20)
- `--no-summary`: Disabilita sommario testuale

### Uso Moduli Individuali

#### Clustering

```python
from src.data_analysis.loader import DataLoader
from src.data_analysis.normalizer import normalize_features
from src.user_modeling.clustering import cluster_users

# Carica e normalizza
loader = DataLoader('data/raw/user_features.csv')
data = loader.load()
normalized = normalize_features(data)

# Clustering con k ottimale
labels, distances, clusterer = cluster_users(
    normalized,
    find_optimal=True
)

# Aggiungi al DataFrame
normalized['cluster'] = labels
normalized['cluster_distance'] = distances
```

#### Collaborative Filtering

```python
from src.anomaly_detection.collaborative_filtering import detect_anomalies_cf

# Anomaly detection
cf_results, cf_detector = detect_anomalies_cf(
    normalized,
    threshold_percentile=25,
    show_top=10
)

# Top anomalie CF
print(cf_results.head(10))
```

#### Graph Ranking

```python
from src.anomaly_detection.graph_ranking import detect_anomalies_graph

# Costruisci matrice similarità (da CF)
sim_matrix = cf_detector.similarity_matrix_
user_ids = normalized.index.tolist()

# Graph anomaly detection
graph_results, ranker = detect_anomalies_graph(
    sim_matrix,
    user_ids,
    similarity_threshold=0.7
)
```

#### Combinazione Score

```python
from src.anomaly_detection.scorer import combine_anomaly_scores

# Combina score
final_results, scorer = combine_anomaly_scores(
    cluster_distances=normalized['cluster_distance'],
    cf_scores=cf_results['cf_anomaly_score'],
    graph_scores=graph_results['graph_anomaly_score'],
    user_ids=normalized.index,
    weight_cluster=0.3,
    weight_cf=0.4,
    weight_graph=0.3,
    output_path='data/results/anomalies.csv'
)

# Mostra sommario
scorer.print_summary(top_k=20)
```

### Dashboard Streamlit

La dashboard fornisce visualizzazione interattiva dei risultati:

**Features:**
- Overview con metriche chiave
- Distribuzione score con soglie configurabili
- Top anomalie con tabelle interattive
- Confronto tecniche (Cluster vs CF vs Graph)
- Analisi cluster con statistiche dettagliate
- Export CSV filtrato
- Ricerca utenti specifici
- Matrice correlazione feature

**Uso:**

```bash
streamlit run src/visualization/dashboard.py
```

Nella sidebar:
1. Inserisci path a `anomalies.csv`
2. Opzionalmente path a `user_features.csv` per analisi cluster
3. Clicca "Carica Dati"
4. Esplora i 4 tab disponibili

## Configurazione

Modifica `config.py` per cambiare parametri di default:

```python
# Numero cluster
CLUSTERING_CONFIG = {
    'n_clusters': 5,  # None per auto
    'find_optimal': True
}

# Soglia grafo
GRAPH_CONFIG = {
    'similarity_threshold': 0.7
}

# Pesi combinazione
SCORE_WEIGHTS = {
    'cluster': 0.3,
    'cf': 0.4,
    'graph': 0.3
}
```

## Output

### `anomalies.csv`

Contiene ranking finale con colonne:

- `rank`: Posizione ranking (1 = più anomalo)
- `user_id`: ID utente
- `final_anomaly_score`: Score finale combinato [0-1]
- `cluster_distance_norm`: Distanza cluster normalizzata
- `cf_score_norm`: CF score normalizzato
- `graph_score_norm`: Graph score normalizzato
- `cluster`: Cluster di appartenenza

### `anomalies_extended.csv`

Include tutte le feature originali + score.

## Testare il Sistema

Verifica funzionamento moduli:

```bash
# Test clustering
python -c "from src.user_modeling.clustering import UserClusterer; print('OK')"

# Test CF
python -c "from src.anomaly_detection.collaborative_filtering import CollaborativeFilter; print('OK')"

# Test Graph
python -c "from src.anomaly_detection.graph_ranking import GraphRanker; print('OK')"

# Test Scorer
python -c "from src.anomaly_detection.scorer import AnomalyScorer; print('OK')"

# Visualizza config
python config.py
```

## Note Importanti

- **Dataset**: Richiede CERT r4.2 ridotto (~5GB) in `data/raw/r4.2/`
- **Feature extraction**: Processo richiede 3-4 ore su dataset ridotto
- **Memory**: Pipeline richiede ~4-8GB RAM
- **Clustering ottimale**: Ricerca k con silhouette può richiedere tempo
- **Similarità**: Soglia 0.7 per grafo è conservativa, riduci per più connessioni

## Riferimenti

- **Dataset**: [CERT Insider Threat Test Dataset v4.2](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099)
- **Feature Extraction**: [lcd-dal/feature-extraction-for-CERT](https://github.com/lcd-dal/feature-extraction-for-CERT-insider-threat-test-datasets)

## Autore

Progetto per corso Sistemi Intelligenti - Università: Rilevamento di comportamenti anomali
Sistemi Intelligenti per Internet (UniRoma3)

## Descrizione
Sistema per il rilevamento di comportamenti anomali in rete basato su tecniche di Information Retrieval, User Modeling e Collaborative Filtering. L'input è costituito dai log (relazioni utente–risorsa, timestamp, azione) estratti dal dataset CERT Insider Threat, preprocessato con il tool open source feature-extraction-for-CERT-insider-threat-test-datasets.

## Obiettivi
- Analisi e normalizzazione dei dati (parsing, feature engineering)
- Costruzione di un modello di profilazione utente basato su metriche comportamentali
- Implementazione di moduli di Collaborative Filtering e ranking su grafo per individuare deviazioni
- Dashboard per la visualizzazione e l'esplorazione dei risultati

## Dataset
- CERT Insider Threat (preprocessato). Inserire i file sotto `data/raw/` o indicare il percorso di importazione
- Script di preprocessing usato: feature-extraction-for-CERT-insider-threat-test-datasets

## Struttura del repository
```
.
├── data/
│   ├── raw/            # dati originali
│   └── processed/      # dati puliti e feature
├── src/
│   └── anomaly_detection/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── user_modeling.py
│       ├── graph_ranking.py
│       ├── evaluation.py
│       └── utils.py
├── notebooks/          # notebook di esplorazione
├── cli/                # script eseguibili
├── dashboard/          # interfaccia Streamlit
│   ├── app.py
│   ├── sample_data/
│   └── README.md
├── results/            # output generati
├── requirements.txt
└── README.md
```

## Dipendenze principali
- Python 3.8+
- pandas, numpy
- scikit-learn
- networkx
- matplotlib / plotly
- streamlit
- jupyter

### Installazione
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Uso rapido

### 1. Preprocessing
```bash
python cli/run_preprocessing.py --input data/raw --output data/processed
```

### 2. User Modeling
```bash
python cli/run_user_modeling.py --input data/processed --output results/profiles
```

### 3. Graph Ranking
```bash
python cli/run_graph_ranking.py --profiles results/profiles --out results/anomalies.csv
```

### 4. Dashboard
```bash
streamlit run dashboard/app.py -- --profiles results/profiles/profiles.csv
```

## Dashboard (Streamlit)

Interfaccia web per l'esplorazione interattiva dei profili utente e delle anomalie rilevate.

### Avvio rapido
```bash
pip install streamlit pandas
streamlit run dashboard/app.py -- --profiles results/profiles/profiles.csv
```

### Funzionalità
- Filtri dinamici (punteggio, user_id, dipartimento)
- Visualizzazione top-N utenti
- Timeline attività per utente selezionato
- Export CSV dei risultati filtrati
