# Insider Threat Detector

Progetto di Sistemi intelligenti per internet - Sistema di rilevamento delle minacce interne mediante analisi dati e machine learning.

## Struttura del Progetto

```
insiderthreatdetector/
│
├── data/                          # Directory per i dati
│   ├── raw/                       # Dati grezzi
│   ├── processed/                 # Dati processati
│   └── external/                  # Dati esterni
│
├── notebooks/                     # Jupyter notebooks per analisi esplorative
│   └── exploratory_analysis.ipynb
│
├── src/                           # Codice sorgente del progetto
│   ├── __init__.py               # Package initialization
│   ├── preprocessing.py          # Modulo per preprocessing dei dati
│   ├── user_modeling.py          # Modulo per modellazione utenti
│   ├── collaborative_filtering.py # Modulo per filtering collaborativo
│   └── visualization.py          # Modulo per visualizzazioni
│
├── models/                        # Modelli salvati e serializzati
│   └── trained_models/           # Modelli addestrati
│
├── requirements.txt               # Dipendenze Python
└── README.md                      # Questo file

```

## Descrizione

Questo progetto implementa un sistema di rilevamento delle minacce interne (insider threat detection) utilizzando tecniche di:
- **Analisi dei dati** per identificare pattern comportamentali
- **Machine Learning** per modellare il comportamento degli utenti
- **Collaborative Filtering** per identificare anomalie basate su comportamenti simili
- **Visualizzazione** per l'interpretazione dei risultati

## Installazione

1. Clona il repository:
```bash
git clone https://github.com/MatScl/insiderthreatdetector.git
cd insiderthreatdetector
```

2. Crea un ambiente virtuale (raccomandato):
```bash
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate
```

3. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

## Moduli Principali

### preprocessing.py
Gestisce il caricamento, pulizia e trasformazione dei dati:
- `load_data()`: Carica i dati da file
- `clean_data()`: Pulisce e preprocessa i dati grezzi
- `extract_features()`: Estrae le feature dai dati preprocessati

### user_modeling.py
Implementa la modellazione del comportamento degli utenti:
- `build_user_profile()`: Costruisce il profilo comportamentale di un utente
- `detect_anomalies()`: Rileva anomalie nel comportamento
- `update_profile()`: Aggiorna il profilo con nuovi dati

### collaborative_filtering.py
Implementa tecniche di filtering collaborativo:
- `build_user_similarity_matrix()`: Costruisce matrice di similarità tra utenti
- `find_similar_users()`: Trova utenti con comportamenti simili
- `recommend_threat_indicators()`: Raccomanda indicatori di minaccia

### visualization.py
Fornisce strumenti di visualizzazione:
- `plot_user_behavior()`: Visualizza il comportamento di un utente
- `plot_threat_heatmap()`: Crea heatmap dei punteggi di minaccia
- `plot_anomaly_timeline()`: Visualizza anomalie su timeline
- `generate_report()`: Genera report di analisi

## Uso

```python
from src import preprocessing, user_modeling, visualization

# Carica e preprocessa i dati
data = preprocessing.load_data('data/raw/user_logs.csv')
clean_data = preprocessing.clean_data(data)
features = preprocessing.extract_features(clean_data)

# Costruisci profili utente
user_profile = user_modeling.build_user_profile(features)

# Rileva anomalie
anomaly_score = user_modeling.detect_anomalies(user_profile, current_behavior)

# Visualizza risultati
visualization.plot_user_behavior(data, user_id='USER001')
```

## Sviluppo

Per contribuire al progetto:
1. Crea un branch per la tua feature
2. Implementa le modifiche
3. Aggiungi test appropriati
4. Crea una pull request

## Licenza

Progetto sviluppato per il corso di Sistemi Intelligenti per Internet.

## Autori

- MatScl
