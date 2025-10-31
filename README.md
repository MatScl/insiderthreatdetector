# Progetto: Rilevamento di comportamenti anomali
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
