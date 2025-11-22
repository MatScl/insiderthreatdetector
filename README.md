# Insider Threat Detection

Sistema di anomaly detection per CERT dataset v4.2. Combina clustering, collaborative filtering e graph ranking per trovare utenti anomali.

## Quick Start

```bash
# installa
pip install -r requirements.txt

# preprocessing (se serve)
cd feature-extraction-for-CERT-insider-threat-test-datasets/r4.2/
python feature_extraction.py  # 3-4 ore
cp ExtractedData/sessionr4.2.csv ../../data/raw/user_features.csv

# run
python main.py --find-optimal-k --top-k 20

# dashboard
streamlit run src/visualization/dashboard.py
```

## Come funziona

1. **Clustering** (K-Means) → distanza da centroide cluster
2. **Collaborative Filtering** → similarità coseno, utenti isolati = anomali  
3. **Graph Ranking** → PageRank su grafo similarità

Score finale = 0.3×cluster + 0.4×CF + 0.3×graph (normalizzati [0,1])  
Threshold = 90° percentile (top 10%)

## Output

`data/results/anomalies.csv` con colonne:
- `rank`: posizione (1 = più anomalo)
- `user_id`, `final_anomaly_score`
- `cluster_distance_norm`, `cf_score_norm`, `graph_score_norm`

## Config

Modifica `config.py` per cambiare pesi, soglie, parametri clustering.

## Limiti

- Feature aggregate settimanalmente (no real-time)
- Parametri fissi (no auto-tuning)
- Ground truth non integrato (validazione manuale)
- CF memory-based (scalabilità limitata oltre 1000 utenti)

## Riferimenti

- Dataset: https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099
- Feature extraction: https://github.com/lcd-dal/feature-extraction-for-CERT-insider-threat-test-datasets

---
Sistemi Intelligenti per Internet - UniRoma3 - 2024/2025
