"""
Configuration File
==================

Configurazione centralizzata per parametri del sistema.
Modifica questi valori per cambiare il comportamento della pipeline.

Author: Matteo Sclafani
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# Input/Output files
INPUT_FEATURES_FILE = RAW_DATA_DIR / "user_features.csv"
OUTPUT_ANOMALIES_FILE = RESULTS_DIR / "anomalies.csv"
OUTPUT_ANOMALIES_EXTENDED_FILE = RESULTS_DIR / "anomalies_extended.csv"

# Clustering parameters
CLUSTERING_CONFIG = {
    'n_clusters': None,  # None per auto-detection
    'find_optimal': True,  # Cerca k ottimale con silhouette
    'k_range': range(2, 11),  # Range k per ricerca ottimale
    'random_state': 42  # Seed per riproducibilità
}

# Collaborative Filtering parameters
CF_CONFIG = {
    'similarity_threshold': 0.0,  # Soglia minima similarità
    'threshold_percentile': 25  # Percentile per anomalie CF
}

# Graph Ranking parameters
GRAPH_CONFIG = {
    'similarity_threshold': 0.7,  # Soglia per creare archi nel grafo
    'pagerank_alpha': 0.85,  # Damping factor PageRank
    'pagerank_max_iter': 100  # Max iterazioni PageRank
}

# Score combination weights
SCORE_WEIGHTS = {
    'cluster': 0.3,  # Peso distanza cluster
    'cf': 0.4,  # Peso Collaborative Filtering
    'graph': 0.3  # Peso Graph Ranking
}

# Anomaly detection thresholds
ANOMALY_THRESHOLDS = {
    'percentile': 90,  # Percentile per identificare anomalie
    'top_k': 20  # Numero top anomalie da reportare
}

# Feature normalization
NORMALIZATION_CONFIG = {
    'method': 'standard',  # 'standard' o 'minmax'
    'behavioral_features': [
        'logon_count',
        'email_count',
        'file_count',
        'http_count',
        'device_count',
        'off_hours_activity',
        'external_email_ratio'
    ]
}

# Logging
LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'file': 'insider_threat_detection.log',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# Dashboard
DASHBOARD_CONFIG = {
    'port': 8501,
    'default_top_k': 20,
    'default_percentile': 90
}

# Feature extraction (riferimento allo script esterno)
FEATURE_EXTRACTION_CONFIG = {
    'script_path': PROJECT_ROOT / 'feature-extraction-for-CERT-insider-threat-test-datasets' / 'r4.2' / 'feature_extraction.py',
    'input_dir': RAW_DATA_DIR / 'r4.2',
    'output_dir': RAW_DATA_DIR / 'r4.2' / 'ExtractedData'
}


def get_config():
    """
    Restituisce dizionario con tutta la configurazione.
    
    Returns:
        Dictionary con parametri configurazione
    """
    return {
        'paths': {
            'project_root': str(PROJECT_ROOT),
            'input_features': str(INPUT_FEATURES_FILE),
            'output_anomalies': str(OUTPUT_ANOMALIES_FILE),
            'output_extended': str(OUTPUT_ANOMALIES_EXTENDED_FILE)
        },
        'clustering': CLUSTERING_CONFIG,
        'collaborative_filtering': CF_CONFIG,
        'graph': GRAPH_CONFIG,
        'weights': SCORE_WEIGHTS,
        'thresholds': ANOMALY_THRESHOLDS,
        'normalization': NORMALIZATION_CONFIG,
        'logging': LOGGING_CONFIG,
        'dashboard': DASHBOARD_CONFIG
    }


def print_config():
    """
    Stampa configurazione corrente.
    """
    config = get_config()
    
    print("\n" + "="*60)
    print("CONFIGURAZIONE SISTEMA")
    print("="*60)
    
    for section, params in config.items():
        print(f"\n[{section.upper()}]")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    print_config()
