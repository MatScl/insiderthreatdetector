Progetto universitario di Sistemi Intelligenti per Internet (UniRoma3).
L’obiettivo è sviluppare un sistema intelligente per il rilevamento di comportamenti anomali in rete basato su tecniche di Information Retrieval, User Modeling e Collaborative Filtering.
Il dataset utilizzato è CERT Insider Threat, preprocessato con il tool open source feature-extraction-for-CERT-insider-threat-test-datasets (GitHub).
A partire dai log estratti (relazioni utente–risorsa, timestamp, tipo di azione), voglio costruire:

un modulo di analisi e normalizzazione dati,

un modello di profilazione utente (User Modeling) basato su metriche di comportamento,

un modulo di Collaborative Filtering e ranking su grafo per individuare deviazioni comportamentali,

una dashboard di visualizzazione dei risultati.
Il progetto deve includere codice Python (pandas, scikit-learn, networkx, matplotlib/plotly), una struttura modulare con data/, notebooks/, src/, e un README dettagliato.