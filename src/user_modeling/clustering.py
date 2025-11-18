"""
User Modeling Module - K-Means Clustering
=========================================

Questo modulo implementa il clustering degli utenti basato su comportamenti normali.
Utilizza K-Means per raggruppare utenti con pattern simili e calcolare distanze dai centroidi.
Le distanze elevate indicano comportamenti anomali rispetto al gruppo di appartenenza.

Author: Matteo Sclafani
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple, Optional
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserClusterer:
    """
    Classe per clustering utenti tramite K-Means.
    
    Il clustering raggruppa utenti con comportamenti simili e calcola la distanza
    di ogni utente dal centroide del proprio cluster. Distanze elevate indicano
    comportamenti anomali rispetto al gruppo di appartenenza.
    """
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        """
        Inizializza il clusterer.
        
        Args:
            n_clusters: Numero di cluster da creare (default: 5)
            random_state: Seed per riproducibilità (default: 42)
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.cluster_centers_ = None
        self.labels_ = None
        
    def fit(self, features: pd.DataFrame) -> 'UserClusterer':
        """
        Addestra il modello K-Means sui dati degli utenti.
        
        Args:
            features: DataFrame con feature normalizzate (righe=utenti, colonne=feature)
        
        Returns:
            Self per method chaining
        """
        logger.info(f"Inizio training K-Means con {self.n_clusters} cluster su {len(features)} utenti")
        
        # Verifica che ci siano abbastanza utenti
        if len(features) < self.n_clusters:
            raise ValueError(f"Numero utenti ({len(features)}) < numero cluster ({self.n_clusters})")
        
        # Training K-Means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,  # Numero di inizializzazioni diverse
            max_iter=300  # Massimo numero di iterazioni
        )
        
        self.labels_ = self.kmeans.fit_predict(features)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        
        # Calcola silhouette score per valutare qualità clustering
        silhouette = silhouette_score(features, self.labels_)
        logger.info(f"Clustering completato. Silhouette score: {silhouette:.3f}")
        
        return self
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Assegna cluster a nuovi utenti.
        
        Args:
            features: DataFrame con feature normalizzate
        
        Returns:
            Array di cluster labels
        """
        if self.kmeans is None:
            raise ValueError("Modello non addestrato. Chiamare fit() prima di predict()")
        
        return self.kmeans.predict(features)
    
    def calculate_distances_to_centroid(self, features: pd.DataFrame) -> np.ndarray:
        """
        Calcola distanza euclidea di ogni utente dal centroide del proprio cluster.
        
        Distanze elevate indicano comportamenti anomali rispetto al gruppo normale.
        
        Args:
            features: DataFrame con feature normalizzate
        
        Returns:
            Array di distanze (una per utente)
        """
        if self.kmeans is None:
            raise ValueError("Modello non addestrato. Chiamare fit() prima")
        
        # Ottieni cluster labels
        labels = self.predict(features)
        
        # Calcola distanza euclidea per ogni utente
        distances = np.zeros(len(features))
        centers = self.cluster_centers_ if self.cluster_centers_ is not None else np.array([])
        
        for i, (idx, row) in enumerate(features.iterrows()):
            cluster_id = labels[i]
            centroid = centers[cluster_id]
            # Distanza euclidea: sqrt(sum((x - centroid)^2))
            distances[i] = np.linalg.norm(row.values - centroid)
        
        return distances
    
    def get_cluster_statistics(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola statistiche per ogni cluster.
        
        Args:
            features: DataFrame con feature normalizzate
        
        Returns:
            DataFrame con statistiche per cluster (size, avg_distance, max_distance)
        """
        if self.kmeans is None:
            raise ValueError("Modello non addestrato")
        
        labels = self.predict(features)
        distances = self.calculate_distances_to_centroid(features)
        
        # Crea DataFrame con cluster info
        cluster_df = pd.DataFrame({
            'cluster': labels,
            'distance': distances
        })
        
        # Calcola statistiche aggregate
        stats = cluster_df.groupby('cluster').agg({
            'distance': ['count', 'mean', 'std', 'max']
        }).round(3)
        
        stats.columns = ['size', 'avg_distance', 'std_distance', 'max_distance']
        stats = stats.reset_index()
        
        logger.info(f"\nStatistiche cluster:\n{stats}")
        
        return stats
    
    def find_optimal_k(self, features: pd.DataFrame, k_range: range = range(2, 11)) -> int:
        """
        Trova numero ottimale di cluster usando metodo elbow (inerzia) e silhouette.
        
        Args:
            features: DataFrame con feature normalizzate
            k_range: Range di k da testare (default: 2-10)
        
        Returns:
            Numero ottimale di cluster
        """
        logger.info(f"Ricerca k ottimale in range {k_range}")
        
        inertias = []
        silhouettes = []
        
        for k in k_range:
            # Training con k cluster
            kmeans_temp = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans_temp.fit_predict(features)
            
            # Calcola metriche
            inertias.append(kmeans_temp.inertia_)
            silhouettes.append(silhouette_score(features, labels))
            
            logger.info(f"k={k}: inertia={kmeans_temp.inertia_:.2f}, silhouette={silhouettes[-1]:.3f}")
        
        # Trova k con silhouette massimo
        optimal_k = k_range[np.argmax(silhouettes)]
        logger.info(f"K ottimale (max silhouette): {optimal_k}")
        
        return optimal_k


def cluster_users(features: pd.DataFrame, 
                  n_clusters: Optional[int] = None,
                  find_optimal: bool = False,
                  random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, UserClusterer]:
    """
    Funzione helper per clustering utenti con un'unica chiamata.
    
    Args:
        features: DataFrame con feature normalizzate (righe=utenti, colonne=feature)
        n_clusters: Numero cluster desiderato (se None e find_optimal=False, usa 5)
        find_optimal: Se True, cerca automaticamente k ottimale
        random_state: Seed per riproducibilità
    
    Returns:
        Tupla (cluster_labels, distances, clusterer):
            - cluster_labels: Array con cluster di appartenenza per ogni utente
            - distances: Array con distanza dal centroide per ogni utente
            - clusterer: Oggetto UserClusterer addestrato
    
    Example:
        >>> from src.data_analysis.loader import DataLoader
        >>> from src.data_analysis.normalizer import normalize_features
        >>> 
        >>> # Carica e normalizza dati
        >>> loader = DataLoader('data/raw/user_features.csv')
        >>> data = loader.load()
        >>> normalized_data = normalize_features(data)
        >>> 
        >>> # Clustering automatico
        >>> labels, distances, clusterer = cluster_users(
        >>>     normalized_data,
        >>>     find_optimal=True
        >>> )
        >>> 
        >>> # Aggiungi risultati al DataFrame
        >>> normalized_data['cluster'] = labels
        >>> normalized_data['cluster_distance'] = distances
    """
    logger.info("=== Avvio clustering utenti ===")
    
    # Determina numero cluster
    if find_optimal:
        logger.info("Ricerca automatica k ottimale...")
        temp_clusterer = UserClusterer(n_clusters=2)  # Temporaneo per find_optimal_k
        n_clusters = temp_clusterer.find_optimal_k(features)
    elif n_clusters is None:
        n_clusters = 5
        logger.info(f"Uso k default: {n_clusters}")
    
    # Inizializza e addestra clusterer
    clusterer = UserClusterer(n_clusters=n_clusters, random_state=random_state)
    clusterer.fit(features)
    
    # Calcola labels e distanze
    cluster_labels = clusterer.labels_ if clusterer.labels_ is not None else np.array([])
    distances = clusterer.calculate_distances_to_centroid(features)
    
    # Mostra statistiche
    clusterer.get_cluster_statistics(features)
    
    logger.info(f"Clustering completato: {n_clusters} cluster, {len(features)} utenti")
    logger.info(f"Distanza media dai centroidi: {distances.mean():.3f}")
    logger.info(f"Distanza massima: {distances.max():.3f}")
    
    return cluster_labels, distances, clusterer
