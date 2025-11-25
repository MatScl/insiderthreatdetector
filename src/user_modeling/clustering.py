"""K-Means clustering per user modeling"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserClusterer:
    """Clustering K-Means per raggruppare utenti simili"""
    
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.cluster_centers_ = None
        self.labels_ = None
        
    def fit(self, features):
        """Fit K-Means sui dati"""
        if len(features) < self.n_clusters:
            raise ValueError(f"Troppi pochi utenti per {self.n_clusters} cluster")
        
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        self.labels_ = self.kmeans.fit_predict(features)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        
        # valuta qualità clustering
        silhouette = silhouette_score(features, self.labels_)
        logger.info(f"K-Means: {self.n_clusters} clusters, silhouette={silhouette:.3f}")
        
        return self
    
    def predict(self, features):
        """Assegna cluster a nuovi utenti"""
        if self.kmeans is None:
            raise ValueError("Call fit() first")
        return self.kmeans.predict(features)
    
    def calculate_distances_to_centroid(self, features):
        """Calcola distanza euclidea dal centroide del proprio cluster"""
        if self.kmeans is None:
            raise ValueError("Call fit() first")
        
        labels = self.predict(features)
        distances = np.zeros(len(features))
        centers = self.cluster_centers_ if self.cluster_centers_ is not None else np.array([])

        # Calcola distanza euclidea per ogni utente associato al cluster
        for i, row in enumerate(features.itertuples(index=False)):
            cluster_id = labels[i]
            centroid = centers[cluster_id]
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
            kmeans_temp = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans_temp.fit_predict(features)
            
            inertias.append(kmeans_temp.inertia_)
            silhouettes.append(silhouette_score(features, labels))
        
        optimal_k = k_range[np.argmax(silhouettes)]
        logger.info(f"Optimal k: {optimal_k} (silhouette={max(silhouettes):.3f})")
        
        return optimal_k


def cluster_users(features, n_clusters=None, find_optimal=False, random_state=42):
    """Helper per clustering veloce"""
    
    if find_optimal:
        temp_clusterer = UserClusterer(n_clusters=2)
        n_clusters = temp_clusterer.find_optimal_k(features)
    elif n_clusters is None:
        n_clusters = 5
    
    clusterer = UserClusterer(n_clusters=n_clusters, random_state=random_state)
    clusterer.fit(features)
    
    cluster_labels = clusterer.labels_ if clusterer.labels_ is not None else np.array([])
    distances = clusterer.calculate_distances_to_centroid(features)
    
    clusterer.get_cluster_statistics(features)
    
    logger.info(f"Completed: {n_clusters} clusters, avg_dist={distances.mean():.3f}")
    
    return cluster_labels, distances, clusterer

