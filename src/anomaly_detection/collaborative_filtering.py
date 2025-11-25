"""Anomaly detection tramite Collaborative Filtering user-user"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollaborativeFilter:
    """Rileva anomalie basandosi su similarità coseno tra utenti"""
    
    def __init__(self, similarity_threshold=0.0):
        self.similarity_threshold = similarity_threshold
        self.similarity_matrix_ = None
        self.user_indices_ = None
        
    def build_similarity_matrix(self, features):
        """Costruisce matrice similarità coseno NxN"""
        self.user_indices_ = features.index.tolist()
        self.similarity_matrix_ = cosine_similarity(features.values)
        
        logger.info(f"Similarity matrix: {self.similarity_matrix_.shape}, avg={self._get_avg_similarity_excluding_diagonal():.3f}")
        return self.similarity_matrix_
    
    def _get_avg_similarity_excluding_diagonal(self):
        """Media similarità escludendo diagonale"""
        if self.similarity_matrix_ is None:
            return 0.0
        
        mask = ~np.eye(self.similarity_matrix_.shape[0], dtype=bool)
        return self.similarity_matrix_[mask].mean()
    
    def calculate_avg_similarity(self, user_idx=None):
        """Calcola similarità media per utente/i"""
        if self.similarity_matrix_ is None:
            raise ValueError("Call build_similarity_matrix() first")
        
        n_users = self.similarity_matrix_.shape[0]
        
        if user_idx is not None:
            similarities = self.similarity_matrix_[user_idx].copy()
            similarities[user_idx] = 0
            return similarities.sum() / (n_users - 1)
        else:
            avg_similarities = np.zeros(n_users)
            for i in range(n_users):
                similarities = self.similarity_matrix_[i].copy()
                similarities[i] = 0
                avg_similarities[i] = similarities.sum() / (n_users - 1)
            return avg_similarities
    
    def get_similar_users(self, user_idx, top_k=5):
        """Trova top-k utenti più simili"""
        if self.similarity_matrix_ is None:
            raise ValueError("Call build_similarity_matrix() first")
        
        similarities = self.similarity_matrix_[user_idx].copy()
        similarities[user_idx] = -1
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        user_indices = self.user_indices_ if self.user_indices_ is not None else []
        
        result = pd.DataFrame({
            'user_id': [user_indices[i] for i in top_indices],
            'similarity': similarities[top_indices]
        })
        
        return result
    
    def detect_anomalies(self, features, threshold_percentile=25):
        """Rileva anomalie via bassa similarità media"""
        if self.similarity_matrix_ is None:
            self.build_similarity_matrix(features)
        
        avg_similarities = self.calculate_avg_similarity()
        threshold = np.percentile(avg_similarities, threshold_percentile)
        
        results = pd.DataFrame({
            'user_id': self.user_indices_,
            'avg_similarity': avg_similarities,
            'cf_anomaly_score': 1 - avg_similarities,
            'is_anomaly': avg_similarities < threshold
        })
        
        n_anomalies = results['is_anomaly'].sum()
        logger.info(f"CF anomalies: {n_anomalies}/{len(results)} ({n_anomalies/len(results)*100:.1f}%), threshold={threshold:.3f}")
        
        results = results.sort_values('cf_anomaly_score', ascending=False).reset_index(drop=True)
        
        return results
    
    def get_similarity_statistics(self):
        """Stats sulla matrice di similarità"""
        if self.similarity_matrix_ is None:
            raise ValueError("Call build_similarity_matrix() first")
        
        mask = ~np.eye(self.similarity_matrix_.shape[0], dtype=bool)
        similarities = self.similarity_matrix_[mask]
        
        stats = pd.DataFrame({
            'metric': ['mean', 'median', 'std', 'min', 'max', 'q25', 'q75'],
            'value': [
                similarities.mean(),
                np.median(similarities),
                similarities.std(),
                similarities.min(),
                similarities.max(),
                np.percentile(similarities, 25),
                np.percentile(similarities, 75)
            ]
        })
        
        return stats


def detect_anomalies_cf(features, threshold_percentile=25, show_top=10):
    """Helper per CF anomaly detection"""
    cf_detector = CollaborativeFilter()
    results = cf_detector.detect_anomalies(features, threshold_percentile)
    
    cf_detector.get_similarity_statistics()
    
    if show_top > 0:
        top_anomalies = results.head(show_top)[['user_id', 'avg_similarity', 'cf_anomaly_score']]
        for i, (idx, row) in enumerate(top_anomalies.iterrows(), 1):
            logger.info(f"  {i}. {row['user_id']}: sim={row['avg_similarity']:.3f}, score={row['cf_anomaly_score']:.3f}")
    
    return results, cf_detector

