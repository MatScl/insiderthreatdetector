"""
Collaborative Filtering Anomaly Detection
==========================================

Questo modulo implementa l'anomaly detection basata su Collaborative Filtering.
Calcola la similarità coseno tra utenti e identifica anomalie come utenti
con bassa similarità media rispetto agli altri (comportamenti isolati).

Author: Matteo Sclafani
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollaborativeFilter:
    """
    Detector di anomalie basato su Collaborative Filtering user-user.
    
    Costruisce una matrice di similarità tra utenti basata su cosine similarity.
    Utenti con bassa similarità media sono considerati anomali (comportamenti isolati).
    """
    
    def __init__(self, similarity_threshold: float = 0.0):
        """
        Inizializza il collaborative filter.
        
        Args:
            similarity_threshold: Soglia minima similarità per considerare utenti simili (default: 0.0)
        """
        self.similarity_threshold = similarity_threshold
        self.similarity_matrix_ = None
        self.user_indices_ = None
        
    def build_similarity_matrix(self, features: pd.DataFrame) -> np.ndarray:
        """
        Costruisce matrice di similarità coseno tra tutti gli utenti.
        
        La similarità coseno misura l'angolo tra vettori di feature:
        - 1.0 = utenti identici
        - 0.0 = utenti ortogonali (nessuna correlazione)
        - -1.0 = utenti opposti
        
        Args:
            features: DataFrame con feature normalizzate (righe=utenti, colonne=feature)
        
        Returns:
            Matrice NxN di similarità (N = numero utenti)
        """
        logger.info(f"Costruzione matrice similarità per {len(features)} utenti")
        
        # Salva indici utenti per riferimento
        self.user_indices_ = features.index.tolist()
        
        # Calcola similarità coseno
        # Output: matrice NxN dove [i,j] = similarità tra utente i e utente j
        self.similarity_matrix_ = cosine_similarity(features.values)
        
        # La diagonal è sempre 1 (utente simile a se stesso), la ignoriamo dopo
        logger.info(f"Matrice similarità creata: shape {self.similarity_matrix_.shape}")
        logger.info(f"Similarità media (esclusa diagonale): {self._get_avg_similarity_excluding_diagonal():.3f}")
        
        return self.similarity_matrix_
    
    def _get_avg_similarity_excluding_diagonal(self) -> float:
        """
        Calcola similarità media escludendo la diagonale.
        
        Returns:
            Valore medio di similarità tra utenti diversi
        """
        if self.similarity_matrix_ is None:
            return 0.0
        
        # Maschera per escludere diagonale
        mask = ~np.eye(self.similarity_matrix_.shape[0], dtype=bool)
        return self.similarity_matrix_[mask].mean()
    
    def calculate_avg_similarity(self, user_idx: Optional[int] = None):
        """
        Calcola similarità media di ogni utente rispetto agli altri.
        
        Args:
            user_idx: Se specificato, calcola solo per questo utente (opzionale)
        
        Returns:
            Array con similarità media per ogni utente (o valore singolo se user_idx)
        """
        if self.similarity_matrix_ is None:
            raise ValueError("Matrice similarità non costruita. Chiamare build_similarity_matrix() prima")
        
        n_users = self.similarity_matrix_.shape[0]
        
        if user_idx is not None:
            # Singolo utente: media delle similarità con tutti gli altri
            similarities = self.similarity_matrix_[user_idx].copy()
            similarities[user_idx] = 0  # Escludi se stesso
            return similarities.sum() / (n_users - 1)
        else:
            # Tutti gli utenti
            avg_similarities = np.zeros(n_users)
            for i in range(n_users):
                similarities = self.similarity_matrix_[i].copy()
                similarities[i] = 0  # Escludi diagonale
                avg_similarities[i] = similarities.sum() / (n_users - 1)
            
            return avg_similarities
    
    def get_similar_users(self, user_idx: int, top_k: int = 5) -> pd.DataFrame:
        """
        Trova i k utenti più simili a un dato utente.
        
        Args:
            user_idx: Indice dell'utente target
            top_k: Numero di utenti simili da restituire
        
        Returns:
            DataFrame con utenti simili ordinati per similarità decrescente
        """
        if self.similarity_matrix_ is None:
            raise ValueError("Matrice similarità non costruita")
        
        # Ottieni similarità con tutti gli altri
        similarities = self.similarity_matrix_[user_idx].copy()
        similarities[user_idx] = -1  # Escludi se stesso
        
        # Trova top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        user_indices = self.user_indices_ if self.user_indices_ is not None else []
        
        result = pd.DataFrame({
            'user_id': [user_indices[i] for i in top_indices],
            'similarity': similarities[top_indices]
        })
        
        return result
    
    def detect_anomalies(self, features: pd.DataFrame, threshold_percentile: float = 25) -> pd.DataFrame:
        """
        Rileva utenti anomali basandosi su bassa similarità media.
        
        Utenti con similarità media sotto il percentile specificato sono anomali.
        Questi utenti hanno comportamenti isolati, diversi dalla maggioranza.
        
        Args:
            features: DataFrame con feature normalizzate
            threshold_percentile: Percentile sotto cui considerare anomalie (default: 25)
        
        Returns:
            DataFrame con risultati anomaly detection:
                - user_id: ID utente
                - avg_similarity: Similarità media con altri utenti
                - cf_anomaly_score: Score anomalia (1 - avg_similarity)
                - is_anomaly: Flag booleano se sotto soglia
        """
        logger.info(f"=== Anomaly Detection via Collaborative Filtering ===")
        
        # Costruisci matrice se non esiste
        if self.similarity_matrix_ is None:
            self.build_similarity_matrix(features)
        
        # Calcola similarità medie
        avg_similarities = self.calculate_avg_similarity()
        
        # Calcola soglia dal percentile
        threshold = np.percentile(avg_similarities, threshold_percentile)
        logger.info(f"Soglia anomalia (percentile {threshold_percentile}): {threshold:.3f}")
        
        # Crea DataFrame risultati
        results = pd.DataFrame({
            'user_id': self.user_indices_,
            'avg_similarity': avg_similarities,
            'cf_anomaly_score': 1 - avg_similarities,  # Score: 1 - similarità
            'is_anomaly': avg_similarities < threshold
        })
        
        # Statistiche
        n_anomalies = results['is_anomaly'].sum()
        logger.info(f"Utenti anomali rilevati: {n_anomalies}/{len(results)} ({n_anomalies/len(results)*100:.1f}%)")
        logger.info(f"Similarità media globale: {avg_similarities.mean():.3f}")
        logger.info(f"Anomaly score medio: {results['cf_anomaly_score'].mean():.3f}")
        
        # Ordina per anomaly score decrescente
        results = results.sort_values('cf_anomaly_score', ascending=False).reset_index(drop=True)
        
        return results
    
    def get_similarity_statistics(self) -> pd.DataFrame:
        """
        Calcola statistiche sulla matrice di similarità.
        
        Returns:
            DataFrame con statistiche aggregate
        """
        if self.similarity_matrix_ is None:
            raise ValueError("Matrice similarità non costruita")
        
        # Estrai valori escludendo diagonale
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
        
        logger.info(f"\nStatistiche similarità:\n{stats}")
        
        return stats


def detect_anomalies_cf(features: pd.DataFrame, 
                        threshold_percentile: float = 25,
                        show_top: int = 10) -> Tuple[pd.DataFrame, CollaborativeFilter]:
    """
    Funzione helper per anomaly detection via CF con una chiamata.
    
    Args:
        features: DataFrame con feature normalizzate (righe=utenti, colonne=feature)
        threshold_percentile: Percentile sotto cui considerare anomalie
        show_top: Numero di top anomalie da mostrare nel log
    
    Returns:
        Tupla (results_df, cf_detector):
            - results_df: DataFrame con anomaly scores e flags
            - cf_detector: Oggetto CollaborativeFilter per ulteriori analisi
    """
    logger.info("=== Avvio Collaborative Filtering Anomaly Detection ===")
    
    # Inizializza detector
    cf_detector = CollaborativeFilter()
    
    # Esegui detection
    results = cf_detector.detect_anomalies(features, threshold_percentile)
    
    # Mostra statistiche
    cf_detector.get_similarity_statistics()
    
    # Mostra top anomalie
    if show_top > 0:
        logger.info(f"\nTop {show_top} anomalie CF:")
        top_anomalies = results.head(show_top)[['user_id', 'avg_similarity', 'cf_anomaly_score']]
        for i, (idx, row) in enumerate(top_anomalies.iterrows(), 1):
            logger.info(f"  {i}. User {row['user_id']}: "
                       f"similarity={row['avg_similarity']:.3f}, "
                       f"anomaly_score={row['cf_anomaly_score']:.3f}")
    
    return results, cf_detector
