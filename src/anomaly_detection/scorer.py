"""
Anomaly Score Combiner
======================

Questo modulo combina gli score di anomalia da diverse tecniche:
- Clustering (distanza da centroide)
- Collaborative Filtering (bassa similarità)
- Graph Ranking (isolamento nella rete)

Produce uno score finale pesato e ranking degli utenti anomali.

Author: Matteo Sclafani
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyScorer:
    """
    Combina score da multiple tecniche di anomaly detection.
    
    Normalizza e combina gli score con pesi configurabili per produrre
    un ranking finale degli utenti più sospetti.
    """
    
    def __init__(self, 
                 weight_cluster: float = 0.3,
                 weight_cf: float = 0.4,
                 weight_graph: float = 0.3):
        """
        Inizializza il combiner con pesi per ogni tecnica.
        
        Args:
            weight_cluster: Peso per cluster distance (default: 0.3)
            weight_cf: Peso per CF anomaly score (default: 0.4)
            weight_graph: Peso per graph anomaly score (default: 0.3)
        """
        # Normalizza pesi a somma = 1
        total = weight_cluster + weight_cf + weight_graph
        self.weight_cluster = weight_cluster / total
        self.weight_cf = weight_cf / total
        self.weight_graph = weight_graph / total
        
        logger.info(f"Pesi normalizzati: Cluster={self.weight_cluster:.2f}, "
                   f"CF={self.weight_cf:.2f}, Graph={self.weight_graph:.2f}")
        
        self.combined_scores_ = None
        
    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalizza scores in range [0, 1] usando min-max scaling.
        
        Args:
            scores: Array di scores da normalizzare
        
        Returns:
            Array normalizzato in [0, 1]
        """
        min_score = scores.min()
        max_score = scores.max()
        
        # Evita divisione per zero
        if max_score - min_score < 1e-10:
            return np.zeros_like(scores)
        
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized
    
    def combine_scores(self,
                      cluster_distances: pd.Series,
                      cf_scores: pd.Series,
                      graph_scores: pd.Series,
                      user_ids: pd.Series) -> pd.DataFrame:
        """
        Combina score dalle tre tecniche in un unico score finale.
        
        Args:
            cluster_distances: Serie con distanze dai centroidi
            cf_scores: Serie con CF anomaly scores
            graph_scores: Serie con graph anomaly scores
            user_ids: Serie con ID utenti
        
        Returns:
            DataFrame con scores combinati:
                - user_id
                - cluster_distance_norm
                - cf_score_norm
                - graph_score_norm
                - final_anomaly_score (weighted combination)
                - rank (1 = più anomalo)
        """
        logger.info("=== Combinazione Score Anomalie ===")
        
        # Converti a numpy per manipolazione
        cluster_arr = np.array(cluster_distances.values)
        cf_arr = np.array(cf_scores.values)
        graph_arr = np.array(graph_scores.values)
        
        # Normalizza tutti gli score in [0, 1]
        cluster_norm = self.normalize_scores(cluster_arr)
        cf_norm = self.normalize_scores(cf_arr)
        graph_norm = self.normalize_scores(graph_arr)
        
        logger.info(f"Score normalizzati:")
        logger.info(f"  Cluster: mean={cluster_norm.mean():.3f}, std={cluster_norm.std():.3f}")
        logger.info(f"  CF: mean={cf_norm.mean():.3f}, std={cf_norm.std():.3f}")
        logger.info(f"  Graph: mean={graph_norm.mean():.3f}, std={graph_norm.std():.3f}")
        
        # Combina con media pesata
        final_score = (
            self.weight_cluster * cluster_norm +
            self.weight_cf * cf_norm +
            self.weight_graph * graph_norm
        )
        
        logger.info(f"Score finale: mean={final_score.mean():.3f}, std={final_score.std():.3f}")
        
        # Crea DataFrame risultati
        results = pd.DataFrame({
            'user_id': user_ids,
            'cluster_distance_norm': cluster_norm,
            'cf_score_norm': cf_norm,
            'graph_score_norm': graph_norm,
            'final_anomaly_score': final_score
        })
        
        # Ordina per score decrescente e assegna rank
        results = results.sort_values('final_anomaly_score', ascending=False).reset_index(drop=True)
        results['rank'] = range(1, len(results) + 1)
        
        # Salva per riferimento
        self.combined_scores_ = results
        
        return results
    
    def get_top_anomalies(self, top_k: int = 20) -> pd.DataFrame:
        """
        Restituisce i top-k utenti più anomali.
        
        Args:
            top_k: Numero di utenti da restituire
        
        Returns:
            DataFrame con top anomalie
        """
        if self.combined_scores_ is None:
            raise ValueError("Score non combinati. Chiamare combine_scores() prima")
        
        return self.combined_scores_.head(top_k)
    
    def get_anomalies_by_threshold(self, percentile: float = 90) -> pd.DataFrame:
        """
        Restituisce utenti con score sopra un certo percentile.
        
        Args:
            percentile: Percentile soglia (default: 90 = top 10%)
        
        Returns:
            DataFrame con utenti anomali sopra soglia
        """
        if self.combined_scores_ is None:
            raise ValueError("Score non combinati")
        
        threshold = np.percentile(self.combined_scores_['final_anomaly_score'], percentile)
        logger.info(f"Soglia anomalia (percentile {percentile}): {threshold:.3f}")
        
        anomalies = self.combined_scores_[
            self.combined_scores_['final_anomaly_score'] >= threshold
        ].copy()
        
        logger.info(f"Utenti sopra soglia: {len(anomalies)}/{len(self.combined_scores_)} "
                   f"({len(anomalies)/len(self.combined_scores_)*100:.1f}%)")
        
        return anomalies
    
    def export_results(self, output_path: str, top_k: Optional[int] = None) -> None:
        """
        Esporta risultati in CSV.
        
        Args:
            output_path: Path del file CSV di output
            top_k: Se specificato, esporta solo top-k anomalie (altrimenti tutti)
        """
        if self.combined_scores_ is None:
            raise ValueError("Score non combinati")
        
        # Seleziona dati da esportare
        if top_k is not None:
            data_to_export = self.combined_scores_.head(top_k)
            logger.info(f"Esportazione top {top_k} anomalie in {output_path}")
        else:
            data_to_export = self.combined_scores_
            logger.info(f"Esportazione di tutti gli utenti in {output_path}")
        
        # Crea directory se non esiste
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Esporta
        data_to_export.to_csv(output_path, index=False)
        logger.info(f"Risultati salvati: {len(data_to_export)} righe")
    
    def print_summary(self, top_k: int = 10) -> None:
        """
        Stampa sommario dei risultati.
        
        Args:
            top_k: Numero di top anomalie da mostrare
        """
        if self.combined_scores_ is None:
            raise ValueError("Score non combinati")
        
        print("\n" + "="*60)
        print("SOMMARIO ANOMALY DETECTION")
        print("="*60)
        print(f"\nTotale utenti analizzati: {len(self.combined_scores_)}")
        print(f"\nPesi utilizzati:")
        print(f"  - Cluster Distance: {self.weight_cluster:.2f}")
        print(f"  - Collaborative Filtering: {self.weight_cf:.2f}")
        print(f"  - Graph Ranking: {self.weight_graph:.2f}")
        
        print(f"\nScore finale - Statistiche:")
        scores = self.combined_scores_['final_anomaly_score']
        print(f"  Media: {scores.mean():.3f}")
        print(f"  Mediana: {scores.median():.3f}")
        print(f"  Std Dev: {scores.std():.3f}")
        print(f"  Min: {scores.min():.3f}")
        print(f"  Max: {scores.max():.3f}")
        
        print(f"\n{'='*60}")
        print(f"TOP {top_k} UTENTI PIÙ ANOMALI")
        print("="*60)
        
        top = self.combined_scores_.head(top_k)
        for idx, row in top.iterrows():
            print(f"\n#{row['rank']} - User ID: {row['user_id']}")
            print(f"  Final Score: {row['final_anomaly_score']:.4f}")
            print(f"  Cluster: {row['cluster_distance_norm']:.3f} | "
                  f"CF: {row['cf_score_norm']:.3f} | "
                  f"Graph: {row['graph_score_norm']:.3f}")
        
        print("\n" + "="*60 + "\n")


def combine_anomaly_scores(cluster_distances: pd.Series,
                           cf_scores: pd.Series,
                           graph_scores: pd.Series,
                           user_ids: pd.Series,
                           weight_cluster: float = 0.3,
                           weight_cf: float = 0.4,
                           weight_graph: float = 0.3,
                           output_path: Optional[str] = None,
                           show_summary: bool = True) -> Tuple[pd.DataFrame, AnomalyScorer]:
    """
    Funzione helper per combinare score con una chiamata.
    
    Args:
        cluster_distances: Serie con distanze dai centroidi
        cf_scores: Serie con CF anomaly scores
        graph_scores: Serie con graph anomaly scores
        user_ids: Serie con ID utenti
        weight_cluster: Peso per cluster distance
        weight_cf: Peso per CF score
        weight_graph: Peso per graph score
        output_path: Path per export CSV (opzionale)
        show_summary: Se mostrare sommario testuale
    
    Returns:
        Tupla (results_df, scorer):
            - results_df: DataFrame con score combinati e ranking
            - scorer: Oggetto AnomalyScorer per ulteriori analisi
    """
    logger.info("=== Avvio Combinazione Score ===")
    
    # Inizializza scorer
    scorer = AnomalyScorer(
        weight_cluster=weight_cluster,
        weight_cf=weight_cf,
        weight_graph=weight_graph
    )
    
    # Combina score
    results = scorer.combine_scores(
        cluster_distances=cluster_distances,
        cf_scores=cf_scores,
        graph_scores=graph_scores,
        user_ids=user_ids
    )
    
    # Export se richiesto
    if output_path is not None:
        scorer.export_results(output_path)
    
    # Mostra sommario
    if show_summary:
        scorer.print_summary()
    
    return results, scorer
