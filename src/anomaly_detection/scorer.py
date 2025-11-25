"""Combina score da clustering, CF e graph ranking"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyScorer:
    """Combina score multipli con weighted average"""
    
    def __init__(self, weight_cluster=0.3, weight_cf=0.4, weight_graph=0.3):
        # auto-normalize weights
        total = weight_cluster + weight_cf + weight_graph
        self.weight_cluster = weight_cluster / total
        self.weight_cf = weight_cf / total
        self.weight_graph = weight_graph / total
        
        logger.info(f"Weights: cluster={self.weight_cluster:.2f}, cf={self.weight_cf:.2f}, graph={self.weight_graph:.2f}")
        self.combined_scores_ = None
        
    def normalize_scores(self, scores):
        """Min-max normalization [0,1]"""
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score < 1e-10:
            return np.zeros_like(scores)
        
        return (scores - min_score) / (max_score - min_score)
    
    def combine_scores(self, cluster_distances, cf_scores, graph_scores, user_ids):
        """Combina i 3 score in uno finale"""
        cluster_arr = np.array(cluster_distances)
        cf_arr = np.array(cf_scores)
        graph_arr = np.array(graph_scores)
        
        # normalizza tutto in [0,1]
        cluster_norm = self.normalize_scores(cluster_arr)
        cf_norm = self.normalize_scores(cf_arr)
        graph_norm = self.normalize_scores(graph_arr)
        
        # weighted average
        final_score = (
            self.weight_cluster * cluster_norm +
            self.weight_cf * cf_norm +
            self.weight_graph * graph_norm
        )
        
        results = pd.DataFrame({
            'user_id': user_ids,
            'cluster_distance_norm': cluster_norm,
            'cf_score_norm': cf_norm,
            'graph_score_norm': graph_norm,
            'final_anomaly_score': final_score
        })
        
        results = results.sort_values('final_anomaly_score', ascending=False)
        results['rank'] = range(1, len(results) + 1)
        results = results.reset_index(drop=True)
        
        self.combined_scores_ = results
        logger.info(f"Combined {len(results)} scores, top score={results['final_anomaly_score'].max():.3f}")
        
        return results
    
    def get_top_anomalies(self, top_k=20):
        """Top-k utenti più anomali"""
        if self.combined_scores_ is None:
            raise ValueError("Call combine_scores() first")
        return self.combined_scores_.head(top_k)
    
    def get_anomalies_by_threshold(self, percentile=90):
        """Utenti sopra percentile soglia"""
        if self.combined_scores_ is None:
            raise ValueError("Call combine_scores() first")
        
        threshold = np.percentile(self.combined_scores_['final_anomaly_score'], percentile)
        anomalies = self.combined_scores_[
            self.combined_scores_['final_anomaly_score'] >= threshold
        ].copy()
        
        logger.info(f"Threshold p{percentile}={threshold:.3f}: {len(anomalies)} users")
        return anomalies
    
    def export_results(self, output_path, top_k=None):
        """Esporta risultati in CSV"""
        if self.combined_scores_ is None:
            raise ValueError("Call combine_scores() first")
        
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
    
    def print_summary(self, top_k=10):
        """Stampa sommario risultati"""
        if self.combined_scores_ is None:
            raise ValueError("Score non combinati")
        
        print(f"\nUtenti analizzati: {len(self.combined_scores_)}")
        print(f"Pesi: cluster={self.weight_cluster:.2f}, cf={self.weight_cf:.2f}, graph={self.weight_graph:.2f}")
        
        scores = self.combined_scores_['final_anomaly_score']
        print(f"Score: mean={scores.mean():.3f}, median={scores.median():.3f}, std={scores.std():.3f}")
        
        print(f"\nTop {top_k} anomalie:")
        top = self.combined_scores_.head(top_k)
        for idx, row in top.iterrows():
            print(f"  #{row['rank']} {row['user_id']}: {row['final_anomaly_score']:.4f} "
                  f"(c={row['cluster_distance_norm']:.2f}, cf={row['cf_score_norm']:.2f}, g={row['graph_score_norm']:.2f})")


def combine_anomaly_scores(cluster_distances, cf_scores, graph_scores, user_ids,
                           weight_cluster=0.3, weight_cf=0.4, weight_graph=0.3,
                           output_path=None, show_summary=True):
    """Helper per combinare score in una chiamata"""
    scorer = AnomalyScorer(weight_cluster, weight_cf, weight_graph)
    results = scorer.combine_scores(cluster_distances, cf_scores, graph_scores, user_ids)
    
    if output_path:
        scorer.export_results(output_path)
    if show_summary:
        scorer.print_summary()
    
    return results, scorer
