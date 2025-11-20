"""
Graph-based Anomaly Detection with PageRank
============================================

Questo modulo implementa anomaly detection basata su analisi del grafo.
Costruisce un grafo di similarità tra utenti e usa PageRank e degree centrality
per identificare utenti isolati o periferici nella rete comportamentale.

Author: Matteo Sclafani
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Tuple, Optional
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphRanker:
    """
    Detector di anomalie basato su analisi del grafo di similarità.
    
    Costruisce un grafo non diretto dove:
    - Nodi = utenti
    - Archi = similarità sopra soglia
    - Peso archi = valore similarità
    
    Usa PageRank e degree centrality per trovare utenti isolati.
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Inizializza il graph ranker.
        
        Args:
            similarity_threshold: Soglia minima similarità per creare arco (default: 0.7)
        """
        self.similarity_threshold = similarity_threshold
        self.graph = None
        self.pagerank_ = None
        self.degree_centrality_ = None
        
    def build_graph(self, similarity_matrix: np.ndarray, user_ids: list) -> nx.Graph:
        """
        Costruisce grafo di similarità da matrice.
        
        Args:
            similarity_matrix: Matrice NxN di similarità tra utenti
            user_ids: Lista di ID utenti corrispondenti alle righe/colonne
        
        Returns:
            NetworkX Graph object
        """
        logger.info(f"Costruzione grafo con soglia similarità: {self.similarity_threshold}")
        
        # Crea grafo non diretto
        self.graph = nx.Graph()
        
        # Aggiungi nodi (utenti)
        self.graph.add_nodes_from(user_ids)
        
        # Aggiungi archi per similarità sopra soglia
        n_users = len(user_ids)
        n_edges = 0
        
        for i in range(n_users):
            for j in range(i + 1, n_users):  # Solo metà superiore (grafo non diretto)
                similarity = similarity_matrix[i, j]
                
                if similarity >= self.similarity_threshold:
                    # Aggiungi arco con peso = similarità
                    self.graph.add_edge(user_ids[i], user_ids[j], weight=similarity)
                    n_edges += 1
        
        logger.info(f"Grafo costruito: {n_users} nodi, {n_edges} archi")
        logger.info(f"Densità grafo: {nx.density(self.graph):.3f}")
        
        # Identifica componenti connesse
        components = list(nx.connected_components(self.graph))
        logger.info(f"Componenti connesse: {len(components)}")
        
        if len(components) > 1:
            logger.warning(f"ATTENZIONE: Grafo non completamente connesso!")
            sizes = [len(c) for c in components]
            logger.info(f"Dimensioni componenti: {sizes}")
        
        return self.graph
    
    def calculate_pagerank(self, alpha: float = 0.85, max_iter: int = 100) -> dict:
        """
        Calcola PageRank per ogni nodo del grafo.
        
        PageRank misura l'importanza di un nodo nella rete.
        Utenti con basso PageRank sono isolati/periferici (potenziali anomalie).
        
        Args:
            alpha: Damping factor per PageRank (default: 0.85)
            max_iter: Massimo numero iterazioni (default: 100)
        
        Returns:
            Dictionary {user_id: pagerank_score}
        """
        if self.graph is None:
            raise ValueError("Grafo non costruito. Chiamare build_graph() prima")
        
        logger.info("Calcolo PageRank...")
        
        # Calcola PageRank con peso degli archi
        self.pagerank_ = nx.pagerank(
            self.graph,
            alpha=alpha,
            max_iter=max_iter,
            weight='weight'
        )
        
        # Statistiche
        scores = list(self.pagerank_.values())
        logger.info(f"PageRank completato:")
        logger.info(f"  Media: {np.mean(scores):.6f}")
        logger.info(f"  Std: {np.std(scores):.6f}")
        logger.info(f"  Min: {np.min(scores):.6f}")
        logger.info(f"  Max: {np.max(scores):.6f}")
        
        return self.pagerank_
    
    def calculate_degree_centrality(self) -> dict:
        """
        Calcola degree centrality per ogni nodo.
        
        Degree centrality misura il numero di connessioni di un nodo.
        Utenti con bassa degree centrality hanno poche connessioni (isolati).
        
        Returns:
            Dictionary {user_id: degree_centrality}
        """
        if self.graph is None:
            raise ValueError("Grafo non costruito. Chiamare build_graph() prima")
        
        logger.info("Calcolo Degree Centrality...")
        
        # Calcola degree centrality
        self.degree_centrality_ = nx.degree_centrality(self.graph)
        
        # Statistiche
        scores = list(self.degree_centrality_.values())
        logger.info(f"Degree Centrality completata:")
        logger.info(f"  Media: {np.mean(scores):.3f}")
        logger.info(f"  Std: {np.std(scores):.3f}")
        logger.info(f"  Min: {np.min(scores):.3f}")
        logger.info(f"  Max: {np.max(scores):.3f}")
        
        return self.degree_centrality_
    
    def get_isolated_nodes(self, top_k: int = 10) -> pd.DataFrame:
        """
        Trova nodi più isolati (basso degree).
        
        Args:
            top_k: Numero di nodi isolati da restituire
        
        Returns:
            DataFrame con nodi ordinati per degree crescente
        """
        if self.degree_centrality_ is None:
            self.calculate_degree_centrality()
        
        # Converti in DataFrame e ordina
        degree_dict = self.degree_centrality_ if self.degree_centrality_ is not None else {}
        df = pd.DataFrame({
            'user_id': list(degree_dict.keys()),
            'degree_centrality': list(degree_dict.values())
        })
        
        df = df.sort_values('degree_centrality').head(top_k).reset_index(drop=True)
        
        logger.info(f"\nTop {top_k} nodi più isolati:")
        for i, (idx, row) in enumerate(df.iterrows(), 1):
            logger.info(f"  {i}. User {row['user_id']}: degree={row['degree_centrality']:.3f}")
        
        return df
    
    def get_anomaly_scores(self, user_ids: list) -> pd.DataFrame:
        """
        Calcola anomaly scores basati su metriche del grafo.
        
        Combina PageRank e Degree Centrality:
        - Basso PageRank → alto anomaly score
        - Basso Degree → alto anomaly score
        
        Args:
            user_ids: Lista di ID utenti da valutare
        
        Returns:
            DataFrame con scores:
                - user_id
                - pagerank
                - degree_centrality
                - graph_anomaly_score (media normalizzata inversa)
        """
        logger.info("=== Calcolo Graph Anomaly Scores ===")
        
        # Calcola metriche se non presenti
        if self.pagerank_ is None:
            self.calculate_pagerank()
        if self.degree_centrality_ is None:
            self.calculate_degree_centrality()
        
        # Estrai scores
        pr_dict = self.pagerank_ if self.pagerank_ is not None else {}
        deg_dict = self.degree_centrality_ if self.degree_centrality_ is not None else {}
        
        pagerank_scores = np.array([pr_dict.get(uid, 0) for uid in user_ids])
        degree_scores = np.array([deg_dict.get(uid, 0) for uid in user_ids])
        
        # Normalizza scores in [0, 1]
        pr_min, pr_max = pagerank_scores.min(), pagerank_scores.max()
        deg_min, deg_max = degree_scores.min(), degree_scores.max()
        
        pr_norm = (pagerank_scores - pr_min) / (pr_max - pr_min + 1e-10)
        deg_norm = (degree_scores - deg_min) / (deg_max - deg_min + 1e-10)
        
        # Anomaly score: inverti (basso PageRank/Degree = alta anomalia)
        # Media di (1 - pagerank_norm) e (1 - degree_norm)
        graph_anomaly_score = ((1 - pr_norm) + (1 - deg_norm)) / 2
        
        # Crea DataFrame
        results = pd.DataFrame({
            'user_id': user_ids,
            'pagerank': pagerank_scores,
            'degree_centrality': degree_scores,
            'graph_anomaly_score': graph_anomaly_score
        })
        
        # Ordina per anomaly score decrescente
        results = results.sort_values('graph_anomaly_score', ascending=False).reset_index(drop=True)
        
        logger.info(f"Graph anomaly score medio: {graph_anomaly_score.mean():.3f}")
        logger.info(f"Graph anomaly score max: {graph_anomaly_score.max():.3f}")
        
        return results
    
    def get_graph_statistics(self) -> dict:
        """
        Calcola statistiche sul grafo.
        
        Returns:
            Dictionary con metriche del grafo
        """
        if self.graph is None:
            raise ValueError("Grafo non costruito")
        
        # Calcola average degree manualmente
        total_degree = sum(d for n, d in self.graph.degree())  # type: ignore
        n_nodes = self.graph.number_of_nodes()
        
        stats = {
            'n_nodes': n_nodes,
            'n_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'n_components': nx.number_connected_components(self.graph),
            'avg_clustering': nx.average_clustering(self.graph),
            'avg_degree': total_degree / n_nodes if n_nodes > 0 else 0
        }
        
        logger.info("\nStatistiche Grafo:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
        
        return stats


def detect_anomalies_graph(similarity_matrix: np.ndarray,
                          user_ids: list,
                          similarity_threshold: float = 0.7,
                          show_top: int = 10) -> Tuple[pd.DataFrame, GraphRanker]:
    """
    Funzione helper per anomaly detection via graph ranking con una chiamata.
    
    Args:
        similarity_matrix: Matrice NxN di similarità tra utenti
        user_ids: Lista di ID utenti
        similarity_threshold: Soglia per creare archi nel grafo
        show_top: Numero di top anomalie da mostrare
    
    Returns:
        Tupla (results_df, graph_ranker):
            - results_df: DataFrame con graph anomaly scores
            - graph_ranker: Oggetto GraphRanker per ulteriori analisi
    """
    logger.info("=== Avvio Graph-based Anomaly Detection ===")
    
    # Inizializza ranker
    ranker = GraphRanker(similarity_threshold=similarity_threshold)
    
    # Costruisci grafo
    ranker.build_graph(similarity_matrix, user_ids)
    
    # Calcola anomaly scores
    results = ranker.get_anomaly_scores(user_ids)
    
    # Statistiche grafo
    ranker.get_graph_statistics()
    
    # Mostra top anomalie
    if show_top > 0:
        logger.info(f"\nTop {show_top} anomalie Graph:")
        top_anomalies = results.head(show_top)
        for i, (idx, row) in enumerate(top_anomalies.iterrows(), 1):
            logger.info(f"  {i}. User {row['user_id']}: "
                       f"PageRank={row['pagerank']:.6f}, "
                       f"Degree={row['degree_centrality']:.3f}, "
                       f"Anomaly={row['graph_anomaly_score']:.3f}")
    
    return results, ranker
