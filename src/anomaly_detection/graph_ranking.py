"""Graph-based anomaly detection con PageRank e degree centrality"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphRanker:
    """Rileva anomalie via analisi grafo di similarità"""
    
    def __init__(self, similarity_threshold=0.7):
        self.similarity_threshold = similarity_threshold
        self.graph = None
        self.pagerank_ = None
        self.degree_centrality_ = None
        
    def build_graph(self, similarity_matrix, user_ids):
        """Costruisce grafo da matrice similarità"""
        self.graph = nx.Graph()
        self.graph.add_nodes_from(user_ids)
        
        n_users = len(user_ids)
        n_edges = 0
        
        for i in range(n_users):
            for j in range(i + 1, n_users):
                similarity = similarity_matrix[i, j]
                
                if similarity >= self.similarity_threshold:
                    self.graph.add_edge(user_ids[i], user_ids[j], weight=similarity)
                    n_edges += 1
        
        logger.info(f"Graph: {n_users} nodes, {n_edges} edges, density={nx.density(self.graph):.3f}")
        
        components = list(nx.connected_components(self.graph))
        if len(components) > 1:
            logger.warning(f"Graph not fully connected: {len(components)} components")
        
        return self.graph
    
    def calculate_pagerank(self, alpha=0.85, max_iter=100):
        """Calcola PageRank (basso = isolato/anomalo)"""
        if self.graph is None:
            raise ValueError("Call build_graph() first")
        
        self.pagerank_ = nx.pagerank(self.graph, alpha=alpha, max_iter=max_iter, weight='weight')
        
        scores = list(self.pagerank_.values())
        logger.info(f"PageRank: mean={np.mean(scores):.6f}, std={np.std(scores):.6f}")
        
        return self.pagerank_
    
    def calculate_degree_centrality(self):
        """Calcola degree centrality (basso = poche connessioni)"""
        if self.graph is None:
            raise ValueError("Call build_graph() first")
        
        self.degree_centrality_ = nx.degree_centrality(self.graph)
        
        scores = list(self.degree_centrality_.values())
        logger.info(f"Degree: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")
        
        return self.degree_centrality_
    
    def get_isolated_nodes(self, top_k=10):
        """Trova nodi più isolati"""
        if self.degree_centrality_ is None:
            self.calculate_degree_centrality()
        
        degree_dict = self.degree_centrality_ if self.degree_centrality_ is not None else {}
        df = pd.DataFrame({
            'user_id': list(degree_dict.keys()),
            'degree_centrality': list(degree_dict.values())
        })
        
        df = df.sort_values('degree_centrality').head(top_k).reset_index(drop=True)
        return df
    
    def get_anomaly_scores(self, user_ids):
        """Combina PageRank e Degree in anomaly score"""
        if self.pagerank_ is None:
            self.calculate_pagerank()
        if self.degree_centrality_ is None:
            self.calculate_degree_centrality()
        
        pr_dict = self.pagerank_ if self.pagerank_ is not None else {}
        deg_dict = self.degree_centrality_ if self.degree_centrality_ is not None else {}
        
        pagerank_scores = np.array([pr_dict.get(uid, 0) for uid in user_ids])
        degree_scores = np.array([deg_dict.get(uid, 0) for uid in user_ids])
        
        # normalizza in [0,1]
        pr_min, pr_max = pagerank_scores.min(), pagerank_scores.max()
        deg_min, deg_max = degree_scores.min(), degree_scores.max()
        
        pr_norm = (pagerank_scores - pr_min) / (pr_max - pr_min + 1e-10)
        deg_norm = (degree_scores - deg_min) / (deg_max - deg_min + 1e-10)
        
        # FIXME: maybe use different weights for PR vs Degree?
        graph_anomaly_score = ((1 - pr_norm) + (1 - deg_norm)) / 2
        
        results = pd.DataFrame({
            'user_id': user_ids,
            'pagerank': pagerank_scores,
            'degree_centrality': degree_scores,
            'graph_anomaly_score': graph_anomaly_score
        })
        
        results = results.sort_values('graph_anomaly_score', ascending=False).reset_index(drop=True)
        logger.info(f"Graph score: mean={graph_anomaly_score.mean():.3f}, max={graph_anomaly_score.max():.3f}")
        
        return results
    
    def get_graph_statistics(self):
        """Stats sul grafo"""
        if self.graph is None:
            raise ValueError("Call build_graph() first")
        
        total_degree = sum(d for n, d in self.graph.degree())  # type: ignore
        n_nodes = self.graph.number_of_nodes()
        
        stats = {
            'n_nodes': n_nodes,
            'n_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'n_components': nx.number_connected_components(self.graph),
            'avg_degree': total_degree / n_nodes if n_nodes > 0 else 0
        }
        
        return stats


def detect_anomalies_graph(similarity_matrix, user_ids, similarity_threshold=0.7, show_top=10):
    """Helper per graph-based anomaly detection"""
    ranker = GraphRanker(similarity_threshold=similarity_threshold)
    ranker.build_graph(similarity_matrix, user_ids)
    results = ranker.get_anomaly_scores(user_ids)
    
    ranker.get_graph_statistics()
    
    if show_top > 0:
        top_anomalies = results.head(show_top)
        for i, (idx, row) in enumerate(top_anomalies.iterrows(), 1):
            logger.info(f"  {i}. {row['user_id']}: PR={row['pagerank']:.6f}, deg={row['degree_centrality']:.3f}")
    
    return results, ranker

