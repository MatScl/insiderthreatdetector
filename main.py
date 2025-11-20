"""
Pipeline per insider threat detection:
1. Carica CSV
2. Normalizza
3. Clustering (K-Means)
4. Anomaly detection (CF + Graph)
5. Combina score
6. Export

Usage:
    python main.py --input data/raw/user_features.csv --output data/results/anomalies.csv
"""

import argparse
import logging
from pathlib import Path
import pandas as pd

from src.data_analysis.loader import DataLoader
from src.data_analysis.normalizer import normalize_features
from src.user_modeling.clustering import cluster_users
from src.anomaly_detection.collaborative_filtering import detect_anomalies_cf
from src.anomaly_detection.graph_ranking import detect_anomalies_graph
from src.anomaly_detection.scorer import combine_anomaly_scores

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('detection.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse argomenti CLI"""
    parser = argparse.ArgumentParser(description='Insider Threat Detection Pipeline')
    
    parser.add_argument('--input', type=str, default='data/raw/user_features.csv',
                       help='Path CSV features')
    parser.add_argument('--output', type=str, default='data/results/anomalies.csv',
                       help='Path output anomalie')
    parser.add_argument('--n-clusters', type=int, default=None,
                       help='Numero cluster (default: auto)')
    parser.add_argument('--find-optimal-k', action='store_true',
                       help='Cerca k ottimale')
    parser.add_argument('--similarity-threshold', type=float, default=0.7,
                       help='Soglia similarità grafo')
    parser.add_argument('--weight-cluster', type=float, default=0.3,
                       help='Peso cluster distance')
    parser.add_argument('--weight-cf', type=float, default=0.4,
                       help='Peso CF score')
    parser.add_argument('--weight-graph', type=float, default=0.3,
                       help='Peso graph score')
    parser.add_argument('--top-k', type=int, default=20,
                       help='Top anomalie da mostrare')
    parser.add_argument('--no-summary', action='store_true',
                       help='Non mostrare sommario')
    
    return parser.parse_args()


def run_pipeline(input_path, output_path, n_clusters=None, find_optimal_k=False,
                similarity_threshold=0.7, weight_cluster=0.3, weight_cf=0.4,
                weight_graph=0.3, top_k=20, show_summary=True):
    """Esegue pipeline completa"""
    # TODO: add progress bar maybe?
    logger.info("Avvio pipeline")
    
    # Step 1: Carica dati
    loader = DataLoader(input_path)
    data = loader.load()
    feature_columns = loader.feature_names
    if not feature_columns:
        logger.error("Nessuna feature numerica trovata")
        return None
    
    if data is None:
        logger.error("Errore caricamento dati")
        return None
    
    logger.info(f"Caricati {len(data)} utenti, {len(data.columns)} feature")
    
    # Step 2: Normalizza
    normalized_data = normalize_features(data, feature_columns=feature_columns)
    normalized_data.set_index('user_id', inplace=True)
    feature_matrix = normalized_data[feature_columns].copy()
    
    # Step 3: Clustering
    cluster_labels, cluster_distances, clusterer = cluster_users(
        feature_matrix, n_clusters=n_clusters, find_optimal=find_optimal_k
    )
    normalized_data['cluster'] = cluster_labels
    normalized_data['cluster_distance'] = cluster_distances
    logger.info(f"Clustering: {clusterer.n_clusters} cluster")
    
    # Step 4: CF
    cf_results, cf_detector = detect_anomalies_cf(feature_matrix, show_top=0)
    
    # Step 5: Graph ranking
    if cf_detector.similarity_matrix_ is None:
        logger.error("Matrice similarità mancante")
        return None
    
    graph_results, graph_ranker = detect_anomalies_graph(
        similarity_matrix=cf_detector.similarity_matrix_,
        user_ids=feature_matrix.index.tolist(),
        similarity_threshold=similarity_threshold,
        show_top=0
    )
    
    # Step 6: Combina score
    cluster_dist_series = pd.Series(cluster_distances, index=feature_matrix.index)
    cf_score_series = cf_results.set_index('user_id')['cf_anomaly_score']
    graph_score_series = graph_results.set_index('user_id')['graph_anomaly_score']
    user_ids_series = pd.Series(feature_matrix.index, name='user_id')
    
    final_results, scorer = combine_anomaly_scores(
        cluster_dist_series, cf_score_series, graph_score_series, user_ids_series,
        weight_cluster, weight_cf, weight_graph,
        output_path=output_path, show_summary=show_summary
    )
    
    # Aggiungi cluster ai risultati
    cluster_info = normalized_data[['cluster']].reset_index()
    cluster_info.columns = ['user_id', 'cluster']
    final_results = final_results.merge(cluster_info, on='user_id', how='left')
    
    # Salva versione estesa
    extended_output = output_path.replace('.csv', '_extended.csv')
    normalized_data_with_scores = normalized_data.copy()
    normalized_data_with_scores['final_anomaly_score'] = final_results.set_index('user_id')['final_anomaly_score']
    normalized_data_with_scores['rank'] = final_results.set_index('user_id')['rank']
    normalized_data_with_scores.reset_index().to_csv(extended_output, index=False)
    logger.info(f"Risultati estesi: {extended_output}")
    
    # Top anomalie
    if top_k > 0:
        logger.info(f"\nTop {top_k} anomalie:")
        top_anomalies = final_results.head(top_k)
        for idx, row in top_anomalies.iterrows():
            logger.info(f"#{row['rank']} {row['user_id']} (cluster {row.get('cluster', 'N/A')}): "
                       f"score={row['final_anomaly_score']:.4f}")
    
    logger.info("Pipeline completata")
    logger.info(f"Risultati: {output_path}")
    
    return final_results


def main_pipeline():
    """Entry point"""
    args = parse_arguments()
    
    if not Path(args.input).exists():
        logger.error(f"File non trovato: {args.input}")
        return
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    results = run_pipeline(
        args.input, args.output, args.n_clusters, args.find_optimal_k,
        args.similarity_threshold, args.weight_cluster, args.weight_cf,
        args.weight_graph, args.top_k, not args.no_summary
    )
    
    if results is not None:
        logger.info("Dashboard: streamlit run src/visualization/dashboard.py")


if __name__ == "__main__":
    main_pipeline()
