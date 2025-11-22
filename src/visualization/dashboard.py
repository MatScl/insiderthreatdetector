"""
Dashboard Semplificata - Insider Threat Detection
==================================================
Carica anomalies.csv e mostra grafici automaticamente.

Usage: streamlit run src/visualization/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.visualization.plots import (
    plot_top_anomalies_bar,
    plot_score_distribution,
    plot_cluster_distribution,
    plot_scatter_clusters,
    plot_correlation_heatmap,
    plot_anomaly_threshold
)

st.set_page_config(page_title="Insider Threat Detection", layout="wide")


def main():
    st.title("Insider Threat Detection - Dashboard")
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Carica anomalies.csv",
        type=['csv'],
        help="File generato da: python main.py --find-optimal-k"
    )
    
    if uploaded_file is None:
        st.info("""
        ### Carica il file anomalies.csv per visualizzare i risultati
        
        **Come generare il file:**
        ```bash
        python main.py --find-optimal-k --top-k 20
        ```
        Questo crea `data/results/anomalies.csv`
        """)
        return
    
    # Carica dati
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✓ {len(df)} utenti caricati")
    except Exception as e:
        st.error(f"Errore: {e}")
        return
    
    # Verifica colonne
    if 'final_anomaly_score' not in df.columns:
        st.error("Il file deve contenere 'final_anomaly_score'")
        return
    
    # ===== METRICHE =====
    st.markdown("## Panoramica")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Utenti", len(df))
    col2.metric("Score Medio", f"{df['final_anomaly_score'].mean():.3f}")
    col3.metric("Score Max", f"{df['final_anomaly_score'].max():.3f}")
    col4.metric("Anomalie (>90%)", 
                (df['final_anomaly_score'] >= np.percentile(df['final_anomaly_score'], 90)).sum())
    
    st.markdown("---")
    
    # ===== TOP ANOMALIE =====
    st.markdown("## Top 20 Utenti Più Anomali")
    fig = plot_top_anomalies_bar(df, top_k=min(20, len(df)))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ===== DISTRIBUZIONE =====
    st.markdown("## Distribuzione Score")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plot_score_distribution(df, 'final_anomaly_score', 'Distribuzione Score Finale')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = plot_anomaly_threshold(df, 'final_anomaly_score', percentile=90)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ===== CLUSTER (se disponibile) =====
    if 'cluster' in df.columns:
        st.markdown("## Analisi Cluster")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plot_cluster_distribution(np.array(df['cluster']))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'cluster_distance' in df.columns and 'cf_anomaly_score' in df.columns:
                fig = plot_scatter_clusters(df, 'cluster_distance', 'cf_anomaly_score', 
                                           'cluster', 'Cluster Distance vs CF Score')
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
    
    # ===== CORRELAZIONI =====
    score_cols = [c for c in ['cluster_distance', 'cf_anomaly_score', 'pagerank_score', 
                               'degree_centrality', 'final_anomaly_score'] if c in df.columns]
    
    if len(score_cols) >= 2:
        st.markdown("## Correlazione Score")
        fig = plot_correlation_heatmap(df, score_cols)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
    
    # ===== TABELLA =====
    st.markdown("## Top Anomalie - Dettaglio")
    
    top = df.nlargest(20, 'final_anomaly_score')
    cols = ['user_id', 'final_anomaly_score', 'rank'] + \
           [c for c in ['cluster', 'cluster_distance', 'cf_anomaly_score', 'pagerank_score'] 
            if c in df.columns]
    
    display = top[cols].copy()
    for col in display.select_dtypes(include=['float64', 'float32']).columns:
        display[col] = display[col].round(4)
    
    st.dataframe(display, use_container_width=True, height=400)
    
    # Download
    st.download_button(
        "Scarica CSV",
        display.to_csv(index=False),
        "top_anomalies.csv",
        "text/csv"
    )


if __name__ == "__main__":
    main()
