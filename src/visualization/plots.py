"""
Plotting Utilities
==================
Funzioni per visualizzazioni Plotly.

Author: Matteo Sclafani
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, List


def plot_top_anomalies_bar(results: pd.DataFrame, 
                           top_k: int = 20,
                           title: str = "Top Utenti Anomali") -> go.Figure:
    """Bar chart top utenti anomali."""
    top_data = results.head(top_k).copy().iloc[::-1]
    
    fig = go.Figure(data=[
        go.Bar(
            y=top_data['user_id'].astype(str),
            x=top_data['final_anomaly_score'],
            orientation='h',
            marker=dict(color=top_data['final_anomaly_score'], colorscale='Reds')
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Anomaly Score",
        yaxis_title="User ID",
        height=max(400, top_k * 20)
    )
    return fig


def plot_score_distribution(results: pd.DataFrame,
                            score_col: str = 'final_anomaly_score',
                            title: str = "Distribuzione Score") -> go.Figure:
    """Istogramma distribuzione score."""
    fig = go.Figure(data=[
        go.Histogram(x=results[score_col], nbinsx=50, marker_color='steelblue')
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title=score_col,
        yaxis_title="Frequenza"
    )
    return fig


def plot_cluster_distribution(cluster_labels: np.ndarray,
                              title: str = "Distribuzione Cluster") -> go.Figure:
    """Bar chart distribuzione cluster."""
    unique, counts = np.unique(cluster_labels, return_counts=True)
    
    fig = go.Figure(data=[
        go.Bar(x=unique, y=counts, marker_color='lightseagreen')
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Cluster ID",
        yaxis_title="Numero Utenti"
    )
    return fig


def plot_correlation_heatmap(data: pd.DataFrame,
                             columns: Optional[List[str]] = None,
                             title: str = "Matrice Correlazione") -> go.Figure:
    """Heatmap correlazioni."""
    if columns:
        corr_data = data[columns]
    else:
        corr_data = data.select_dtypes(include=[np.number])
    
    corr_matrix = corr_data.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(title=title, height=500)
    return fig


def plot_scatter_clusters(data: pd.DataFrame,
                          x_col: str,
                          y_col: str,
                          cluster_col: str = 'cluster',
                          title: str = "Scatter Plot Cluster") -> go.Figure:
    """Scatter plot colorato per cluster."""
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color=cluster_col,
        title=title,
        labels={x_col: x_col.replace('_', ' ').title(),
                y_col: y_col.replace('_', ' ').title()}
    )
    return fig


def plot_anomaly_threshold(results: pd.DataFrame,
                           score_col: str = 'final_anomaly_score',
                           percentile: float = 90,
                           title: str = "Score vs Soglia Anomalie") -> go.Figure:
    """Scatter plot con linea soglia."""
    threshold = np.percentile(results[score_col], percentile)
    is_anomaly = results[score_col] >= threshold
    
    fig = go.Figure()
    
    # Normali
    fig.add_trace(go.Scatter(
        x=results[~is_anomaly].index,
        y=results[~is_anomaly][score_col],
        mode='markers',
        name='Normale',
        marker=dict(color='lightblue', size=4)
    ))
    
    # Anomalie
    fig.add_trace(go.Scatter(
        x=results[is_anomaly].index,
        y=results[is_anomaly][score_col],
        mode='markers',
        name='Anomalia',
        marker=dict(color='red', size=6)
    ))
    
    # Soglia
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Soglia {percentile}% = {threshold:.3f}"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Indice Utente",
        yaxis_title=score_col
    )
    return fig
