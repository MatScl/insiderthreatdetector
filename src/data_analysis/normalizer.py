"""
Feature Normalizer per Insider Threat Detection

Normalizza le feature numeriche usando StandardScaler (z-score).
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureNormalizer:
    """Normalizza feature comportamentali (mean=0, std=1)"""

    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_columns = feature_columns
        
    def _resolve_feature_columns(self, df: pd.DataFrame) -> List[str]:
        if self.feature_columns is not None:
            missing = set(self.feature_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")
            return list(self.feature_columns)

        numeric_cols = df.select_dtypes(include=['number', 'bool']).columns.tolist()
        inferred = [col for col in numeric_cols if col != 'user_id']
        if not inferred:
            raise ValueError("No numeric columns available for normalization.")
        self.feature_columns = inferred
        return inferred
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizza feature e restituisce nuovo DataFrame.
        
        Args:
            df: DataFrame con colonne user_id + FEATURE_COLUMNS
            
        Returns:
            DataFrame normalizzato (user_id + feature normalizzate)
        """
        feature_columns = self._resolve_feature_columns(df)
        
        # Estrai user_id
        user_ids = df['user_id'].copy()
        
        # Normalizza solo le feature numeriche
        features = df[feature_columns].values
        normalized = self.scaler.fit_transform(features)
        self.is_fitted = True
        
        # Ricrea DataFrame
        df_norm = pd.DataFrame(
            normalized,
            columns=feature_columns
        )
        df_norm.insert(0, 'user_id', user_ids)
        
        print(f"[OK] Normalized {len(df_norm)} users, {len(feature_columns)} features")
        return df_norm
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizza nuovi dati usando scaler già fittato.
        
        Args:
            df: DataFrame con stessa struttura del training
            
        Returns:
            DataFrame normalizzato
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_transform() first.")
        
        user_ids = df['user_id'].copy()
        if self.feature_columns is None:
            raise ValueError("Scaler not fitted. Feature columns unknown.")

        features = df[self.feature_columns].values
        normalized = self.scaler.transform(features)
        
        df_norm = pd.DataFrame(normalized, columns=self.feature_columns)
        df_norm.insert(0, 'user_id', user_ids)
        return df_norm
    
    def get_scaling_params(self) -> dict:
        """Restituisce media e std per ogni feature"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted.")
        
        if self.scaler.mean_ is None or self.scaler.scale_ is None:
            raise ValueError("Scaler parameters not available.")

        if self.feature_columns is None:
            raise ValueError("Feature columns not set. Fit the scaler first.")
        
        return {
            'mean': dict(zip(self.feature_columns, self.scaler.mean_)),
            'std': dict(zip(self.feature_columns, self.scaler.scale_))
        }
    
    def save(self, filepath: str = 'data/processed/normalized_features.csv'):
        """Salva DataFrame normalizzato su disco"""
        # Implementato in funzione helper sotto
        pass


# Funzione helper per uso rapido
def normalize_features(
    df: pd.DataFrame,
    save_path: str = "",
    feature_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Normalizza feature con StandardScaler.
    
    Args:
        df: DataFrame raw da normalizzare
        save_path: Path dove salvare output (opzionale)
        
    Returns:
        DataFrame normalizzato
    """
    normalizer = FeatureNormalizer(feature_columns=feature_columns)
    df_norm = normalizer.fit_transform(df)
    
    # Salva se richiesto
    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_norm.to_csv(output_path, index=False)
        print(f"[SAVED] Normalized data to {output_path}")
    
    return df_norm


if __name__ == '__main__':
    # Test normalizer
    from loader import load_user_features
    
    # Carica dati raw
    df_raw = load_user_features('data/raw/user_features.csv')
    print("\nRaw data sample:")
    print(df_raw.head())
    
    # Normalizza
    df_norm = normalize_features(
        df_raw,
        save_path='data/processed/normalized_features.csv'
    )
    print("\nNormalized data sample:")
    print(df_norm.head())
    
    # Mostra parametri scaling
    normalizer = FeatureNormalizer()
    normalizer.fit_transform(df_raw)
    print("\nScaling parameters:")
    params = normalizer.get_scaling_params()
    print(f"Mean: {params['mean']}")
    print(f"Std: {params['std']}")