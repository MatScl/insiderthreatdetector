"""
Feature Normalizer per Insider Threat Detection

Normalizza le feature numeriche usando StandardScaler (z-score).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path


class FeatureNormalizer:
    """Normalizza feature comportamentali (mean=0, std=1)"""
    
    FEATURE_COLUMNS = [
        'logon_count',
        'after_hours_logon',
        'file_access',
        'sensitive_files',
        'email_count',
        'external_emails',
        'usb_connections'
    ]
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizza feature e restituisce nuovo DataFrame.
        
        Args:
            df: DataFrame con colonne user_id + FEATURE_COLUMNS
            
        Returns:
            DataFrame normalizzato (user_id + feature normalizzate)
        """
        # Valida colonne
        missing = set(self.FEATURE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        
        # Estrai user_id
        user_ids = df['user_id'].copy()
        
        # Normalizza solo le feature numeriche
        features = df[self.FEATURE_COLUMNS].values
        normalized = self.scaler.fit_transform(features)
        self.is_fitted = True
        
        # Ricrea DataFrame
        df_norm = pd.DataFrame(
            normalized,
            columns=self.FEATURE_COLUMNS
        )
        df_norm.insert(0, 'user_id', user_ids)
        
        print(f"✅ Normalized {len(df_norm)} users, {len(self.FEATURE_COLUMNS)} features")
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
        features = df[self.FEATURE_COLUMNS].values
        normalized = self.scaler.transform(features)
        
        df_norm = pd.DataFrame(normalized, columns=self.FEATURE_COLUMNS)
        df_norm.insert(0, 'user_id', user_ids)
        return df_norm
    
    def get_scaling_params(self) -> dict:
        """Restituisce media e std per ogni feature"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted.")
        
        if self.scaler.mean_ is None or self.scaler.scale_ is None:
            raise ValueError("Scaler parameters not available.")
        
        return {
            'mean': dict(zip(self.FEATURE_COLUMNS, self.scaler.mean_)),
            'std': dict(zip(self.FEATURE_COLUMNS, self.scaler.scale_))
        }
    
    def save(self, filepath: str = 'data/processed/normalized_features.csv'):
        """Salva DataFrame normalizzato su disco"""
        # Implementato in funzione helper sotto
        pass


# Funzione helper per uso rapido
def normalize_features(
    df: pd.DataFrame,
    save_path: str = ""
) -> pd.DataFrame:
    """
    Normalizza feature con StandardScaler.
    
    Args:
        df: DataFrame raw da normalizzare
        save_path: Path dove salvare output (opzionale)
        
    Returns:
        DataFrame normalizzato
    """
    normalizer = FeatureNormalizer()
    df_norm = normalizer.fit_transform(df)
    
    # Salva se richiesto
    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_norm.to_csv(output_path, index=False)
        print(f"💾 Saved normalized data to {output_path}")
    
    return df_norm


if __name__ == '__main__':
    # Test normalizer
    from loader import load_user_features
    
    # Carica dati raw
    df_raw = load_user_features('data/raw/user_features.csv')
    print("\n📊 Raw data sample:")
    print(df_raw.head())
    
    # Normalizza
    df_norm = normalize_features(
        df_raw,
        save_path='data/processed/normalized_features.csv'
    )
    print("\n📊 Normalized data sample:")
    print(df_norm.head())
    
    # Mostra parametri scaling
    normalizer = FeatureNormalizer()
    normalizer.fit_transform(df_raw)
    print("\n📈 Scaling parameters:")
    params = normalizer.get_scaling_params()
    print(f"Mean: {params['mean']}")
    print(f"Std: {params['std']}")