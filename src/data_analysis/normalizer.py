"""Normalizzazione z-score delle feature numeriche"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureNormalizer:
    """Normalizza feature con StandardScaler (mean=0, std=1)"""

    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_columns = feature_columns
        
    def _resolve_feature_columns(self, df):
        if self.feature_columns is not None:
            missing = set(self.feature_columns) - set(df.columns)  #diff tra set colonne attese e colonne presenti
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            return list(self.feature_columns)

        # select numeric and boolean columns in dataframe
        numeric_cols = df.select_dtypes(include=['number', 'bool']).columns.tolist() 
        inferred = [col for col in numeric_cols if col != 'user_id']
        if not inferred:
            raise ValueError("No numeric columns found")
        self.feature_columns = inferred
        return inferred
        
    def fit_transform(self, df):
        """Normalizza e restituisce nuovo DataFrame"""
        feature_columns = self._resolve_feature_columns(df)
        
        user_ids = df['user_id'].copy()
        features = df[feature_columns].values
        normalized = self.scaler.fit_transform(features)
        self.is_fitted = True
        
        df_norm = pd.DataFrame(normalized, columns=feature_columns)
        df_norm.insert(0, 'user_id', user_ids)
        
        print(f"Normalized {len(df_norm)} users, {len(feature_columns)} features")
        return df_norm
    
    def transform(self, df):
        """Applica normalizzazione a nuovi dati"""
        if not self.is_fitted:
            raise ValueError("Call fit_transform() first")
        
        user_ids = df['user_id'].copy()
        if self.feature_columns is None:
            raise ValueError("Feature columns unknown")

        features = df[self.feature_columns].values
        normalized = self.scaler.transform(features)
        
        df_norm = pd.DataFrame(normalized, columns=self.feature_columns)
        df_norm.insert(0, 'user_id', user_ids)
        return df_norm
    
    def get_scaling_params(self):
        """Ritorna media e std per ogni feature"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted")
        
        if self.scaler.mean_ is None or self.scaler.scale_ is None:
            raise ValueError("Params not available")

        if self.feature_columns is None:
            raise ValueError("Fit scaler first")
        
        return {
            'mean': dict(zip(self.feature_columns, self.scaler.mean_)),
            'std': dict(zip(self.feature_columns, self.scaler.scale_))
        }


def normalize_features(df, save_path="", feature_columns=None):
    """Helper per normalizzare velocemente"""
    normalizer = FeatureNormalizer(feature_columns=feature_columns)
    df_norm = normalizer.fit_transform(df)
    
    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_norm.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
    
    return df_norm


if __name__ == '__main__':
    from loader import load_user_features
    
    df_raw = load_user_features('data/raw/user_features.csv')
    print("\nRaw data:")
    print(df_raw.head())
    
    df_norm = normalize_features(df_raw)
    print("\nNormalized:")
    print(df_norm.head())
