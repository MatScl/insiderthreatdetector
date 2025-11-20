"""Carica user_features.csv e gestisce le colonne numeriche"""

from pathlib import Path
from typing import List, Optional
import pandas as pd


class DataLoader:
    """Loader per CSV feature utente"""

    # colonne da escludere dalle feature numeriche
    DEFAULT_METADATA_COLUMNS = [
        'starttime', 'endtime', 'user_id', 'role', 'b_unit', 'f_unit',
        'dept', 'team', 'ITAdmin', 'O', 'C', 'E', 'A', 'N', 'insider'
    ]

    def __init__(self, filepath: str, exclude_columns: Optional[List[str]] = None):
        self.filepath = Path(filepath)
        self.data = None
        self.exclude_columns = set(exclude_columns or [])
        self.feature_names: List[str] = []
        
    def load(self):
        """Carica CSV e auto-rileva feature numeriche"""
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        self.data = pd.read_csv(self.filepath)
        
        if 'user_id' not in self.data.columns:
            raise ValueError("Column 'user_id' is required")

        # TODO: maybe add support for categorical features
        numeric_columns = self.data.select_dtypes(include=['number', 'bool']).columns
        metadata_cols = set(col for col in self.DEFAULT_METADATA_COLUMNS if col in self.data.columns)
        excluded = metadata_cols | self.exclude_columns | {'user_id'}
        self.feature_names = [col for col in numeric_columns if col not in excluded]

        if not self.feature_names:
            raise ValueError("No numeric features found")
        
        print(f"Loaded {len(self.data)} users, {len(self.feature_names)} features")
        return self.data
    
    def get_summary(self):
        """Stats del dataset"""
        if self.data is None:
            raise ValueError("Call load() first")
        
        return {
            'n_users': len(self.data),
            'n_features': len(self.feature_names),
            'missing_values': self.data.isnull().sum().to_dict(),
            'feature_stats': self.data[self.feature_names].describe().to_dict()
        }


def load_user_features(filepath: str = 'data/raw/user_features.csv',
                       exclude_columns: Optional[List[str]] = None):
    """Quick helper per caricare il CSV"""
    loader = DataLoader(filepath, exclude_columns=exclude_columns)
    return loader.load()


if __name__ == '__main__':
    df = load_user_features()
    print(df.head())
    print(f"Shape: {df.shape}")
