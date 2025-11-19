"""
Data Loader per Insider Threat Detection

Carica user_features.csv e valida la struttura.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd


class DataLoader:
    """Carica e valida il CSV delle feature utente"""

    DEFAULT_METADATA_COLUMNS = [
        'starttime', 'endtime', 'user_id', 'role', 'b_unit', 'f_unit',
        'dept', 'team', 'ITAdmin', 'O', 'C', 'E', 'A', 'N', 'insider'
    ]

    def __init__(self, filepath: str, exclude_columns: Optional[List[str]] = None):
        """
        Args:
            filepath: Path al file user_features.csv
            exclude_columns: Colonne aggiuntive da escludere dalle feature
        """
        self.filepath = Path(filepath)
        self.data = None
        self.exclude_columns = set(exclude_columns or [])
        self.feature_names: List[str] = []
        
    def load(self) -> pd.DataFrame:
        """
        Carica il CSV e valida colonne.
        
        Returns:
            DataFrame con feature utente
            
        Raises:
            FileNotFoundError: Se il file non esiste
            ValueError: Se mancano colonne richieste
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        # Carica CSV
        self.data = pd.read_csv(self.filepath)
        
        if 'user_id' not in self.data.columns:
            raise ValueError("Column 'user_id' is required in user_features.csv")

        numeric_columns = self.data.select_dtypes(include=['number', 'bool']).columns
        metadata_cols = set(col for col in self.DEFAULT_METADATA_COLUMNS if col in self.data.columns)
        excluded = metadata_cols | self.exclude_columns | {'user_id'}
        self.feature_names = [col for col in numeric_columns if col not in excluded]

        if not self.feature_names:
            raise ValueError("No numeric feature columns found after exclusions.")
        
        print(
            f"[OK] Loaded {len(self.data)} users, {len(self.feature_names)} numeric features "
            f"from {self.filepath}"
        )
        return self.data
    
    def get_summary(self) -> dict:
        """Statistiche descrittive del dataset"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        return {
            'n_users': len(self.data),
            'n_features': len(self.feature_names),
            'missing_values': self.data.isnull().sum().to_dict(),
            'feature_stats': self.data[self.feature_names].describe().to_dict()
        }


# Funzione helper per uso rapido
def load_user_features(filepath: str = 'data/raw/user_features.csv',
                       exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Carica user_features.csv con validazione automatica.
    
    Args:
        filepath: Path al CSV (default: data/raw/user_features.csv)
        
    Returns:
        DataFrame validato
    """
    loader = DataLoader(filepath, exclude_columns=exclude_columns)
    return loader.load()


if __name__ == '__main__':
    # Test del loader
    df = load_user_features()
    print(df.head())
    print(f"\nShape: {df.shape}")

    