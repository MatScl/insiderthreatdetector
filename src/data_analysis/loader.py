"""
Data Loader per Insider Threat Detection

Carica user_features.csv e valida la struttura.
"""

import pandas as pd
from pathlib import Path


class DataLoader:
    """Carica e valida il CSV delle feature utente"""
    
    REQUIRED_COLUMNS = [
        'user_id',
        'logon_count',
        'after_hours_logon',
        'file_access',
        'sensitive_files',
        'email_count',
        'external_emails',
        'usb_connections'
    ]
    
    def __init__(self, filepath: str):
        """
        Args:
            filepath: Path al file user_features.csv
        """
        self.filepath = Path(filepath)
        self.data = None
        
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
        
        # Valida colonne
        missing_cols = set(self.REQUIRED_COLUMNS) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.feature_names = [col for col in self.data.columns if col != 'user_id']
        
        print(f"[OK] Loaded {len(self.data)} users from {self.filepath}")
        return self.data
    
    def get_summary(self) -> dict:
        """Statistiche descrittive del dataset"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        return {
            'n_users': len(self.data),
            'n_features': len(self.REQUIRED_COLUMNS) - 1,  # escludi user_id
            'missing_values': self.data.isnull().sum().to_dict(),
            'feature_stats': self.data.describe().to_dict()
        }


# Funzione helper per uso rapido
def load_user_features(filepath: str = 'data/raw/user_features.csv') -> pd.DataFrame:
    """
    Carica user_features.csv con validazione automatica.
    
    Args:
        filepath: Path al CSV (default: data/raw/user_features.csv)
        
    Returns:
        DataFrame validato
    """
    loader = DataLoader(filepath)
    return loader.load()


if __name__ == '__main__':
    # Test del loader
    df = load_user_features()
    print(df.head())
    print(f"\nShape: {df.shape}")

    