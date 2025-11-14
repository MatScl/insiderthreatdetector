"""
Script per ridurre il dataset CERT a ~500MB
Campiona un subset di utenti e mantiene tutti i loro log

Usage:
    python reduce_cert_dataset.py --input /path/to/full/cert --output data/raw --size 500
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import os

def get_file_size_mb(filepath):
    """Ritorna dimensione file in MB"""
    return os.path.getsize(filepath) / (1024 * 1024)

def sample_users(df, column='user', n_users=500, random_state=42):
    """
    Campiona n utenti casuali e ritorna subset del DataFrame
    
    Args:
        df: DataFrame originale
        column: nome colonna utente
        n_users: numero utenti da campionare
        random_state: seed per riproducibilità
    
    Returns:
        DataFrame filtrato con solo utenti campionati
    """
    unique_users = df[column].unique()
    print(f"  Utenti totali nel file: {len(unique_users)}")
    
    # Campiona utenti casuali
    np.random.seed(random_state)
    sampled_users = np.random.choice(unique_users, size=min(n_users, len(unique_users)), replace=False)
    
    # Filtra DataFrame
    df_sampled = df[df[column].isin(sampled_users)]
    
    reduction_pct = (1 - len(df_sampled)/len(df)) * 100
    print(f"  Righe originali: {len(df):,}")
    print(f"  Righe campionate: {len(df_sampled):,} ({reduction_pct:.1f}% riduzione)")
    
    return df_sampled

def reduce_cert_dataset(input_dir, output_dir, target_size_mb=500, n_users=500):
    """
    Riduce dataset CERT a dimensione target
    
    Args:
        input_dir: Directory con CSV CERT originali
        output_dir: Directory output per CSV ridotti
        target_size_mb: Dimensione target in MB (default 500)
        n_users: Numero utenti da campionare (default 500)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # File principali CERT
    files_to_process = {
        'logon.csv': 'user',
        'device.csv': 'user',
        'file.csv': 'user',
        'email.csv': 'user',
        'http.csv': 'user'  # Opzionale
    }
    
    print("="*60)
    print(f"RIDUZIONE DATASET CERT")
    print(f"Target: {target_size_mb}MB, {n_users} utenti")
    print("="*60)
    
    # Step 1: Identifica utenti comuni a tutti i file
    print("\n[1/3] Identificazione utenti comuni...")
    
    common_users = None
    for filename in ['logon.csv', 'device.csv', 'file.csv', 'email.csv']:
        filepath = input_path / filename
        
        if not filepath.exists():
            print(f"[WARNING] {filename} non trovato, skip")
            continue
        
        # Leggi solo colonna user (veloce)
        users = pd.read_csv(filepath, usecols=['user'])['user'].unique()
        
        if common_users is None:
            common_users = set(users)
        else:
            common_users = common_users.intersection(users)
        
        print(f"  {filename}: {len(users)} utenti")
    
    print(f"\n[OK] Utenti comuni a tutti i file: {len(common_users)}")
    
    # Step 2: Campiona utenti
    print(f"\n[2/3] Campionamento di {n_users} utenti casuali...")
    
    np.random.seed(42)
    sampled_users = np.random.choice(
        list(common_users), 
        size=min(n_users, len(common_users)), 
        replace=False
    )
    
    print(f"[OK] Campionati {len(sampled_users)} utenti")
    
    # Step 3: Filtra e salva ogni file
    print(f"\n[3/3] Filtraggio e salvataggio file...")
    
    total_size_mb = 0
    
    for filename, user_col in files_to_process.items():
        filepath = input_path / filename
        
        if not filepath.exists():
            print(f"\n[WARNING] {filename} non trovato, skip")
            continue
        
        print(f"\n[FILE] Processando {filename}...")
        
        # Carica file
        print(f"  Caricamento...")
        df = pd.read_csv(filepath)
        original_size = get_file_size_mb(filepath)
        print(f"  Dimensione originale: {original_size:.1f} MB")
        
        # Filtra per utenti campionati
        df_filtered = df[df[user_col].isin(sampled_users)]
        
        # Salva
        output_file = output_path / filename
        df_filtered.to_csv(output_file, index=False)
        
        new_size = get_file_size_mb(output_file)
        total_size_mb += new_size
        
        reduction_pct = (1 - new_size/original_size) * 100
        print(f"  [OK] Salvato: {new_size:.1f} MB ({reduction_pct:.1f}% riduzione)")
        print(f"  Righe: {len(df):,} → {len(df_filtered):,}")
    
    # Copia psychometric.csv (piccolo, teniamolo intero)
    psycho_file = input_path / 'psychometric.csv'
    if psycho_file.exists():
        print(f"\n[FILE] Copiando psychometric.csv...")
        df_psycho = pd.read_csv(psycho_file)
        df_psycho_filtered = df_psycho[df_psycho['user'].isin(sampled_users)]
        df_psycho_filtered.to_csv(output_path / 'psychometric.csv', index=False)
        size = get_file_size_mb(output_path / 'psychometric.csv')
        total_size_mb += size
        print(f"  [OK] Salvato: {size:.1f} MB")
    
    # Summary
    print("\n" + "="*60)
    print("COMPLETATO!")
    print("="*60)
    print(f"Dimensione totale dataset ridotto: {total_size_mb:.1f} MB")
    print(f"Utenti campionati: {len(sampled_users)}")
    print(f"File salvati in: {output_path}")
    
    if total_size_mb > target_size_mb * 1.2:
        print(f"\n[WARNING] Dimensione supera target ({target_size_mb}MB)")
        print(f"   Suggerimento: riduci n_users a ~{int(n_users * target_size_mb / total_size_mb)}")
    else:
        print(f"\n[OK] Dimensione entro target!")
    
    return total_size_mb

def main():
    parser = argparse.ArgumentParser(description='Riduce dataset CERT a dimensione gestibile')
    parser.add_argument('--input', type=str, required=True, 
                       help='Directory con CSV CERT originali')
    parser.add_argument('--output', type=str, default='data/raw',
                       help='Directory output (default: data/raw)')
    parser.add_argument('--size', type=int, default=500,
                       help='Dimensione target in MB (default: 500)')
    parser.add_argument('--users', type=int, default=500,
                       help='Numero utenti da campionare (default: 500)')
    
    args = parser.parse_args()
    
    reduce_cert_dataset(
        input_dir=args.input,
        output_dir=args.output,
        target_size_mb=args.size,
        n_users=args.users
    )

if __name__ == "__main__":
    main()
