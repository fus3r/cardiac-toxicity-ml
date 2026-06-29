import os
import pandas as pd

def load_and_clean_csv(file_path):
    """Charge le CSV et nettoie les colonnes (suppression espaces et colonnes vides)."""
    df = pd.read_csv(file_path, sep='\t')
    df.columns = df.columns.str.strip() # Nettoie les noms de colonnes
    df = df.dropna(axis=1, how='all')    # Supprime les colonnes fantômes
    
    required = ['x', 'y', 'z', 'ID2013A']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Colonne '{col}' manquante dans {os.path.basename(file_path)}")
    
    df[required] = df[required].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=required)

    df  = df.groupby(['x', 'y', 'z'], as_index=False)['ID2013A'].mean() #enleve les doublons
    return df