import numpy as np
import pandas as pd
from scipy.stats import skew
from extinction import fitzpatrick99

# LSST Passband Wavelengths (Angstroms)
WAVELENGTHS = {'u': 3641, 'g': 4704, 'r': 6155, 'i': 7504, 'z': 8695, 'y': 10056}

def apply_extinction_correction(df):
    """
    Corrects flux for interstellar dust extinction using Fitzpatrick (99) law.
    """
    waves = np.array(list(WAVELENGTHS.values()))
    ext_map = dict(zip(WAVELENGTHS.keys(), fitzpatrick99(waves, 1.0, 3.1)))
    
    # Calculate correction coefficient based on the filter
    df['coeff'] = df['Filter'].map(ext_map)
    
    # Apply the correction formula: Flux_Clean = Flux * 10^(0.4 * A_lambda)
    # A_lambda is estimated via EBV * 3.1 * coeff
    df['Flux_Clean'] = df['Flux'] * (10 ** ((df['EBV'] * 3.1 * df['coeff']) / 2.5))
    return df

def extract_features(df_lc, df_meta):
    """
    Main Feature Engineering Pipeline.
    1. Applies Extinction Correction
    2. Aggregates statistical metrics (Mean, Max, Std)
    3. Computes Physics metrics (Color, Burstiness)
    """
    # 1. Merge Metadata and Lightcurves
    cols = ['object_id', 'EBV']
    if 'target' in df_meta.columns:
        cols.append('target')
        
    df = df_lc.merge(df_meta[cols], on='object_id', how='left')
    
    # 2. Apply Physics Correction
    df = apply_extinction_correction(df)
    
    # 3. Basic Aggregations
    feat = df.groupby(['object_id', 'Filter'])['Flux_Clean'].agg(['mean', 'max', 'std']).unstack()
    feat.columns = [f'{c[1]}_{c[0]}' for c in feat.columns]
    
    # 4. Time Domain Features
    time = df.groupby('object_id')['Time (MJD)'].agg(duration=lambda x: x.max() - x.min())
    feat = feat.merge(time, on='object_id')

    # 5. Advanced Physics Features
    # Color (Temperature approximation)
    for b1, b2 in [('u', 'g'), ('g', 'r'), ('r', 'i'), ('i', 'z'), ('z', 'y')]:
        c1, c2 = f'{b1}_mean', f'{b2}_mean'
        if c1 in feat.columns and c2 in feat.columns:
            feat[f'color_{b1}_{b2}'] = feat[c1] - feat[c2]
            
    # Burstiness (Explosive force)
    for b in ['g', 'r']:
        if f'max_{b}' in feat.columns and f'mean_{b}' in feat.columns:
            feat[f'burst_{b}'] = feat[f'max_{b}'] / (feat[f'mean_{b}'] + 1.0)

    return df_meta[['object_id'] + ([ 'target'] if 'target' in df_meta.columns else [])].merge(feat, on='object_id', how='left')
