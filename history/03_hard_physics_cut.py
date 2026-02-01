import pandas as pd
import numpy as np

# ==========================================
# EXPERIMENT: PURE PHYSICS FILTERS (NO ML)
# Hypothesis: We can classify TDEs solely by their physical properties
#             (Redshift, Temperature/Color) without using a model.
# ==========================================

# 1. Load Metadata
print("Loading Data...")
# We only need the metadata (log) files for this, as they contain Redshift (z)
test_meta = pd.read_csv('../data/test_log.csv')
test_lc = pd.read_csv('../data/test_lightcurves.csv')

# Helper to get color
def get_color(df):
    # Calculate simple mean flux per band
    means = df.groupby(['object_id', 'Filter'])['Flux'].mean().unstack()
    # u - r color (Ultraviolet minus Red)
    # Lower values = Hotter/Bluer objects
    means['u_r_color'] = means['u'] - means['r']
    return means[['u_r_color']]

print("Calculating Colors...")
colors = get_color(test_lc)
df = test_meta.merge(colors, on='object_id', how='left')

# 2. Apply Hard Cuts
print("Applying Cosmological Rules...")

# Rule 1: Redshift Cut (Extragalactic Origin)
# TDEs occur in Supermassive Black Holes, which are in the centers of galaxies.
# They cannot be stars in our Milky Way (z ~ 0).
# WEAKNESS: Many TDEs in the dataset had missing redshift labeled as 0.0.
mask_extragalactic = df['HostGal_SpecZ'] > 0.01

# Rule 2: Color Cut (Blackbody Radiation)
# TDEs are ~30,000 Kelvin. They should appear very blue (negative u-r color).
mask_hot = df['u_r_color'] < 0.0

# 3. Combine Logic
# If it's a galaxy AND it's hot, we call it a TDE.
predictions = (mask_extragalactic & mask_hot).astype(int)

# 4. Save
sub = pd.DataFrame({'object_id': df['object_id'], 'prediction': predictions})
sub.to_csv('submission_pure_physics.csv', index=False)
print(f"Submission generated. Predicted {predictions.sum()} TDEs out of {len(predictions)} objects.")
print("Warning: This approach relies heavily on metadata accuracy.")
