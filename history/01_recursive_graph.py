import pandas as pd
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ==========================================
# EXPERIMENT: RECURSIVE GRAPH FEATURES
# Hypothesis: TDEs form a 'cluster' in feature space.
# Method: Build a K-Nearest Neighbor Graph and extract network centrality.
# ==========================================

# 1. Load & Prep
train_lc = pd.read_csv('../data/train_lightcurves.csv')
train_meta = pd.read_csv('../data/train_log.csv')
test_lc = pd.read_csv('../data/test_lightcurves.csv')
test_meta = pd.read_csv('../data/test_log.csv')

def basic_features(df):
    # Simple Aggregation
    return df.groupby('object_id')['Flux'].agg(['mean', 'std', 'max', 'min']).reset_index()

print("Generating Base Features...")
train_feat = basic_features(train_lc).merge(train_meta[['object_id', 'target']], on='object_id')
test_feat = basic_features(test_lc).merge(test_meta[['object_id']], on='object_id')

# Combine for Graph Construction
train_feat['is_train'] = 1
test_feat['is_train'] = 0
full_data = pd.concat([train_feat, test_feat], ignore_index=True)

# 2. Build the Graph
print("Building Nearest Neighbor Graph...")
features = ['mean', 'std', 'max', 'min']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(full_data[features])

# Connect every star to its 5 nearest neighbors
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(X_scaled)
distances, indices = knn.kneighbors(X_scaled)

# Create NetworkX Graph
G = nx.Graph()
for i in range(len(full_data)):
    for j in indices[i]:
        if i != j:
            # Weight = Inverse distance (Closer = Stronger connection)
            dist = distances[i][list(indices[i]).index(j)]
            G.add_edge(i, j, weight=1.0/(dist + 1e-6))

# 3. Extract Graph Features (Centrality)
print("Calculating Graph Centrality...")
# PageRank: "How 'popular' is this star among its neighbors?"
pagerank = nx.pagerank(G, weight='weight')
# Clustering: "Do my neighbors know each other?"
clustering = nx.clustering(G, weight='weight')

full_data['graph_pagerank'] = full_data.index.map(pagerank)
full_data['graph_clustering'] = full_data.index.map(clustering)

# 4. Train Model with Graph Features
print("Training Model...")
X_train = full_data[full_data['is_train'] == 1].drop(columns=['object_id', 'target', 'is_train'])
y_train = full_data[full_data['is_train'] == 1]['target']
X_test = full_data[full_data['is_train'] == 0].drop(columns=['object_id', 'target', 'is_train'])

model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 5. Predict
probs = model.predict_proba(X_test)[:, 1]
sub = pd.DataFrame({'object_id': test_meta['object_id'], 'prediction': (probs > 0.5).astype(int)})
sub.to_csv('submission_recursive_graph.csv', index=False)
print("Done.")
