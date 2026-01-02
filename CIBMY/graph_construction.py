import numpy as np
import pandas as pd
import torch
from scipy.sparse import lil_matrix, coo_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch_geometric.data import Data

def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path)
    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1]

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    if numeric_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    if categorical_cols:
        for col in categorical_cols:
            if X[col].nunique() > 10:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col])
            else:
                X = pd.get_dummies(X, columns=[col], drop_first=True)

    y = y.astype(int)
    return X.values, y, numeric_cols, categorical_cols


def construct_population_graph1(X, y, gender_col=69, similarity_threshold=0.7, max_neighbors=30,
                                min_same_class_ratio=0.8):
    n_nodes = X.shape[0]
    A = lil_matrix((n_nodes, n_nodes))

    sigma = np.mean(X[:, gender_col])

    death_idx = np.where(y == 1)[0]

    batch_size = 1000
    for i in range(0, n_nodes, batch_size):
        diff = np.abs(X[i:i + batch_size, gender_col][:, None] - X[:, gender_col])
        sim = np.exp(-diff ** 2 / (2 * sigma ** 2))

        for ii in range(sim.shape[0]):
            node_idx = i + ii
            is_death_node = (y[node_idx] == 1)

            sorted_idx = np.argsort(-sim[ii])

            neighbors = []
            same_class_count = 0

            for j in sorted_idx:
                if j == node_idx: continue

                if is_death_node:
                    if y[j] == 1:
                        if len(neighbors) < max_neighbors:
                            neighbors.append(j)
                            same_class_count += 1
                    else:
                        if same_class_count / (len(neighbors) + 1) >= min_same_class_ratio:
                            if len(neighbors) < max_neighbors:
                                neighbors.append(j)
                else:
                    if sim[ii, j] > similarity_threshold and len(neighbors) < max_neighbors:
                        neighbors.append(j)

                if len(neighbors) >= max_neighbors:
                    break

            for j in neighbors:
                A[node_idx, j] = sim[ii, j]

    A = A.maximum(A.T)
    A.setdiag(1)
    A = A.tocoo()
    return A


def construct_population_graph2(X, k=6):
    n_nodes = X.shape[0]

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(X)
    distances, indices = nbrs.kneighbors(X)

    A = lil_matrix((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in indices[i]:
            if i != j:
                A[i, j] = 1 - distances[i, np.where(indices[i] == j)[0][0]]

    A = A.maximum(A.T)
    A.setdiag(1)
    return A.toarray()

def construct_population_graph(X, k=3):
    n_nodes = X.shape[0]

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)

    A = lil_matrix((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j, idx in enumerate(indices[i]):
            if i != idx:
                A[i, idx] = 1

    A = A.maximum(A.T)
    A.setdiag(1)

    return A.toarray()


def create_graph_data(X, A, y):

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    if isinstance(A, (lil_matrix, coo_matrix, csr_matrix)):
        A_coo = A.tocoo()
        edge_index = torch.stack([
            torch.tensor(A_coo.row, dtype=torch.long),
            torch.tensor(A_coo.col, dtype=torch.long)
        ])
        edge_weight = torch.tensor(A_coo.data, dtype=torch.float)
    else:
        edge_index = torch.tensor(np.array(np.where(A > 0)), dtype=torch.long)
        edge_weight = torch.tensor(A[A > 0], dtype=torch.float)


    data = Data(x=X_tensor, edge_index=edge_index, y=y_tensor, edge_weight=edge_weight)
    return data