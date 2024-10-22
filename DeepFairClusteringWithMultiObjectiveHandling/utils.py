import numpy as np
import scipy.sparse as sp
import h5py
import torch
from torch.utils.data import Dataset


import numpy as np
import scipy.sparse as sp

def load_graph(dataset, k):
    # Correctly point to the graph file path
    if k:
        path = 'graph/{}_graph.txt'.format(dataset) 
    else:
        path = 'graph/{}_graph.txt'.format(dataset) 

    # Load the processed features without the header
    data = np.loadtxt('data/{}_processed_features.txt'.format(dataset), skiprows=0)
    #data = np.loadtxt('data/{}.txt'.format(dataset))

    n, _ = data.shape

    # Create index mapping
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    # Load edges from the graph file
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    
    # Create the sparse adjacency matrix
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)

    # Make the adjacency matrix symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    # Normalize and convert to torch sparse tensor
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


'''
class load_data(Dataset):
    def __init__(self, dataset):
        self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
        self.y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))
'''
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class load_data(Dataset):
    def __init__(self, dataset):
        # Load the CSV file
        df = pd.read_csv('data/{}.csv'.format(dataset))
        
        # Identify categorical and numerical features
        categorical_features = ['Education', 'Occupation', 'Sex', 'Race']  # Adjust based on your columns
        numerical_features = ['Age', 'HoursPerWeek']
        
        # One-hot encode categorical features
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        categorical_transformed = one_hot_encoder.fit_transform(df[categorical_features])
        
        # Standardize numerical features
        scaler = StandardScaler()
        numerical_transformed = scaler.fit_transform(df[numerical_features])
        
        # Combine processed features
        self.x = np.hstack([numerical_transformed, categorical_transformed]).astype(np.float32)
        
        # Load labels from the 'Income' column
        self.y = df['Income'].values.astype(int)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]),\
               torch.tensor(self.y[idx]),\
               torch.tensor(idx)





