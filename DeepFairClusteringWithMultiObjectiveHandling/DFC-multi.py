#USE THIS CODE

from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation import eva
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


class SDCN(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, 
                n_input, n_z, n_clusters, v=1):
        super(SDCN, self).__init__()

        # Autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # Cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # Degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        
        sigma = 0.5

        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1-sigma)*h + sigma*tra1, adj)
        h = self.gnn_3((1-sigma)*h + sigma*tra2, adj)
        h = self.gnn_4((1-sigma)*h + sigma*tra3, adj)
        h = self.gnn_5((1-sigma)*h + sigma*z, adj, active=False)
        predict = F.softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def balance_loss(y_pred, sensitive_attrs, weight=1.0):
    """
    Calculates the balance loss to penalize clusters that are imbalanced
    with respect to the combinations of sensitive attributes.
    
    Args:
    - y_pred: Predicted cluster labels (1D array of cluster assignments).
    - sensitive_attrs: Sensitive attributes (2D array).
    - weight: Weight of the balance loss in the total loss function.
    
    Returns:
    - balance_loss_value: A scalar value representing the imbalance penalty.
    """
    # Get the unique clusters and combinations
    clusters = np.unique(y_pred)
    unique_combinations = np.unique(sensitive_attrs, axis=0)
    
    # Initialize balance loss
    balance_loss_value = 0
    
    # Calculate imbalance for each combination
    for combination in unique_combinations:
        combination_mask = np.all(sensitive_attrs == combination, axis=1)
        total_combination_count = np.sum(combination_mask)
        
        # Calculate the proportion of the combination in each cluster
        combination_balance = []
        for cluster in clusters:
            cluster_indices = np.where(y_pred == cluster)[0]
            combination_in_cluster_count = np.sum(combination_mask[cluster_indices])
            combination_balance.append(combination_in_cluster_count / total_combination_count)
        
        # Calculate variance of the balance across clusters (lower variance is more balanced)
        combination_balance = np.array(combination_balance)
        balance_variance = np.var(combination_balance)
        
        # Add to balance loss (higher variance means more imbalance)
        balance_loss_value += balance_variance
    
    # Scale by weight
    balance_loss_value *= weight
    
    return balance_loss_value

'''
def balance_loss_multiple_attributes(y_pred, sensitive_attrs, weight=1.0):
    """
    Calculates the balance loss to penalize clusters that are imbalanced
    with respect to multiple sensitive attributes.
    
    Args:
    - y_pred: Predicted cluster labels (1D array of cluster assignments).
    - sensitive_attrs: A 2D array where each column represents a sensitive attribute (e.g., sex, race).
    - weight: Weight of the balance loss in the total loss function.
    
    Returns:
    - balance_loss_value: A scalar value representing the imbalance penalty for all sensitive attributes.
    """
    # Initialize total balance loss
    total_balance_loss = 0

    # Loop through each sensitive attribute
    for i in range(sensitive_attrs.shape[1]):
        # Extract the current sensitive attribute (column)
        sensitive_attr = sensitive_attrs[:, i]
        
        # Calculate balance loss for this attribute (similar to the single attribute function)
        clusters = np.unique(y_pred)
        unique_attr_values = np.unique(sensitive_attr)
        
        balance_loss_value = 0

        for attr_value in unique_attr_values:
            attr_mask = (sensitive_attr == attr_value)
            total_attr_count = np.sum(attr_mask)

            attr_balance = []
            for cluster in clusters:
                cluster_indices = np.where(y_pred == cluster)[0]
                attr_in_cluster_count = np.sum(attr_mask[cluster_indices])
                attr_balance.append(attr_in_cluster_count / total_attr_count)

            # Calculate variance of the balance across clusters
            attr_balance = np.array(attr_balance)
            balance_variance = np.var(attr_balance)
            balance_loss_value += balance_variance

        total_balance_loss += balance_loss_value

    # Scale by weight and return
    return weight * total_balance_loss

    '''
def calculate_mnce(y_pred, sensitive_attrs):
    """
    Calculate Minimal Normalized Conditional Entropy (MNCE) for a given clustering.
    
    Args:
    - y_pred: Predicted cluster labels (1D array of cluster assignments).
    - sensitive_attrs: Sensitive attributes (2D array).
    
    Returns:
    - MNCE value (between 0 and 1).
    """
    clusters = np.unique(y_pred)
    sensitive_groups = np.unique(sensitive_attrs, axis=0)
    
    # Calculate the global entropy H(G)
    total_samples = len(sensitive_attrs)
    global_entropy = 0
    for group in sensitive_groups:
        group_mask = np.all(sensitive_attrs == group, axis=1)
        group_size = np.sum(group_mask)
        global_entropy -= (group_size / total_samples) * np.log(group_size / total_samples + 1e-10)

    # Calculate the conditional entropy H(G|C_k) for each cluster
    conditional_entropies = []
    for cluster in clusters:
        cluster_indices = np.where(y_pred == cluster)[0]
        cluster_size = len(cluster_indices)
        if cluster_size == 0:
            continue

        cluster_entropy = 0
        for group in sensitive_groups:
            group_mask = np.all(sensitive_attrs == group, axis=1)
            group_in_cluster = np.sum(group_mask[cluster_indices])
            if group_in_cluster > 0:
                cluster_entropy -= (group_in_cluster / cluster_size) * np.log(group_in_cluster / cluster_size + 1e-10)

        conditional_entropies.append(cluster_entropy)
    
    # MNCE is the minimum conditional entropy normalized by the global entropy
    min_conditional_entropy = min(conditional_entropies) if conditional_entropies else 0
    mnce = min_conditional_entropy / (global_entropy + 1e-10)  # To avoid division by zero
    
    return mnce
def balance_loss_multiple_attributes_mnce(y_pred, sensitive_attrs, weight=1.0):
    """
    Calculates the balance loss to penalize clusters that are imbalanced
    with respect to the combinations of multiple sensitive attributes.
    
    Args:
    - y_pred: Predicted cluster labels (1D array of cluster assignments).
    - sensitive_attrs: A 2D array where each column represents a sensitive attribute (e.g., sex, race).
    - weight: Weight of the balance loss in the total loss function.
    
    Returns:
    - balance_loss_value: A scalar value representing the imbalance penalty for all combined sensitive attributes.
    """
    # Initialize total balance loss
    total_balance_loss = 0

    # Create combinations of sensitive attributes (joint categories)
    combined_attrs = [tuple(row) for row in sensitive_attrs]
    unique_combinations = np.unique(combined_attrs, axis=0)

    # Get unique clusters
    clusters = np.unique(y_pred)

    # Loop through each unique combination of sensitive attributes
    for combination in unique_combinations:
        # Create a mask for the current combination of attributes
        combination_mask = np.array(combined_attrs) == combination
        total_combination_count = np.sum(np.all(combination_mask, axis=1))  # Fix

        combination_balance = []

        # Calculate distribution of this combination across clusters
        for cluster in clusters:
            cluster_indices = np.where(y_pred == cluster)[0]
            combination_in_cluster_count = np.sum(np.all(combination_mask[cluster_indices], axis=1))  # Fix
            combination_balance.append(combination_in_cluster_count / total_combination_count)

        # Calculate variance of the balance across clusters for this combination
        combination_balance = np.array(combination_balance)
        balance_variance = np.var(combination_balance)

        # Accumulate the variance for this combination
        total_balance_loss += balance_variance

    # Scale by weight and return
    return weight * total_balance_loss


def balance_loss_multiple_attributes(y_pred, sensitive_attrs, weight=1.0):
    """
    Calculates the balance loss to penalize clusters that are imbalanced
    with respect to the combinations of multiple sensitive attributes.
    
    Args:
    - y_pred: Predicted cluster labels (1D array of cluster assignments).
    - sensitive_attrs: A 2D array where each column represents a sensitive attribute (e.g., sex, race).
    - weight: Weight of the balance loss in the total loss function.
    
    Returns:
    - balance_loss_value: A scalar value representing the imbalance penalty for all combined sensitive attributes.
    """
    # Initialize total balance loss
    total_balance_loss = 0

    # Create combinations of sensitive attributes (joint categories)
    combined_attrs = [tuple(row) for row in sensitive_attrs]
    unique_combinations = np.unique(combined_attrs, axis=0)

    # Get unique clusters
    clusters = np.unique(y_pred)

    # Loop through each unique combination of sensitive attributes
    for combination in unique_combinations:
        # Create a mask for the current combination of attributes
        combination_mask = np.array(combined_attrs) == combination
        total_combination_count = np.sum(combination_mask)

        combination_balance = []

        # Calculate distribution of this combination across clusters
        for cluster in clusters:
            cluster_indices = np.where(y_pred == cluster)[0]
            combination_in_cluster_count = np.sum(combination_mask[cluster_indices])
            combination_balance.append(combination_in_cluster_count / total_combination_count)

        # Calculate variance of the balance across clusters for this combination
        combination_balance = np.array(combination_balance)
        balance_variance = np.var(combination_balance)

        # Accumulate the variance for this combination
        total_balance_loss += balance_variance

    # Scale by weight and return
    return weight * total_balance_loss


def train_sdcn(dataset, alpha=0.5, beta=0.5):
    model = SDCN(500, 500, 2000, 2000, 500, 500,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 n_clusters=args.n_clusters,
                 v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # Load and map sensitive attributes (Sex, Race in this case)
    data_df = pd.read_csv('data/dummy.csv')
    data_df['Sex'] = data_df['Sex'].map({'Male': 1, 'Female': 0})
    
    # Factorize 'Race' or other sensitive attributes as needed
    data_df['Race'], _ = pd.factorize(data_df['Race'])

    # Extract multiple sensitive attributes (Sex, Race in this case)
    sensitive_attrs = data_df[['Sex', 'Race']].values

    # Load the adjacency graph
    adj = load_graph(args.name, args.k)
    adj = adj.to_dense()

    # Prepare dataset for training
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

    # Initialize with k-means on latent space
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    # Variables to track best combined score (silhouette score and balance loss)
    best_combined_score = -float('inf')
    best_epoch = -1
    best_y_pred = None
    best_sensitive_distribution = None

    # Variables to track the least balance loss
    best_balance_loss = float('inf')
    best_balance_epoch = -1
    best_y_pred_balance = None

    for epoch in range(200):
        _, tmp_q, pred, _ = model(data, adj)
        tmp_q = tmp_q.data
        p = target_distribution(tmp_q)

        x_bar, q, pred, _ = model(data, adj)

        # Calculate balance loss for multiple sensitive attributes (Sex, Race)
        y_pred = pred.data.cpu().numpy().argmax(1)
        balance_loss_value = balance_loss_multiple_attributes(y_pred, sensitive_attrs, weight=1.0)

        # Calculate the usual losses (KL, CE, reconstruction)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        # Calculate silhouette score for clustering quality
        if len(np.unique(y_pred)) > 1:  # Ensure there are more than one cluster to compute silhouette score
            sil_score = silhouette_score(z.data.cpu().numpy(), y_pred)
        else:
            sil_score = -1  # If only one cluster, silhouette score is invalid

        # Compute the total loss with the tradeoff parameter alpha
        total_loss = compute_total_loss(alpha, kl_loss, ce_loss, re_loss, balance_loss_value)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Print out loss and score information
        print(f"Epoch {epoch}: Total Loss = {total_loss.item():.4f}, Silhouette Score = {sil_score:.4f}, Balance Loss = {balance_loss_value:.4f}")
        plot_combination_heatmap(y_pred, sensitive_attrs, epoch=epoch)

        # Calculate combined score
        combined_score = sil_score - beta * balance_loss_value

        # Track the best epoch by combined score
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_epoch = epoch
            best_y_pred = y_pred.copy()  # Save the best cluster assignments
            best_sensitive_distribution = sensitive_attrs.copy()  # Save the corresponding sensitive attribute distribution

        # Track the best epoch by least balance loss
        if balance_loss_value < best_balance_loss:
            best_balance_loss = balance_loss_value
            best_balance_epoch = epoch
            best_y_pred_balance = y_pred.copy()  # Save the cluster assignments for the least balance loss

    # After training, print the best epoch by combined score and save the best distribution
    print(f"Best epoch by combined score (Silhouette - Beta * Balance Loss): {best_epoch}, Combined Score: {best_combined_score:.4f}")
    
    # Print the best epoch by least balance loss
    print(f"Best epoch by least balance loss: {best_balance_epoch}, Balance Loss: {best_balance_loss:.4f}")
    
    # Save the best cluster distribution by combined score
    np.savetxt('best_y_pred_combined.txt', best_y_pred, fmt='%d')
    np.savetxt('best_sensitive_distribution_combined.txt', best_sensitive_distribution, fmt='%d')

    # Save the best cluster distribution by least balance loss
    np.savetxt('best_y_pred_balance.txt', best_y_pred_balance, fmt='%d')

    # Visualize the distribution of sensitive attributes after the best epoch by combined score
    #plot_combination_heatmap(best_y_pred, sensitive_attrs, epoch=best_epoch)

    # Visualize the distribution of sensitive attributes after the best epoch by least balance loss
    plot_combination_heatmap(best_y_pred_balance, sensitive_attrs, epoch=best_balance_epoch)

    print("Best cluster predictions and sensitive attribute distribution saved!")
    print("Training completed!")

import matplotlib.pyplot as plt

def train_sdcn_mnce(dataset, alpha=0.5, beta=0.5):
    model = SDCN(500, 500, 2000, 2000, 500, 500,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 n_clusters=args.n_clusters,
                 v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # Load and map sensitive attributes (Sex, Race in this case)
    data_df = pd.read_csv('data/dummy.csv')
    data_df['Sex'] = data_df['Sex'].map({'Male': 1, 'Female': 0})
    
    # Factorize 'Race' or other sensitive attributes as needed
    data_df['Race'], _ = pd.factorize(data_df['Race'])

    # Extract multiple sensitive attributes (Sex, Race in this case)
    sensitive_attrs = data_df[['Sex', 'Race']].values

    # Load the adjacency graph
    adj = load_graph(args.name, args.k)
    adj = adj.to_dense()

    # Prepare dataset for training
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

    # Initialize with k-means on latent space
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    # Variables to track best combined score (silhouette score and balance loss)
    best_combined_score = -float('inf')
    best_epoch = -1
    best_y_pred = None
    best_sensitive_distribution = None

    # Variables to track the least balance loss
    best_balance_loss = float('inf')
    best_balance_epoch = -1
    best_y_pred_balance = None

    # List to store MNCE values across epochs
    mnce_values = []

    for epoch in range(200):
        _, tmp_q, pred, _ = model(data, adj)
        tmp_q = tmp_q.data
        p = target_distribution(tmp_q)

        x_bar, q, pred, _ = model(data, adj)

        # Calculate balance loss for multiple sensitive attributes (Sex, Race)
        y_pred = pred.data.cpu().numpy().argmax(1)
        balance_loss_value = balance_loss_multiple_attributes_mnce(y_pred, sensitive_attrs, weight=1.0)

        # Calculate the usual losses (KL, CE, reconstruction)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        # Calculate silhouette score for clustering quality
        if len(np.unique(y_pred)) > 1:  # Ensure there are more than one cluster to compute silhouette score
            sil_score = silhouette_score(z.data.cpu().numpy(), y_pred)
        else:
            sil_score = -1  # If only one cluster, silhouette score is invalid

        # Compute MNCE
        mnce_value = calculate_mnce(y_pred, sensitive_attrs)

        # Store MNCE value in the list
        mnce_values.append(mnce_value)

        # Compute the total loss with the tradeoff parameter alpha
        total_loss = compute_total_loss(alpha, kl_loss, ce_loss, re_loss, balance_loss_value)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Print out loss and score information
        print(f"Epoch {epoch}: Total Loss = {total_loss.item():.4f}, Silhouette Score = {sil_score:.4f}, "
              f"Balance Loss = {balance_loss_value:.4f}, MNCE = {mnce_value:.4f}")
        #plot_combination_heatmap(y_pred, sensitive_attrs, epoch=epoch)

        # Calculate combined score
        combined_score = sil_score - beta * balance_loss_value

        # Track the best epoch by combined score
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_epoch = epoch
            best_y_pred = y_pred.copy()  # Save the best cluster assignments
            best_sensitive_distribution = sensitive_attrs.copy()  # Save the corresponding sensitive attribute distribution

        # Track the best epoch by least balance loss
        if balance_loss_value < best_balance_loss:
            best_balance_loss = balance_loss_value
            best_balance_epoch = epoch
            best_y_pred_balance = y_pred.copy()  # Save the cluster assignments for the least balance loss

    # After training, print the best epoch by combined score and save the best distribution
    print(f"Best epoch by combined score (Silhouette - Beta * Balance Loss): {best_epoch}, Combined Score: {best_combined_score:.4f}")
    
    # Print the best epoch by least balance loss
    print(f"Best epoch by least balance loss: {best_balance_epoch}, Balance Loss: {best_balance_loss:.4f}")
    
    # Save the best cluster distribution by combined score
    np.savetxt('best_y_pred_combined.txt', best_y_pred, fmt='%d')
    np.savetxt('best_sensitive_distribution_combined.txt', best_sensitive_distribution, fmt='%d')

    # Save the best cluster distribution by least balance loss
    np.savetxt('best_y_pred_balance.txt', best_y_pred_balance, fmt='%d')

    # Visualize the distribution of sensitive attributes after the best epoch by combined score
    plot_combination_heatmap(best_y_pred, sensitive_attrs, epoch=best_epoch)

    # Visualize the distribution of sensitive attributes after the best epoch by least balance loss
    plot_combination_heatmap(best_y_pred_balance, sensitive_attrs, epoch=best_balance_epoch)

    print("Best cluster predictions and sensitive attribute distribution saved!")
    print("Training completed!")

    # Plot MNCE values over the epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 201), mnce_values, label='MNCE', color='blue')
    plt.title('MNCE over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MNCE')
    plt.legend()
    plt.grid(True)
    plt.savefig('mnce_over_epochs.png')
    plt.show()

    print("MNCE values saved and plotted!")



def visualize_multiple_attributes_distribution(cluster_assignments, sensitive_attrs, epoch=None):
    """
    Visualizes the distribution of multiple sensitive attributes (e.g., Sex, Race) across clusters.
    """
    cluster_labels = np.unique(cluster_assignments)
    distribution = []

    for cluster in cluster_labels:
        cluster_indices = np.where(cluster_assignments == cluster)[0]
        for attr_idx, attr_name in enumerate(['Sex', 'Race']):  # Adjust this list based on your attributes
            attr_values, counts = np.unique(sensitive_attrs[cluster_indices, attr_idx], return_counts=True)
            for attr_value, count in zip(attr_values, counts):
                distribution.append([cluster, attr_name, attr_value, count])

    # Convert to DataFrame for visualization
    df = pd.DataFrame(distribution, columns=['Cluster', 'Attribute', 'Value', 'Count'])

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Cluster', y='Count', hue='Attribute', data=df)
    plt.title(f'Distribution of Sensitive Attributes across Clusters (Epoch {epoch if epoch is not None else "Best"})')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.show()

def compute_total_loss(alpha, kl_loss, ce_loss, re_loss, balance_loss_value):
    """
    Combines the traditional clustering loss (KL, CE, reconstruction) and balance loss
    with a tradeoff parameter alpha.

    Args:
    - alpha: Tradeoff parameter (0 < alpha < 1).
    - kl_loss: KL divergence loss.
    - ce_loss: Cross-entropy loss.
    - re_loss: Reconstruction loss.
    - balance_loss_value: Balance loss (penalizing imbalance in sensitive attributes).

    Returns:
    - total_loss: Combined total loss.
    """
    # Clustering loss is a combination of KL, CE, and reconstruction losses
    clustering_loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss

    # Total loss combines clustering loss and balance loss with the tradeoff parameter alpha
    total_loss = (1 - alpha) * clustering_loss + alpha * balance_loss_value

    # Debugging information
    print(f"Clustering Loss: {clustering_loss.item():.4f}, Balance Loss: {balance_loss_value:.4f}, Total Loss: {total_loss.item():.4f}")
    
    return total_loss


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_combination_heatmap(y_pred, sensitive_attrs, epoch=None):
    """
    Function to plot a heatmap of sensitive attribute combinations across clusters.
    
    Args:
    - y_pred: Predicted cluster labels (1D array).
    - sensitive_attrs: 2D array of sensitive attributes (e.g., Sex and Race).
    - epoch: Current epoch for displaying in the title (optional).
    """
    # Get the unique combinations of sensitive attributes
    combinations = np.unique(sensitive_attrs, axis=0)
    
    # Get the unique clusters
    clusters = np.unique(y_pred)
    
    # Initialize the matrix to hold the count of each combination in each cluster
    heatmap_data = np.zeros((len(clusters), len(combinations)))
    
    # Fill the matrix with counts
    for i, combination in enumerate(combinations):
        # Create a mask for this combination
        combination_mask = np.all(sensitive_attrs == combination, axis=1)
        
        for j, cluster in enumerate(clusters):
            # Count how many of this combination are in the cluster
            heatmap_data[j, i] = np.sum((y_pred == cluster) & combination_mask)
    
    # Create the heatmap
    plt.figure(figsize=(10, 6))
    
    # Check if heatmap_data contains float values
    if np.issubdtype(heatmap_data.dtype, np.integer):
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu', xticklabels=combinations, yticklabels=clusters)
    else:
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu', xticklabels=combinations, yticklabels=clusters)
    
    # Add labels and title
    plt.xlabel('Combination (Sex, Race)')
    plt.ylabel('Cluster')
    title = 'Distribution of Combinations Across Clusters'
    if epoch is not None:
        title += f' (Epoch {epoch})'
    plt.title(title)
    
    plt.show()


def visualize_sensitive_attributes(data_df):
    # Plot the distribution of 'Sex'
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.countplot(x='Sex', data=data_df)
    plt.title('Distribution of Sex')
    plt.xlabel('Sex (0: Female, 1: Male)')
    plt.ylabel('Count')

    # Plot the distribution of 'Race'
    plt.subplot(1, 2, 2)
    sns.countplot(x='Race', data=data_df)
    plt.title('Distribution of Race')
    plt.xlabel('Race')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

def check_combinations_balance(data_df):
    # Count combinations of Sex and Race
    combinations_count = data_df.groupby(['Sex', 'Race']).size()
    print("Combinations of Sex and Race counts:\n", combinations_count)

    # Calculate percentage of each combination
    total_count = len(data_df)
    combinations_percentage = combinations_count / total_count
    print("Percentage of each combination:\n", combinations_percentage)

def visualize_clusters_vs_sensitive_attributes(y_pred, data_df):
    # Add cluster labels to the DataFrame
    data_df['Cluster'] = y_pred

    # Plot distribution of clusters vs Sex
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.countplot(x='Cluster', hue='Sex', data=data_df)
    plt.title('Cluster vs Sex')
    plt.xlabel('Cluster')
    plt.ylabel('Count')

    # Plot distribution of clusters vs Race
    plt.subplot(1, 2, 2)
    sns.countplot(x='Cluster', hue='Race', data=data_df)
    plt.title('Cluster vs Race')
    plt.xlabel('Cluster')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

def visualize_pca_clusters(z, y_pred):
    # Project data to 2D for visualization using PCA
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=y_pred, cmap='viridis', s=10)
    plt.title('Cluster Visualization with PCA')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter, label='Cluster Label')
    plt.show()


import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Adding arguments for training
    parser.add_argument('--name', type=str, default='dummy')
    parser.add_argument('--k', type=int, default=5)  # Changed default k to 5 for better graph construction
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_clusters', default=4, type=int)  # Adjust for 2 clusters (income levels)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default='data/synthetic_adult_ae.pkl')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    # Load the dataset
    dataset = load_data(args.name)

    # Add condition for synthetic_adult or dummy dataset
    if args.name == 'synthetic_adult' or args.name == 'dummy':
        args.k = 5  # Define k for KNN graph
        args.n_clusters = 4  # Since the dataset has two income classes
        args.n_input = 20  # Number of features in `processed_features.txt`

    # Other existing dataset configurations
    if args.name == 'usps':
        args.n_clusters = 10
        args.n_input = 256

    if args.name == 'hhar':
        args.k = 5
        args.n_clusters = 6
        args.n_input = 561
        args.pretrain_path = 'data/hhar.pkl'

    if args.name == 'reut':
        args.lr = 1e-4
        args.n_clusters = 4
        args.n_input = 2000

    if args.name == 'acm':
        args.k = None
        args.n_clusters = 3
        args.n_input = 1870

    if args.name == 'dblp':
        args.k = None
        args.n_clusters = 4
        args.n_input = 334

    if args.name == 'cite':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 6
        args.n_input = 3703

    print(args)
    
    # Train SDCN on the given dataset
    train_sdcn_mnce(dataset)

