#USE THIS CODE FOR NON ALPHA BETA TRAINING

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

def balance_loss_single_attribute(y_pred, sensitive_attr, weight=1.0):
    """
    Calculates the balance loss to penalize clusters that are imbalanced
    with respect to a single sensitive attribute.
    
    Args:
    - y_pred: Predicted cluster labels (1D array of cluster assignments).
    - sensitive_attr: A single sensitive attribute (1D array).
    - weight: Weight of the balance loss in the total loss function.
    
    Returns:
    - balance_loss_value: A scalar value representing the imbalance penalty.
    """
    # Get unique clusters and unique values of the sensitive attribute
    clusters = np.unique(y_pred)
    unique_attr_values = np.unique(sensitive_attr)
    
    # Initialize balance loss
    balance_loss_value = 0

    # Calculate imbalance for each unique attribute value
    for attr_value in unique_attr_values:
        # Get the mask for the attribute value
        attr_mask = (sensitive_attr == attr_value)
        total_attr_count = np.sum(attr_mask)

        # Calculate the proportion of the attribute value in each cluster
        attr_balance = []
        for cluster in clusters:
            cluster_indices = np.where(y_pred == cluster)[0]
            attr_in_cluster_count = np.sum(attr_mask[cluster_indices])
            balance = attr_in_cluster_count / total_attr_count if total_attr_count > 0 else 0
            attr_balance.append(balance)

        # Print balance scores for this attribute value
        print(f"Attribute Value {attr_value}: Balance across clusters: {attr_balance}")

        # Calculate variance of the balance across clusters
        attr_balance = np.array(attr_balance)
        balance_variance = np.var(attr_balance)
        
        # Add to balance loss
        balance_loss_value += balance_variance
    
    # Scale by weight
    balance_loss_value *= weight
    
    # Print final balance loss
    print(f"Balance Loss Value: {balance_loss_value:.4f}")
    
    return balance_loss_value



import seaborn as sns
import matplotlib.pyplot as plt


def train_sdcn(dataset):
    model = SDCN(500, 500, 2000, 2000, 500, 500,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # Load and visualize sensitive attributes
    data_df = pd.read_csv('data/dummy.csv')
    data_df['Sex'] = data_df['Sex'].map({'Male': 1, 'Female': 0})
    
    # Extract 'Sex' attribute
    sensitive_sex = data_df['Sex'].values

    adj = load_graph(args.name, args.k)
    adj = adj.to_dense()
    
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

    with torch.no_grad():
        _, _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    for epoch in range(200):
        if epoch % 1 == 0:
            # update_interval
            _, tmp_q, pred, _ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            res1 = tmp_q.cpu().numpy().argmax(1)       # Q
            res2 = pred.data.cpu().numpy().argmax(1)   # Z
            res3 = p.data.cpu().numpy().argmax(1)      # P
            
            # Evaluate clustering with balance for Sex
            eva(y, res1, z.data.cpu().numpy(), sensitive_sex, str(epoch) + 'Q')
            eva(y, res2, z.data.cpu().numpy(), sensitive_sex, str(epoch) + 'Z')
            eva(y, res3, z.data.cpu().numpy(), sensitive_sex, str(epoch) + 'P')

            # Visualize the distribution of Sex across clusters
            visualize_sex_distribution(y_pred, sensitive_sex, epoch)

        x_bar, q, pred, _ = model(data, adj)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)
        
        # Calculate balance loss for 'Sex'
        y_pred = pred.data.cpu().numpy().argmax(1)
        balance_loss_value = balance_loss_single_attribute(y_pred, sensitive_sex, weight=5.0)  # Adjust weight as needed

        # Total loss including balance for 'Sex'
        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss + 0.01 * balance_loss_value

        print(f"Epoch {epoch} - Loss: {loss.item()} (KL: {kl_loss.item()}, CE: {ce_loss.item()}, RE: {re_loss.item()}, Balance Loss (Sex): {balance_loss_value})")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Training completed!")


def visualize_sex_distribution(y_pred, sensitive_sex, epoch):
    """
    Visualizes the distribution of 'Sex' across clusters after each epoch.
    
    Args:
    - y_pred: Predicted cluster labels.
    - sensitive_sex: The 'Sex' values for each data point.
    - epoch: The current epoch number for labeling the plot.
    """
    df = pd.DataFrame({'Cluster': y_pred, 'Sex': sensitive_sex})

    plt.figure(figsize=(10, 4))

    # Plot the distribution of 'Sex' across clusters
    sns.countplot(x='Cluster', hue='Sex', data=df)
    plt.title(f'Sex Distribution Across Clusters (Epoch {epoch})')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.legend(title='Sex', loc='upper right', labels=['Female (0)', 'Male (1)'])
    
    # Save the figure for each epoch for later comparison
    plt.tight_layout()
    #plt.savefig(f'cluster_sex_distribution_epoch_{epoch}.png')
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

'''
def train_sdcn(dataset, alpha=0.5, beta=0.5):
    model = SDCN(500, 500, 2000, 2000, 500, 500,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 n_clusters=args.n_clusters,
                 v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # Load and map sensitive attributes (Sex in this case)
    data_df = pd.read_csv('data/dummy.csv')
    data_df['Sex'] = data_df['Sex'].map({'Male': 1, 'Female': 0})

    # Extract the 'Sex' attribute
    sensitive_sex = data_df['Sex'].values

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
    best_sensitive_sex_distribution = None

    for epoch in range(200):
        _, tmp_q, pred, _ = model(data, adj)
        tmp_q = tmp_q.data
        p = target_distribution(tmp_q)

        x_bar, q, pred, _ = model(data, adj)

        # Calculate balance loss for 'Sex'
        y_pred = pred.data.cpu().numpy().argmax(1)
        balance_loss_value = balance_loss_single_attribute(y_pred, sensitive_sex, weight=1.0)

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
        visualize_sex_distribution(y_pred, sensitive_sex, epoch)

        # Calculate combined score
        combined_score = sil_score - beta * balance_loss_value

        # Track the best epoch by combined score
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_epoch = epoch
            best_y_pred = y_pred.copy()  # Save the best cluster assignments
            best_sensitive_sex_distribution = sensitive_sex.copy()  # Save the corresponding sensitive attribute distribution

    # After training, print the best epoch by combined score and save the best distribution
    print(f"Best epoch by combined score (Silhouette - Beta * Balance Loss): {best_epoch}, Combined Score: {best_combined_score:.4f}")
    
    # Save the best cluster distribution
    np.savetxt('best_y_pred_combined.txt', best_y_pred, fmt='%d')
    np.savetxt('best_sensitive_sex_distribution_combined.txt', best_sensitive_sex_distribution, fmt='%d')

    # Visualize the distribution of Sex after the best epoch
    visualize_sex_distribution(best_y_pred, best_sensitive_sex_distribution, epoch=best_epoch)

    print("Best cluster predictions and sensitive attribute distribution saved!")
    print("Training completed!")
'''

def visualize_sex_distribution(cluster_assignments, sensitive_sex, epoch=None):
    """
    Visualizes the distribution of Sex (Male/Female) across the clusters.
    """
    cluster_labels = np.unique(cluster_assignments)
    distribution = []

    for cluster in cluster_labels:
        cluster_indices = np.where(cluster_assignments == cluster)[0]
        male_count = np.sum(sensitive_sex[cluster_indices] == 1)
        female_count = np.sum(sensitive_sex[cluster_indices] == 0)
        distribution.append([cluster, 'Male', male_count])
        distribution.append([cluster, 'Female', female_count])

    # Convert to DataFrame for visualization
    df = pd.DataFrame(distribution, columns=['Cluster', 'Sex', 'Count'])

    # Plot the distribution
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Cluster', y='Count', hue='Sex', data=df)
    plt.title(f'Distribution of Sex across Clusters (Epoch {epoch if epoch is not None else "Best"})')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
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
    parser.add_argument('--n_clusters', default=2, type=int)  # Adjust for 2 clusters (income levels)
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
        args.n_clusters = 2  # Since the dataset has two income classes
        args.n_input = 20  # Number of features in `processed_features.txt`

    # Other existing dataset configurations
    if args.name == 'usps':
        args.n_clusters = 10
        args.n_input = 256

    if args.name == 'hhar':
        args.k = 5
        args.n_clusters = 6
        args.n_input = 561

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
    train_sdcn(dataset)

