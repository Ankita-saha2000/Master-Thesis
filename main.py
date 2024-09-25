from __future__ import print_function, division
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
import sys
from GNN import GNNLayer
from evaluation import eva, cluster_balance, cluster_balance_generalized
import matplotlib.pyplot as plt

class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z):
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
                 n_input, n_z, n_clusters, v=1, pretrain_path=None):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(n_enc_1=n_enc_1, n_enc_2=n_enc_2, n_enc_3=n_enc_3,
                     n_dec_1=n_dec_1, n_dec_2=n_dec_2, n_dec_3=n_dec_3,
                     n_input=n_input, n_z=n_z)

        # Load pretrained AE weights if available
        if pretrain_path is not None:
            print(f"Loading pretrained model from {pretrain_path}")
            self.ae.load_state_dict(torch.load(pretrain_path, map_location='cpu'))

        # GCN layers for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)

        sigma = 0.5

        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)
        h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)
        h = self.gnn_5((1 - sigma) * h + sigma * z, adj, active=False)
        predict = F.softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def demographic_parity_loss(pred, sensitive_attr):
    """
    Generalized Demographic Parity Loss for multiple sensitive attributes.
    
    Args:
    - pred: Predicted probabilities from the model (output of softmax).
    - sensitive_attr: Sensitive attributes (1D or 2D array).
    
    Returns:
    - Total demographic parity loss across all sensitive attributes.
    """
    dp_loss = 0  # Initialize total demographic parity loss

    # If sensitive_attr is 1D, reshape to 2D for uniformity
    if len(sensitive_attr.shape) == 1:
        sensitive_attr = sensitive_attr.reshape(-1, 1)

    # Loop over each sensitive attribute
    for attr_idx in range(sensitive_attr.shape[1]):
        attr_values = sensitive_attr[:, attr_idx]
        unique_groups = np.unique(attr_values)  # Unique groups for the current sensitive attribute

        group_probs = {}  # Dictionary to store mean prediction probabilities for each group

        # Calculate group probabilities for each unique group
        for group in unique_groups:
            group_probs[group] = pred[attr_values == group].mean(dim=0)

        # Calculate demographic parity loss (L2 norm of differences between group probabilities)
        groups = list(group_probs.keys())
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                dp_loss += torch.norm(group_probs[groups[i]] - group_probs[groups[j]], p=2)

    return dp_loss

'''

def demographic_parity_loss(pred, sensitive_attr):
    group_0_prob = pred[sensitive_attr == 0].mean(dim=0)  # Group 0 (e.g., female)
    group_1_prob = pred[sensitive_attr == 1].mean(dim=0)  # Group 1 (e.g., male)

    dp_loss = torch.norm(group_0_prob - group_1_prob, p=2)  # L2 loss
    return dp_loss

'''

def preprocess_adult_dataset(file_path):
    # Load the dataset using pandas
    data = pd.read_csv(file_path)
    
    # Define the column names (adjust these to match your dataset)
    columns = ['sex', 'age', 'workclass', 'final-weight', 'education', 'education-num', 
               'marital-status', 'occupation', 'relationship', 'race', 'gender', 
               'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    data.columns = columns

    # Print initial dataset shape
    print(f"Initial data shape: {data.shape}")

    # Drop the 'gender' column since it's completely missing
    data = data.drop(columns=['gender'])
    print(f"Data shape after dropping the 'gender' column: {data.shape}")

    # Check for missing values per column
    print("Missing values per column:\n", data.isnull().sum())

    # Fill missing values in categorical columns with mode and numerical columns with median
    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 
                           'relationship', 'race', 'native-country', 'income']
    numerical_columns = ['age', 'final-weight', 'education-num', 'capital-gain', 
                         'capital-loss', 'hours-per-week']

    # Fill missing values in categorical columns with the most frequent value (mode)
    for col in categorical_columns:
        if data[col].isnull().sum() > 0:
            mode_value = data[col].mode().dropna()
            if len(mode_value) > 0:
                print(f"Filling missing values in {col} with mode: {mode_value[0]}")
                data[col] = data[col].fillna(mode_value[0])

    # Fill missing values in numerical columns with the median value
    for col in numerical_columns:
        if data[col].isnull().sum() > 0:
            median_value = data[col].median()
            print(f"Filling missing values in {col} with median: {median_value}")
            data[col] = data[col].fillna(median_value)

    # Convert categorical columns to numeric using LabelEncoder
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])

    print(f"Data shape after filling missing values: {data.shape}")

    # Scale the numerical features
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    print(f"Scaled numerical features: {data[numerical_columns].head()}")

    return data.values  # Convert to NumPy array


import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data_with_blobs(n_samples=1000, n_features=5, n_clusters=4, cluster_std=1.0, num_sensitive_attrs=2):
    """
    Generates synthetic data with specified number of clusters and features using make_blobs.
    Also generates multiple sensitive attributes.
    
    Parameters:
    - n_samples: The number of samples in the dataset.
    - n_features: The number of features (dimensions) for each sample (e.g., 5 or 6).
    - n_clusters: The number of clusters (centers) to generate.
    - cluster_std: The standard deviation of the clusters.
    - num_sensitive_attrs: Number of sensitive attributes to generate (default is 2).
    
    Returns:
    - X: The generated samples with shape (n_samples, n_features).
    - sensitive_attrs: A 2D array of synthetic sensitive attributes (e.g., gender, race).
    """
    # Generate the blobs dataset with n_clusters
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, cluster_std=cluster_std, random_state=42)

    # Create multiple sensitive attributes randomly, using different probability distributions for each
    sensitive_attrs = []
    
    for i in range(num_sensitive_attrs):
        # Example: For each sensitive attribute, use a different random distribution
        if i == 0:
            # Gender-like binary attribute (e.g., 70% 0, 30% 1)
            attr = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        elif i == 1:
            # Race-like categorical attribute (e.g., 50% 0, 30% 1, 20% 2)
            attr = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])
        else:
            # For any other sensitive attribute, create a random binary or categorical distribution
            attr = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])

        sensitive_attrs.append(attr)
    
    # Stack sensitive attributes into a 2D array
    sensitive_attrs = np.column_stack(sensitive_attrs)

    print(f"Generated synthetic data with shape: {X.shape} and {n_clusters} clusters.")
    print(f"Sensitive Attribute Distribution per attribute:")
    for i in range(num_sensitive_attrs):
        print(f"Sensitive Attribute {i + 1} distribution: {np.bincount(sensitive_attrs[:, i])}")
    
    return X, sensitive_attrs


def visualize_dataset(X, sensitive_attr, method='pca'):
    """
    Visualize the synthetic dataset using PCA or t-SNE.
    
    Parameters:
    - X: The dataset (features).
    - sensitive_attr: The sensitive attribute (e.g., binary gender label).
    - method: 'pca' or 'tsne' for visualization method.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if method == 'pca':
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        plt.title("PCA Projection of the Dataset")
    elif method == 'tsne':
        tsne = TSNE(n_components=2, random_state=42)
        X_reduced = tsne.fit_transform(X)
        plt.title("t-SNE Projection of the Dataset")
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    # Plotting
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=sensitive_attr, cmap='coolwarm', s=50, edgecolor='k', alpha=0.75)
    plt.colorbar(scatter, label="Sensitive Attribute")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()



def pretrain_autoencoder(dataset, epochs=30, lr=1e-3):
    model = AE(500, 500, 2000, 2000, 500, 500, n_input=dataset.shape[1], n_z=16).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    dataset_tensor = torch.Tensor(dataset).to(device)

    for epoch in range(epochs):
        model.train()
        x_bar, _, _, _, z = model(dataset_tensor)
        re_loss = F.mse_loss(x_bar, dataset_tensor)

        optimizer.zero_grad()
        re_loss.backward()
        optimizer.step()

        print(f"Pretraining Epoch {epoch+1}/{epochs}, Reconstruction Loss: {re_loss.item()}")

    return model.state_dict()  # Save pretrained weights



def plot_metrics(silhouette_scores, fairness_losses, balances, purities, balance_losses):
    epochs = range(1, len(silhouette_scores) + 1)

    plt.figure(figsize=(14, 10))

    # Plot Silhouette Score
    plt.subplot(3, 2, 1)
    plt.plot(epochs, silhouette_scores, 'b', label='Silhouette Score')
    plt.title('Silhouette Score over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Silhouette Score')
    plt.legend()

    # Plot Fairness Loss
    plt.subplot(3, 2, 2)
    plt.plot(epochs, fairness_losses, 'r', label='Fairness Loss')
    plt.title('Fairness Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Fairness Loss')
    plt.legend()

    # Plot Balance
    plt.subplot(3, 2, 3)
    plt.plot(epochs, balances, 'g', label='Balance')
    plt.title('Balance over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Balance')
    plt.legend()

    # Plot Cluster Purity
    plt.subplot(3, 2, 4)
    plt.plot(epochs, purities, 'c', label='Cluster Purity')
    plt.title('Cluster Purity over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Cluster Purity')
    plt.legend()

    # Plot Balance Loss
    plt.subplot(3, 2, 5)
    plt.plot(epochs, balance_losses, 'm', label='Balance Loss')
    plt.title('Balance Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Balance Loss')
    plt.legend()

    # Show plots
    plt.tight_layout()
    plt.show()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def plot_pca_clusters(data, y_pred, title='PCA Cluster Assignment'):
    """
    Perform PCA on the data and plot the cluster assignments in 2D space.

    Args:
    - data: The dataset in high-dimensional space, as a PyTorch tensor.
    - y_pred: The predicted cluster assignments, as a PyTorch tensor.
    - title: Title of the plot.
    """
    # Detach the tensors and move to CPU if necessary
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Perform PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    # Determine the number of unique clusters
    num_clusters = len(np.unique(y_pred))

    # Plot the clusters in 2D PCA space
    plt.figure(figsize=(8, 6))

    # Use a discrete colormap (such as tab10) that provides distinct colors for each cluster
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y_pred, cmap='tab10', s=50)

    # Add a colorbar with the number of clusters
    cbar = plt.colorbar(scatter, ticks=range(num_clusters))
    cbar.set_label('Cluster Label')
    
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.show()

def plot_tsne_clusters(X, y_pred, sensitive_attr, title='t-SNE Cluster Visualization'):
    """
    Perform t-SNE on the data and plot the clusters with color coding for sensitive attribute.
    
    Parameters:
    - X: The original high-dimensional data (before clustering).
    - y_pred: The predicted cluster assignments from the clustering model.
    - sensitive_attr: The sensitive attribute(s) (1D or 2D array). For 2D, the user must specify the column.
    - title: Title for the plot.
    """
    # Check if sensitive_attr is 1D or 2D
    if len(sensitive_attr.shape) == 1:
        attr_to_plot = sensitive_attr  # If 1D, use the sensitive_attr as is
    elif len(sensitive_attr.shape) == 2:
        attr_to_plot = sensitive_attr[:, 0]  # If 2D, default to using the first sensitive attribute
        print("Sensitive attribute has multiple columns. Defaulting to the first column for visualization.")
    else:
        raise ValueError("Sensitive attribute must be 1D or 2D array")

    # Perform t-SNE to reduce data to 2 dimensions for visualization
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Debugging: Check the shapes
    print(f"X_tsne shape: {X_tsne.shape}, attr_to_plot shape: {attr_to_plot.shape}")

    # Ensure that sensitive_attr has the same length as the data
    if X_tsne.shape[0] != attr_to_plot.shape[0]:
        raise ValueError(f"Length mismatch: t-SNE data has {X_tsne.shape[0]} samples, but sensitive_attr has {attr_to_plot.shape[0]} entries.")

    # Create a scatter plot of the t-SNE reduced data
    plt.figure(figsize=(8, 6))
    
    # Scatter plot color-coded by sensitive attribute
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=attr_to_plot, cmap='coolwarm', s=50, edgecolor='k', alpha=0.75)
    
    # Add a color bar showing the distribution of the sensitive attribute
    cbar = plt.colorbar(scatter, label="Sensitive Attribute (e.g., Gender or Race)")
    
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.show()


def train_sdcn_synthetic(dataset, sensitive_attr, n_clusters=2, min_clusters=2, fairness_weight=0.01, balance_weight=0.001, silhouette_threshold=0.1, epochs=50, evaluation_start=10):
    # Pretrain the autoencoder
    pretrain_epochs = 20
    ae_pretrain_path = pretrain_autoencoder(dataset, epochs=pretrain_epochs)

    # Initialize lists to store metric values for each epoch
    silhouette_scores = []
    fairness_losses = []
    balances_gender = []  # For the first sensitive attribute (e.g., gender)
    balances_race = []  # For the second sensitive attribute (e.g., race)
    purities = []
    balance_losses = []

    try:
        model = SDCN(500, 500, 2000, 2000, 500, 500,
                     n_input=dataset.shape[1],
                     n_z=16,
                     n_clusters=n_clusters,
                     v=1.0).to(device)

        model.ae.load_state_dict(ae_pretrain_path)  # Load pretrained autoencoder weights

        optimizer = Adam(model.parameters(), lr=1e-3)

        adj = torch.eye(dataset.shape[0]).to(device)
        dataset_tensor = torch.Tensor(dataset).to(device)

        # Initialize KMeans clustering
        with torch.no_grad():
            _, _, _, _, z = model.ae(dataset_tensor)
            kmeans = KMeans(n_clusters=n_clusters, n_init=1000)
            y_pred = kmeans.fit_predict(z.cpu().numpy())
            y_pred_last = y_pred
            model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

        # Evaluate the initial clustering performance for the first sensitive attribute (e.g., gender)
        eva(sensitive_attr[:, 0], y_pred, epoch='Initial(Sex)', fairness_loss=None, X=z.cpu().numpy(), sensitive_attr=sensitive_attr[:, 0])

        # Evaluate the initial clustering performance for the second sensitive attribute (e.g., race)
        eva(sensitive_attr[:, 1], y_pred, epoch='Initial (Race)', fairness_loss=None, X=z.cpu().numpy(), sensitive_attr=sensitive_attr[:, 1])

        # Start training loop
        for epoch in range(epochs):
            x_bar, q, pred, z = model(dataset_tensor, adj)
            p = target_distribution(q)

            # Fairness and balance losses
            fairness_loss_gender = demographic_parity_loss(pred, sensitive_attr[:, 0]).item()  # For gender
            fairness_loss_race = demographic_parity_loss(pred, sensitive_attr[:, 1]).item()  # For race
            y_pred = pred.detach().cpu().numpy().argmax(axis=1)

            # Calculate balance score for the first sensitive attribute (e.g., gender)
            balance_gender = cluster_balance_generalized(y_pred, sensitive_attr[:, 0])
            # Calculate balance score for the second sensitive attribute (e.g., race)
            balance_race = cluster_balance_generalized(y_pred, sensitive_attr[:, 1])

            # Use gender balance for the balance loss, but you could choose either or combine them
            balance_loss = 1.0 - balance_gender  # Higher score = lower loss, 1.0 means perfectly balanced

            # Compute other losses
            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
            ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
            re_loss = F.mse_loss(x_bar, dataset_tensor)

            # Total loss including fairness and balance losses
            loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss + fairness_weight * fairness_loss_gender + balance_weight * balance_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate from the beginning
            with torch.no_grad():
                _, tmp_q, pred, _ = model(dataset_tensor, adj)
                y_pred = pred.detach().cpu().numpy().argmax(axis=1)

                # Number of active clusters (clusters with at least one sample)
                active_clusters = len(np.unique(y_pred))

                # Compute silhouette score
                silhouette_avg = silhouette_score(dataset_tensor.cpu().numpy(), y_pred)

                # Store metrics
                metrics_gender = eva(sensitive_attr[:, 0], y_pred, epoch=epoch, fairness_loss=fairness_loss_gender, X=z.cpu().numpy(), sensitive_attr=sensitive_attr[:, 0])
                metrics_race = eva(sensitive_attr[:, 1], y_pred, epoch=f'{epoch} (Race)', fairness_loss=fairness_loss_race, X=z.cpu().numpy(), sensitive_attr=sensitive_attr[:, 1])

                silhouette_scores.append(silhouette_avg)
                fairness_losses.append(fairness_loss_gender)
                balances_gender.append(balance_gender)
                balances_race.append(balance_race)
                purities.append(metrics_gender['purity'])
                balance_losses.append(balance_loss)

            # Apply stopping criteria only after `evaluation_start` epochs
            if epoch >= evaluation_start:
                if active_clusters < min_clusters:
                    print(f"Stopping training: Less than {min_clusters} clusters at epoch {epoch}.")
                    break

                if silhouette_avg < silhouette_threshold:
                    print(f"Stopping training: Silhouette score fell below {silhouette_threshold} at epoch {epoch}.")
                    break

    except Exception as e:
        print(f"Error occurred during training: {e}")

    # Plot metrics after training
    plot_metrics(silhouette_scores, fairness_losses, balances_gender, purities, balance_losses)
    plot_metrics(silhouette_scores, fairness_losses, balances_race, purities, balance_losses)  # Plot for race as well
    plot_pca_clusters(z, y_pred, title='Final Cluster Assignment (PCA)')
    plot_tsne_clusters(dataset, y_pred, sensitive_attr[:, 0], title='Final t-SNE Cluster Visualization with Sex')
    plot_tsne_clusters(dataset, y_pred, sensitive_attr[:, 1], title='Final t-SNE Cluster Visualization with Race')



if __name__ == "__main__":
    # Generate synthetic data with multiple sensitive attributes
    X, sensitive_attrs = generate_synthetic_data_with_blobs(n_samples=1000, n_features=20, n_clusters=4, num_sensitive_attrs=2)

    # Visualize the dataset using t-SNE
    visualize_dataset(X, sensitive_attrs[:, 0], method='tsne')  # Visualize with the first sensitive attribute
    visualize_dataset(X, sensitive_attrs[:, 1], method='tsne')  # Visualize with the second sensitive attribute

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the SDCN model with the synthetic dataset, using multiple sensitive attributes
    train_sdcn_synthetic(X, sensitive_attrs, n_clusters=4, min_clusters=2, fairness_weight=0.5, balance_weight=0.05, silhouette_threshold=0.01, epochs=50, evaluation_start=20)





'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # File path for the dataset
    parser.add_argument('--file_path', type=str, default='/path/to/adult.data')  # Specify the adult dataset file path
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=2, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default=None)
    
    # Fairness and balance weights
    parser.add_argument('--fairness_weight', type=float, default=0.05)
    parser.add_argument('--balance_weight', type=float, default=0.01)
    
    # New parameters for controlling cluster behavior and silhouette score
    parser.add_argument('--min_clusters', type=int, default=2, help='Minimum number of clusters. Training will stop if the number of clusters falls below this.')
    parser.add_argument('--silhouette_threshold', type=float, default=0.1, help='Minimum silhouette score. Training will stop if the silhouette score falls below this.')

    args = parser.parse_args()

    # Check if CUDA is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess the dataset
    dataset = preprocess_adult_dataset(args.file_path)
    sensitive_attr = torch.tensor(dataset[:, 0], dtype=torch.long)  # The first column is 'sex' (0 or 1)
    
    # Set the number of input features to match the dataset shape
    args.n_input = dataset.shape[1]

    # Train the SDCN model with the dataset and the parsed arguments
    train_sdcn(dataset, 
               sensitive_attr, 
               fairness_weight=args.fairness_weight, 
               balance_weight=args.balance_weight, 
               min_clusters=args.min_clusters, 
               silhouette_threshold=args.silhouette_threshold)
'''

'''
def train_sdcn(dataset, sensitive_attr, fairness_weight=0.01, balance_weight=0.001, min_clusters=8, silhouette_threshold=0.1):
    # Pretrain the autoencoder
    pretrain_epochs = 20
    ae_pretrain_path = pretrain_autoencoder(dataset, epochs=pretrain_epochs)

    # Initialize lists to store metric values for each epoch
    silhouette_scores = []
    fairness_losses = []
    balances = []
    purities = []
    balance_losses = []

    with open('debug_log.txt', 'w') as f:
        try:
            model = SDCN(500, 500, 2000, 2000, 500, 500,
                         n_input=args.n_input,
                         n_z=args.n_z,
                         n_clusters=args.n_clusters,
                         v=1.0,
                         pretrain_path=None).to(device)

            model.ae.load_state_dict(ae_pretrain_path)  # Load pretrained autoencoder weights

            print(model)
            f.write("Model initialized and pretrained weights loaded.\n")
            f.flush()

            optimizer = Adam(model.parameters(), lr=args.lr)

            adj = torch.eye(dataset.shape[0]).to(device)
            dataset_tensor = torch.Tensor(dataset).to(device)

            # Initialize KMeans clustering
            with torch.no_grad():
                _, _, _, _, z = model.ae(dataset_tensor)
                kmeans = KMeans(n_clusters=args.n_clusters, n_init=1000)
                y_pred = kmeans.fit_predict(z.cpu().numpy())
                y_pred_last = y_pred
                model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

            # Evaluate the initial clustering performance
            eva(sensitive_attr.cpu().numpy(), y_pred, epoch='Initial', fairness_loss=None, X=z.cpu().numpy(), sensitive_attr=sensitive_attr.cpu().numpy())

            # Start training loop
            for epoch in range(7):
                x_bar, q, pred, z = model(dataset_tensor, adj)
                p = target_distribution(q)

                # Fairness and balance losses
                fairness_loss = demographic_parity_loss(pred, sensitive_attr).item()
                y_pred = pred.detach().cpu().numpy().argmax(axis=1)

                # Calculate balance score
                balance_score = cluster_balance(y_pred, sensitive_attr.cpu().numpy())
                balance_loss = 1.0 - balance_score  # Higher score = lower loss, 1.0 means perfectly balanced

                # Compute other losses
                kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
                ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
                re_loss = F.mse_loss(x_bar, dataset_tensor)

                # Total loss including fairness and balance losses
                loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss + fairness_weight * fairness_loss + balance_weight * balance_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # After each epoch, evaluate the clustering
                with torch.no_grad():
                    _, tmp_q, pred, _ = model(dataset_tensor, adj)
                    y_pred = pred.detach().cpu().numpy().argmax(axis=1)

                    # Number of active clusters (clusters with at least one sample)
                    active_clusters = len(np.unique(y_pred))

                    # Compute silhouette score
                    silhouette_avg = silhouette_score(dataset_tensor.cpu().numpy(), y_pred)

                    # Skip stopping for the first few epochs
                    if epoch > 5:
                        if active_clusters < min_clusters:
                            print(f"Stopping training: Less than {min_clusters} clusters at epoch {epoch}.")
                            break
                        if silhouette_avg < silhouette_threshold:
                            print(f"Stopping training: Silhouette score fell below {silhouette_threshold} at epoch {epoch}.")
                            break

                    # Store metrics
                    metrics = eva(sensitive_attr.cpu().numpy(), y_pred, epoch=epoch, fairness_loss=fairness_loss, X=z.cpu().numpy(), sensitive_attr=sensitive_attr.cpu().numpy())
                    silhouette_scores.append(silhouette_avg)
                    fairness_losses.append(fairness_loss)
                    balances.append(balance_score)
                    purities.append(metrics['purity'])
                    balance_losses.append(balance_loss)

        except Exception as e:
            print(f"Error occurred during training: {e}", flush=True)
            f.write(f"Error occurred during training: {e}\n")
            f.flush()

    # Plot metrics after training
    plot_metrics(silhouette_scores, fairness_losses, balances, purities, balance_losses)
    plot_pca_clusters(z, y_pred, title='Final Cluster Assignment (PCA)')


'''


'''

def train_sdcn_synthetic(dataset, sensitive_attr, n_clusters=2, min_clusters=2, fairness_weight=0.01, balance_weight=0.001, silhouette_threshold=0.1, epochs=50, evaluation_start=10):
    # Pretrain the autoencoder
    pretrain_epochs = 20
    ae_pretrain_path = pretrain_autoencoder(dataset, epochs=pretrain_epochs)

    # Initialize lists to store metric values for each epoch
    silhouette_scores = []
    fairness_losses = []
    balances = []
    purities = []
    balance_losses = []

    try:
        model = SDCN(500, 500, 2000, 2000, 500, 500,
                     n_input=dataset.shape[1],
                     n_z=16,
                     n_clusters=n_clusters,
                     v=1.0).to(device)

        model.ae.load_state_dict(ae_pretrain_path)  # Load pretrained autoencoder weights

        optimizer = Adam(model.parameters(), lr=1e-3)

        adj = torch.eye(dataset.shape[0]).to(device)
        dataset_tensor = torch.Tensor(dataset).to(device)

        # Initialize KMeans clustering
        with torch.no_grad():
            _, _, _, _, z = model.ae(dataset_tensor)
            kmeans = KMeans(n_clusters=n_clusters, n_init=1000)
            y_pred = kmeans.fit_predict(z.cpu().numpy())
            y_pred_last = y_pred
            model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

        # Evaluate the initial clustering performance
        eva(sensitive_attr, y_pred, epoch='Initial', fairness_loss=None, X=z.cpu().numpy(), sensitive_attr=sensitive_attr)

        # Start training loop
        for epoch in range(epochs):
            x_bar, q, pred, z = model(dataset_tensor, adj)
            p = target_distribution(q)

            # Fairness and balance losses
            fairness_loss = demographic_parity_loss(pred, sensitive_attr).item()
            y_pred = pred.detach().cpu().numpy().argmax(axis=1)

            # Calculate balance score
            balance_score = cluster_balance(y_pred, sensitive_attr)
            balance_loss = 1.0 - balance_score  # Higher score = lower loss, 1.0 means perfectly balanced

            # Compute other losses
            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
            ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
            re_loss = F.mse_loss(x_bar, dataset_tensor)

            # Total loss including fairness and balance losses
            loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss + fairness_weight * fairness_loss + balance_weight * balance_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate from the beginning
            with torch.no_grad():
                _, tmp_q, pred, _ = model(dataset_tensor, adj)
                y_pred = pred.detach().cpu().numpy().argmax(axis=1)

                # Number of active clusters (clusters with at least one sample)
                active_clusters = len(np.unique(y_pred))

                # Compute silhouette score
                silhouette_avg = silhouette_score(dataset_tensor.cpu().numpy(), y_pred)

                # Store metrics
                metrics = eva(sensitive_attr, y_pred, epoch=epoch, fairness_loss=fairness_loss, X=z.cpu().numpy(), sensitive_attr=sensitive_attr)
                silhouette_scores.append(silhouette_avg)
                fairness_losses.append(fairness_loss)
                balances.append(balance_score)
                purities.append(metrics['purity'])
                balance_losses.append(balance_loss)

            # Apply stopping criteria only after `evaluation_start` epochs
            if epoch >= evaluation_start:
                if active_clusters < min_clusters:
                    print(f"Stopping training: Less than {min_clusters} clusters at epoch {epoch}.")
                    break

                if silhouette_avg < silhouette_threshold:
                    print(f"Stopping training: Silhouette score fell below {silhouette_threshold} at epoch {epoch}.")
                    break

    except Exception as e:
        print(f"Error occurred during training: {e}")

    # Plot metrics after training
    plot_metrics(silhouette_scores, fairness_losses, balances, purities, balance_losses)
    plot_pca_clusters(z, y_pred, title='Final Cluster Assignment (PCA)')
    plot_tsne_clusters(X, y_pred, sensitive_attr, title='Final t-SNE Cluster Visualization with Sensitive Attribute')


'''


