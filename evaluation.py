import numpy as np
from munkres import Munkres, print_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import contingency_matrix

def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro

def cluster_balance(y_pred, sensitive_attr):
    """
    Calculate the balance of clusters based on the sensitive attribute (sex), 
    where 0 represents Red (female) and 1 represents Blue (male).
    
    Args:
    - y_pred: Predicted cluster labels.
    - sensitive_attr: Sensitive attribute (0 for Red, 1 for Blue).
    
    Returns:
    - Average balance across all clusters.
    """
    clusters = np.unique(y_pred)  # Get unique cluster labels
    balance_list = []

    print(f"Unique clusters: {clusters}")  # Debugging

    for cluster in clusters:
        # Get indices of points in this cluster
        cluster_indices = np.where(y_pred == cluster)[0]
        
        # Count Red (0) and Blue (1) within this cluster
        red_count = np.sum(sensitive_attr[cluster_indices] == 0)
        blue_count = np.sum(sensitive_attr[cluster_indices] == 1)

        print(f"Cluster {cluster}: Red count = {red_count}, Blue count = {blue_count}")  # Debugging

        # If either red or blue is 0, set balance to 0 (completely imbalanced)
        if red_count == 0 or blue_count == 0:
            balance = 0.0
        else:
            # Calculate balance as min(red/blue, blue/red)
            balance = min(red_count / blue_count, blue_count / red_count)
        
        print(f"Cluster {cluster}: Balance = {balance}")  # Debugging

        balance_list.append(balance)

    # Return the average balance across all clusters
    overall_balance = np.mean(balance_list)
    print(f"Overall Balance = {overall_balance}")  # Debugging
    return overall_balance
def cluster_balance_generalized(y_pred, sensitive_attr):
    """
    Generalized function to calculate the balance of clusters based on multiple sensitive attributes.
    
    Args:
    - y_pred: Predicted cluster labels (1D array of cluster assignments).
    - sensitive_attr: Sensitive attribute(s) (1D or 2D array). Can be binary (e.g., male/female),
                      or multi-category (e.g., race, ethnicity).
    
    Returns:
    - Overall balance (minimum ratio).
    """
    clusters = np.unique(y_pred)  # Unique cluster labels
    
    # If sensitive_attr is 1D, reshape to 2D for uniformity
    if len(sensitive_attr.shape) == 1:
        sensitive_attr = sensitive_attr.reshape(-1, 1)
    
    # Number of sensitive attributes
    num_sensitive_attrs = sensitive_attr.shape[1]

    overall_min_balance = np.inf  # Initialize the minimum balance to a large value
    
    # Loop through each sensitive attribute (if there are multiple)
    for attr_idx in range(num_sensitive_attrs):
        # Get the current sensitive attribute (e.g., gender, race, etc.)
        attr_values = sensitive_attr[:, attr_idx]
        unique_groups = np.unique(attr_values)  # Unique groups (e.g., Female, Male, different races)

        # Loop over each unique group in the sensitive attribute (e.g., Male, Female, etc.)
        for group in unique_groups:
            # Get the total number of members of this group in the entire dataset
            total_group_count = np.sum(attr_values == group)

            # If no members of this group exist, skip
            if total_group_count == 0:
                continue

            # Now, calculate the balance in each cluster for this group
            group_min_balance = np.inf  # Start with a large value for the minimum balance in this group

            for cluster in clusters:
                # Get indices of points in this cluster
                cluster_indices = np.where(y_pred == cluster)[0]
                
                # Get the count of members of this group in the cluster
                group_in_cluster_count = np.sum(attr_values[cluster_indices] == group)

                # Calculate the ratio: (members of group in cluster) / (total members of group)
                balance = group_in_cluster_count / total_group_count

                # Update the minimum balance for this group
                group_min_balance = min(group_min_balance, balance)

            # Update the overall minimum balance across all groups and clusters
            overall_min_balance = min(overall_min_balance, group_min_balance)

    print(f"Overall Minimum Balance = {overall_min_balance}")  # Debugging output
    return overall_min_balance


def cluster_purity(y_true, y_pred):
    # Contingency matrix
    contingency = contingency_matrix(y_true, y_pred)
    # Sum the maximum values for each cluster
    purity = np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)
    return purity

def eva(y_true, y_pred, epoch=0, fairness_loss=None, X=None, sensitive_attr=None):
    print(f"eva function called for epoch: {epoch}")
    
    # Debug the inputs
    print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    if X is not None:
        print(f"X shape: {X.shape}")
    if sensitive_attr is not None:
        print(f"sensitive_attr shape: {sensitive_attr.shape}")
    
    # Calculate silhouette score
    if X is not None and len(np.unique(y_pred)) > 1:
        silhouette_avg = silhouette_score(X, y_pred)
        print(f"Silhouette Score: {silhouette_avg}")
    else:
        silhouette_avg = -1
        print(f"Silhouette Score skipped for epoch {epoch}: Only one cluster or no data")
    
    # Calculate cluster purity
    purity = cluster_purity(y_true, y_pred)
    print(f"Cluster Purity: {purity}")
    
    # Calculate balance
    if sensitive_attr is not None and len(np.unique(y_pred)) > 1:
        balance = cluster_balance(y_pred, sensitive_attr)
        print(f"Balance: {balance}")
    else:
        balance = -1
        print(f"Balance skipped for epoch {epoch}: Only one cluster or no sensitive attribute")
    
    # Print results for debugging
    if fairness_loss is not None:
        print(f"{epoch}: silhouette {silhouette_avg:.4f}, balance {balance:.4f}, purity {purity:.4f}, fairness_loss {fairness_loss:.4f}")
    else:
        print(f"{epoch}: silhouette {silhouette_avg:.4f}, balance {balance:.4f}, purity {purity:.4f}")

    # Return metrics as a dictionary
    return {
        'silhouette': silhouette_avg,
        'balance': balance,
        'purity': purity
    }





