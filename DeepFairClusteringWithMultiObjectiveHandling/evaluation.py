import numpy as np
from munkres import Munkres, print_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear
from sklearn import metrics


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


from sklearn.metrics import silhouette_score

def eva(y_true, y_pred, features, sensitive_attrs, epoch=0):
    # Calculate cluster accuracy and F1
    acc, f1 = cluster_acc(y_true, y_pred)
    
    # Calculate NMI and ARI
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    
    # Calculate silhouette score
    if len(np.unique(y_pred)) > 1:  # Ensure there is more than one cluster to calculate silhouette score
        sil_score = silhouette_score(features, y_pred)
    else:
        sil_score = -1  # Default score if only one cluster
    
    # Calculate balance
    balance = cluster_balance_combinations(y_pred, sensitive_attrs)
    
    # Print all metrics
    print(epoch, ': silhouette {:.4f}'.format(sil_score), ', balance {:.4f}'.format(balance))



from sklearn.metrics import silhouette_score

def cluster_balance_combinations(y_pred, sensitive_attrs):
    """
    Function to calculate the balance of clusters based on combinations of sensitive attributes.
    
    Args:
    - y_pred: Predicted cluster labels (1D array of cluster assignments).
    - sensitive_attrs: Sensitive attributes (2D array). Each column represents a sensitive attribute.
                       For example, columns could represent sex and race.
    
    Returns:
    - Overall balance (minimum ratio).
    """
    clusters = np.unique(y_pred)  # Unique cluster labels
    #print(f"Unique clusters: {clusters}")

    # If sensitive_attrs is 1D, reshape to 2D for uniformity
    if len(sensitive_attrs.shape) == 1:
        sensitive_attrs = sensitive_attrs.reshape(-1, 1)
    
    # Get all unique combinations of sensitive attribute values (e.g., Female-Black, Male-White)
    unique_combinations = np.unique(sensitive_attrs, axis=0)
    #print(f"Unique combinations of sensitive attributes:\n{unique_combinations}")
    
    overall_min_balance = np.inf  # Initialize the minimum balance to a large value
    
    # Loop over each unique combination of sensitive attribute values
    for combination in unique_combinations:
        # Get the total number of members of this combination in the entire dataset
        combination_mask = np.all(sensitive_attrs == combination, axis=1)
        total_combination_count = np.sum(combination_mask)

        #print(f"Combination: {combination}, Total count in dataset: {total_combination_count}")

        # If no members of this combination exist, skip
        if total_combination_count == 0:
            #print(f"Combination {combination} has no members, skipping...")
            continue

        # Calculate the balance within each cluster for this combination
        combination_min_balance = np.inf  # Start with a large value for the minimum balance in this combination

        for cluster in clusters:
            # Get indices of points in this cluster
            cluster_indices = np.where(y_pred == cluster)[0]
            
            # Count members of the combination in the cluster
            combination_in_cluster_count = np.sum(combination_mask[cluster_indices])

            # Calculate the ratio: (members of combination in cluster) / (total members of combination)
            balance = combination_in_cluster_count / total_combination_count

            #print(f"Cluster: {cluster}, Combination in cluster count: {combination_in_cluster_count}, "
                  #f"Balance for this combination in cluster: {balance:.4f}")

            # Update the minimum balance for this combination
            combination_min_balance = min(combination_min_balance, balance)

        #print(f"Minimum balance for combination {combination}: {combination_min_balance:.4f}")

        # Update the overall minimum balance across all combinations and clusters
        overall_min_balance = min(overall_min_balance, combination_min_balance)
    
    #print(f"Overall minimum balance across all combinations and clusters: {overall_min_balance:.4f}")

    return overall_min_balance

'''
def eva(y_true, y_pred, sensitive_attrs, epoch=0):
    # Calculate the cluster balance using race and sex
    balance = cluster_balance_combinations(y_pred, sensitive_attrs)
    
    # Calculate the silhouette score
    silhouette = silhouette_score(y_true.reshape(-1, 1), y_pred)
    
    # For now, we are not using other metrics like balance loss
    print(f'Epoch {epoch}: balance {balance:.4f}, silhouette score {silhouette:.4f}')
'''