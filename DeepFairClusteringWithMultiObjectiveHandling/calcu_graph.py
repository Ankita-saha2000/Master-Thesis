import pandas as pd
import numpy as np
import h5py
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
'''
topk = 10

def construct_graph(features, label, method='heat'):
    fname = 'graph/hhar10_graph.txt'
    num = len(label)
    dist = None

    if method == 'heat':
        dist = -0.5 * pair(features) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk+1))[-(topk+1):]
        inds.append(ind)

    f = open(fname, 'w')
    counter = 0
    A = np.zeros_like(dist)
    for i, v in enumerate(inds):
        mutual_knn = False
        for vv in v:
            if vv == i:
                pass
            else:
                if label[vv] != label[i]:
                    counter += 1
                f.write('{} {}\n'.format(i, vv))
    f.close()
    print('error rate: {}'.format(counter / (num * topk)))

'''
'''
f = h5py.File('data/usps.h5', 'r')
train = f.get('train')
test = f.get('test')
X_tr = train.get('data')[:]
y_tr = train.get('target')[:]
X_te = test.get('data')[:]
y_te = test.get('target')[:]
f.close()
usps = np.concatenate((X_tr, X_te)).astype(np.float32)
label = np.concatenate((y_tr, y_te)).astype(np.int32)
'''

'''
hhar = np.loadtxt('data/hhar.txt', dtype=float)
label = np.loadtxt('data/hhar_label.txt', dtype=int)


reut = np.loadtxt('data/reut.txt', dtype=float)
label = np.loadtxt('data/reut_label.txt', dtype=int)



construct_graph(hhar, label, 'ncos')
'''
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances as pair
from sklearn.preprocessing import normalize, OneHotEncoder, StandardScaler

topk = 10

def construct_graph(features, label, method='heat'):
    fname = 'graph/synthetic_adult_graph.txt'
    num = len(label)
    dist = None

    if method == 'heat':
        dist = -0.5 * pair(features) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk+1))[-(topk+1):]
        inds.append(ind)

    with open(fname, 'w') as f:
        counter = 0
        for i, v in enumerate(inds):
            for vv in v:
                if vv != i and label[vv] != label[i]:
                    counter += 1
                f.write('{} {}\n'.format(i, vv))
    print('error rate: {}'.format(counter / (num * topk)))

# Load the synthetic Adult-like dataset
synthetic_adult_data = pd.read_csv('data/dummy.csv')

# Step 1: Preprocess the data
# One-hot encode categorical features
categorical_features = ['Education', 'Occupation', 'Sex', 'Race']
one_hot_encoder = OneHotEncoder(sparse_output=False)  # Corrected the argument
categorical_transformed = one_hot_encoder.fit_transform(synthetic_adult_data[categorical_features])

# Standardize numerical features
numerical_features = ['Age', 'HoursPerWeek']
scaler = StandardScaler()
numerical_transformed = scaler.fit_transform(synthetic_adult_data[numerical_features])

# Combine processed features
processed_features = np.hstack([numerical_transformed, categorical_transformed])

# Extract labels (e.g., 'Income')
labels = synthetic_adult_data['Income'].values

# Construct the graph using the 'ncos' method
construct_graph(processed_features, labels, 'ncos')

