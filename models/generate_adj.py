import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp

from sklearn.metrics.pairwise import haversine_distances


def geographical_distance(x=None, to_rad=True):
    _AVG_EARTH_RADIUS_KM = 6371.0088

    # Extract values of X if it is a DataFrame, else assume it is 2-dim array of lat-lon pairs
    latlon_pairs = x.values if isinstance(x, pd.DataFrame) else x

    # If the input values are in degrees, convert them in radians
    if to_rad:
        latlon_pairs = np.vectorize(np.radians)(latlon_pairs)

    distances = haversine_distances(latlon_pairs) * _AVG_EARTH_RADIUS_KM

    # Cast response
    if isinstance(x, pd.DataFrame):
        res = pd.DataFrame(distances, x.index, x.index)
    else:
        res = distances

    return res


def thresholded_gaussian_kernel(x, theta=None, threshold=None, threshold_on_input=False):
    if theta is None:
        theta = np.std(x)
    weights = np.exp(-np.square(x / theta))
    if threshold is not None:
        mask = x > threshold if threshold_on_input else weights < threshold
        weights[mask] = 0.
    return weights


def get_similarity_AQI(dist, thr=0.1, include_self=False, force_symmetric=False, sparse=False):
    theta = np.std(dist[:36, :36])  # use same theta for both air and air36
    adj = thresholded_gaussian_kernel(dist, theta=theta, threshold=thr)
    if not include_self:
        adj[np.diag_indices_from(adj)] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj


def get_adj_AQI36():
    df = pd.read_csv("./data/pm25/SampleData/pm25_latlng.txt")
    df = df[['latitude', 'longitude']]
    res = geographical_distance(df, to_rad=False).values
    adj = get_similarity_AQI(res)
    return adj



def get_similarity_NACSE(dist, thr=0.1, include_self=False, force_symmetric=False, sparse=False):
    theta = np.std(dist)  # use same theta for both air and air36
    adj = thresholded_gaussian_kernel(dist, theta=theta, threshold=thr)
    if not include_self:
        adj[np.diag_indices_from(adj)] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj

def get_similarity_AWN(dist, thr=0.1, include_self=False, force_symmetric=False, sparse=False):
    theta = np.std(dist)  # use same theta for both air and air36
    adj = thresholded_gaussian_kernel(dist, theta=theta, threshold=thr)
    if not include_self:
        adj[np.diag_indices_from(adj)] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj

def get_adj_awn(total_stations=67):
    train_locs = np.load('./data/nacse/zone_8_train_locs.npy')
    test_locs = np.load('./data/nacse/zone_8_test_locs.npy')
    locations = np.zeros((total_stations * 2, 2))
    # print(f"train locs: {train_locs.shape}")
    for i in range(total_stations):
        if i < train_locs.shape[0]:
            locations[2*i] = train_locs[i, :2]
            locations[2*i+1] = train_locs[i, :2]
        else:
            locations[2*i] = test_locs[i - train_locs.shape[0], :2]
            locations[2*i+1] = test_locs[i - train_locs.shape[0], :2]
    # locations[:train_locs.shape[0], :] = train_locs[:, :2]
    # locations[train_locs.shape[0]:, :] = test_locs[:, :2]
    res = geographical_distance(locations)
    adj = get_similarity_AWN(res)
    return adj

def get_adj_nacse(total_stations=179):
    train_locs = np.load('./data/nacse/X_OR_temps_train_loc.npy')
    test_locs = np.load('./data/nacse/X_OR_temps_test_loc.npy')
    locations = np.zeros((total_stations * 2, 2))
    # print(f"train locs: {train_locs.shape}")
    for i in range(total_stations):
        if i < train_locs.shape[0]:
            locations[2*i] = train_locs[i, :2]
            locations[2*i+1] = train_locs[i, :2]
        else:
            locations[2*i] = test_locs[i - train_locs.shape[0], :2]
            locations[2*i+1] = test_locs[i - train_locs.shape[0], :2]
    # locations[:train_locs.shape[0], :] = train_locs[:, :2]
    # locations[train_locs.shape[0]:, :] = test_locs[:, :2]
    res = geographical_distance(locations)
    adj = get_similarity_NACSE(res)
    return adj

def get_similarity_metrla(thr=0.1, force_symmetric=False, sparse=False):
    dist = np.load('./data/metr_la/metr_la_dist.npy')
    finite_dist = dist.reshape(-1)
    finite_dist = finite_dist[~np.isinf(finite_dist)]
    sigma = finite_dist.std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < thr] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj


def get_similarity_pemsbay(thr=0.1, force_symmetric=False, sparse=False):
    dist = np.load('./data/pems_bay/pems_bay_dist.npy')
    finite_dist = dist.reshape(-1)
    finite_dist = finite_dist[~np.isinf(finite_dist)]
    sigma = finite_dist.std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < thr] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj


# in Graph-wavenet
def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def compute_support_gwn(adj, device=None):
    adj_mx = [asym_adj(adj), asym_adj(np.transpose(adj))]
    support = [torch.tensor(i).to(device) for i in adj_mx]
    return support