import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, fowlkes_mallows_score, jaccard_score
from scipy.spatial import distance_matrix
import scanpy as sc

def load_adata(data_path):
    adata = sc.read_h5ad(data_path)
    return adata

def compute_clustering_metrics(config):

    adata = load_adata(config['graph_builder']['data_path'])
    
    truth_labels = adata.obs[config['graph_builder']['cell_type_column_name']]
    cluster_labels = adata.obs[config['graph_builder']['cluster_column_name']]
    coords = adata.obsm['spatial']
    
    # External evaluation measures
    ari = adjusted_rand_score(truth_labels, cluster_labels)
    nmi = normalized_mutual_info_score(truth_labels, cluster_labels)
    jaccard = jaccard_score(truth_labels, cluster_labels, average='macro')
    fmi = fowlkes_mallows_score(truth_labels, cluster_labels)
    
    # Internal evaluation measures    
    silhouette = silhouette_score(coords, cluster_labels)
    chaos = compute_CHAOS(cluster_labels, coords)
    PAS = compute_PAS(cluster_labels, coords)
    
    metrics_data = {
            'Adjusted Rand Index (ARI)': ari,
            'Normalized Mutual Information (NMI)': nmi,
            'Jaccard Index': jaccard,
            'Fowlkes-Mallows Index (FMI)': fmi,
            'Silhouette Coefficient': silhouette,
            'CHAOS': chaos,
            'PAS' : PAS
        }
    
    metrics_df = pd.DataFrame(metrics_data , index=[1])
    
    print(metrics_df)
    
    return metrics_df

def compute_CHAOS(cluster_labels, location):
    clusterlabel = np.array(cluster_labels)
    location = np.array(location)
    matched_location = StandardScaler().fit_transform(location)

    clusterlabel_unique = np.unique(clusterlabel)
    dist_val = np.zeros(len(clusterlabel_unique))
    
    count = 0
    for k in clusterlabel_unique:
        location_cluster = matched_location[clusterlabel==k,:]
        if len(location_cluster)<=2:
            continue
        n_location_cluster = len(location_cluster)
        results = [fx_1NN(i,location_cluster) for i in range(n_location_cluster)]
        dist_val[count] = np.sum(results)
        count = count + 1

    return np.sum(dist_val)/len(clusterlabel)

def compute_PAS(cluster_labels, location):
    clusterlabel = np.array(cluster_labels)
    location = np.array(location)
    matched_location = location
    results = [fx_kNN(i,matched_location,k=10,cluster_in=clusterlabel) for i in range(matched_location.shape[0])]
    return np.sum(results)/len(clusterlabel)

def fx_1NN(i,location_in):
        location_in = np.array(location_in)
        dist_array = distance_matrix(location_in[i,:][None,:],location_in)[0,:]
        dist_array[i] = np.inf
        return np.min(dist_array)
    
def fx_kNN(i,location_in,k,cluster_in):
    location_in = np.array(location_in)
    cluster_in = np.array(cluster_in)
    dist_array = distance_matrix(location_in[i,:][None,:],location_in)[0,:]
    dist_array[i] = np.inf
    ind = np.argsort(dist_array)[:k]
    cluster_use = np.array(cluster_in)
    if np.sum(cluster_use[ind]!=cluster_in[i])>(k/2):
        return 1
    else:
        return 0
    
