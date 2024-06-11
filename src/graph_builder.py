import networkx as nx
import pandas as pd
import numpy as np
import anndata as ad
from sklearn.neighbors import NearestNeighbors

from scipy.sparse import isspmatrix, csr_matrix 
class DataHandler:
    def __init__(self,file_path):
        self.file_path = file_path
        self.adata = None

    def load_data(self):
        if self.adata is None:
            self.adata = ad.read_h5ad(self.file_path)
        return self.adata

class GraphBuilder:
    def __init__(self , config):
        self.config = config
        self.data_handler = DataHandler(self.config['graph_builder']['data_path'])
        self.truth_G = nx.Graph()
        self.pred_G = nx.Graph()

    def build_graph(self , coordinate_data , label_data):
        graph = nx.Graph()
        pos = {}
        for i, (index, row) in enumerate(coordinate_data.iterrows()):
            pos[i] = (row['x'], row['y'])  # Store position with node index as key
            graph.add_node(i, pos=pos[i], group=label_data.iloc[i])
        
        pos_array = np.array(list(pos.values()))
        num_nbrs = self.config['graph_builder']['num_neighbors'] + 1
        nbrs = NearestNeighbors(n_neighbors=num_nbrs)
        nbrs.fit(pos_array)
        _ , indices = nbrs.kneighbors(pos_array)
        
        # build KNN graph
        for i , neighbors in enumerate(indices):
            for n in neighbors[1:]:
                graph.add_edge(i , n)
        return graph

    ## calculate Gene Similarity with 2 vector 
    def calculate_gene_similarity(self, graph, gene_expression_matrix):
        if isspmatrix(gene_expression_matrix): 
           gene_expression_matrix = gene_expression_matrix.toarray()
        pearson_matrix = np.corrcoef(gene_expression_matrix)
        
        nodes = list(graph.nodes())
        if len(nodes) != pearson_matrix.shape[0]:
            raise ValueError("The number of nodes ")
        
        for u, v in graph.edges():
            
            group_u = graph.nodes[u]['group']
            group_v = graph.nodes[v]['group'] 
            
            # same group use similarity
            if group_u == group_v:
                graph.edges[u, v]['gene_similarity_weight'] = pearson_matrix[u, v]
           
            # different group use distance 
            else:
                graph.edges[u, v]['gene_similarity_weight'] = 1 - pearson_matrix[u, v]
            
        
    # a hyperparameter
    def calculate_anomaly_weight(self , graph , dict_severity_levels):
        severity_mapping = {category['name']: category['severity_level'] for category in dict_severity_levels}
        
        for u ,v in graph.edges():
            group_u = graph.nodes[u]['group']
            group_v = graph.nodes[v]['group']
            
            if group_u == group_v:
                graph.edges[u, v]['anomaly_severity_weight'] = 0.5 + 0.1 * severity_mapping[group_u]
           
            else:
                graph.edges[u, v]['anomaly_severity_weight'] = 1
    
    def copy_weights(self, truth_graph, pred_graph):
        for u, v in truth_graph.edges():
            if pred_graph.has_edge(u, v):
                if 'gene_similarity_weight' in truth_graph[u][v]:
                    pred_graph[u][v]['gene_similarity_weight'] = truth_graph[u][v]['gene_similarity_weight']
                if 'ad_weight' in truth_graph[u][v]:
                    pred_graph[u][v]['anomaly_severity_weight'] = truth_graph[u][v]['anomaly_severity_weight']
        
    def process_graph(self):
        adata = self.data_handler.load_data()
        
        coordinate_data  = pd.DataFrame({
            'x' : adata.obsm['spatial'][: , 0],
            'y' : adata.obsm['spatial'][: , 1]
        })
        
        truth_label = adata.obs[self.config['graph_builder']['cell_type_column_name']]
        cluster_label = adata.obs[self.config['graph_builder']['cluster_column_name']]
        
        self.truth_G = self.build_graph(coordinate_data , truth_label)
        self.pred_G = self.build_graph(coordinate_data , cluster_label)
        
        if self.config['graph_builder']['apply_gene_similarity']:     
            gene_expression_matrix = adata.X 
            self.calculate_gene_similarity(self.truth_G, gene_expression_matrix)
        
        if self.config['graph_builder']['apply_anomaly_severity_weight']:
            self.calculate_anomaly_weight(self.truth_G , self.config['graph_builder']['severity_levels'])
        
        self.copy_weights(self.truth_G, self.pred_G) 
    
        return self.truth_G , self.pred_G