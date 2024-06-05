import networkx as nx
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

class GraphBuilder:
    def __init__(self , config):
        self.config = config
        self.truth_G = nx.Graph()
        self.pred_G = nx.Graph()
        
    def load_data(self , file_path):
        return pd.read_csv(file_path)
    
    def build_graph(self , coordinate_data):
        graph = nx.Graph()
        pos = {}
        for idx, row in coordinate_data.iterrows():
            graph.add_node(idx, group=row['group'])
            pos[idx] = (row['x'], row['y'])
        
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
    def calculate_gene_similarity(self, graph, gene_expression_data):
        expression_matrix = gene_expression_data.iloc[:, 1:].values.astype(float)
        
        normalized_data = (expression_matrix - np.mean(expression_matrix, axis=1, keepdims=True)) / \
                        np.std(expression_matrix, axis=1, keepdims=True)
        
        pearson_matrix = np.corrcoef(normalized_data)
        
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
    def calculate_AD_weight(self , graph):
        for edge in graph.edges():
            u , v = edge
            if graph.nodes[u]['group'] == 'A' and graph.nodes[v]['group'] == 'A':
                graph[u][v]['ad_weight'] = 0.7
            elif graph.nodes[u]['group'] == 'Normal' and graph.nodes[v]['group'] == 'Normal':
                graph[u][v]['ad_weight'] = 0.4
            else:
                graph[u][v]['ad_weight'] = 1

    
    def copy_weights(self, truth_graph, pred_graph):
        for u, v in truth_graph.edges():
            if pred_graph.has_edge(u, v):
                if 'gene_similarity_weight' in truth_graph[u][v]:
                    pred_graph[u][v]['gene_similarity_weight'] = truth_graph[u][v]['gene_similarity_weight']
                if 'ad_weight' in truth_graph[u][v]:
                    pred_graph[u][v]['ad_weight'] = truth_graph[u][v]['ad_weight']
        
        
    def process_graph(self):
        truth_data = self.load_data(self.config['graph_builder']['truth_file_path'])
        pred_data = self.load_data(self.config['graph_builder']['pred_file_path'])
        
        self.truth_G = self.build_graph(truth_data)
        self.pred_G = self.build_graph(pred_data)
        
        if self.config['graph_builder']['apply_gene_similarity']:
            gene_expression_data = self.load_data(self.config['graph_builder']['gene_expression_file_path'])
            self.calculate_gene_similarity(self.truth_G, gene_expression_data)
        
        if self.config['graph_builder']['apply_AD_weight']:
            self.calculate_AD_weight(self.truth_G)
        
            
        self.copy_weights(self.truth_G, self.pred_G) 
    
        return self.truth_G , self.pred_G