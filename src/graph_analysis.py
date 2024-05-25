import networkx as nx
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import time

class GraphAnalyzer:
    def __init__(self , config , truth_G , pred_G):
        self.config = config
        self.truth_G = truth_G
        self.pred_G = pred_G
    
    def sampling_after_kde(self, samples, num_samples, bandwidth=0.2, random_seed=42):
        np.random.seed(random_seed)
        
        total_start_time = time.time()
        
        # KDE 拟合开始计时
        fit_start_time = time.time()
        
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        
        kde.fit(samples)
        
        # 打印拟合的时间
        fit_end_time = time.time()
        print(f"KDE fitting took {fit_end_time - fit_start_time:.2f} seconds.")
        
        # 抽样开始计时
        sample_start_time = time.time()
        
        sample_sets = kde.sample(num_samples)  # random samples
        
        # 打印抽样的时间
        sample_end_time = time.time()
        print(f"Sampling took {sample_end_time - sample_start_time:.2f} seconds.")
        
        # 打印总时间
        total_end_time = time.time()
        print(f"Total process took {total_end_time - total_start_time:.2f} seconds.")
        
        return sample_sets
    '''
    def sampling_after_kde( self, samples , num_samples , bandwidths = np.linspace(0.1 , 1.0 , 10) , random_seed = 42):
        np.random.seed(random_seed)
        
        total_start_time = time.time()
        
        # KDE 拟合开始计时
        fit_start_time = time.time()
        
        kde = KernelDensity(kernel = 'gaussian')
        
        params = {'bandwidth': bandwidths}
        grid = GridSearchCV(kde, params, cv=5)
        
        grid.fit(samples)

        # 打印拟合的时间
        fit_end_time = time.time()
        print(f"KDE fitting took {fit_end_time - fit_start_time:.2f} seconds.")
   

        
        kde = grid.best_estimator_
        
        # 抽样开始计时
        sample_start_time = time.time()
        sample_sets = kde.sample(num_samples) # random samples
        
        sample_end_time = time.time()
        print(f"Sampling took {sample_end_time - sample_start_time:.2f} seconds.")

        # 打印总时间
        total_end_time = time.time()
        print(f"Total process took {total_end_time - total_start_time:.2f} seconds.")
        
        return sample_sets
       '''     
     
    
    def get_edge_attributes(self , graph , apply_gene_similarity, apply_AD_weight):
        unique_groups = set(node_data['group'] for _ , node_data in graph.nodes(data = True))
        
        group_to_onehot = {}
        
        for group in unique_groups:
            group_to_onehot[group] = np.array([1 if i == group else 0 for i in unique_groups])
        
        
        samples = []    
        
        for u, v in graph.edges():
            group_u = graph.nodes[u]['group']
            group_v = graph.nodes[v]['group']
            
            encoding = np.zeros(len(unique_groups))
            
            if group_u == group_v:
                encoding = group_to_onehot[group_u]

            if apply_gene_similarity:
                gene_similarity_weight = graph[u][v].get('gene_similarity_weight', 1)
                encoding *= gene_similarity_weight

            if apply_AD_weight:
                ad_weight = graph[u][v].get('ad_weight', 1)
                encoding *= ad_weight
                
            samples.append(encoding)
        
            
        return np.array(samples)
            
            
        
    def analyze_graph(self):
        
        graph_building_time = time.time()
        
        apply_gene_similarity = self.config['graph_builder']['apply_gene_similarity']
        apply_AD_weight = self.config['graph_builder']['apply_AD_weight']
        
        num_samples = len(self.pred_G.edges())
        sample_times = self.config['graph_analysis']['sample_times']        
        
        samples_truth = self.get_edge_attributes(self.truth_G, apply_gene_similarity, apply_AD_weight)
        samples_pred = self.get_edge_attributes(self.pred_G, apply_gene_similarity, apply_AD_weight)

        # 打印 get_edge_attributes 的时间
        print(f"get_edge_attributes took {time.time() - graph_building_time:.2f} seconds.")

        sample_sets_truth = [self.sampling_after_kde(samples_truth , num_samples) for _ in range(sample_times)]
        sample_sets_pred = [self.sampling_after_kde(samples_pred , num_samples) for _ in range(sample_times)]
        
        # 打印
        print(f"Sampling took {time.time() - graph_building_time:.2f} seconds.")
        
        # 转成numpy数组
        # sample_sets_truth = np.array(sample_sets_truth)
        # sample_sets_pred = np.array(sample_sets_pred)
        
        return sample_sets_truth , sample_sets_pred