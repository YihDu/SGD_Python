import networkx as nx
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class GraphAnalyzer:
    def __init__(self , config , truth_G , pred_G):
        self.config = config
        self.truth_G = truth_G
        self.pred_G = pred_G
    
    def fit_kde_and_sample(self, samples, num_samples , sample_times , bandwidth=0.2, random_seed=42):
        
        fit_start_time = time.time()
        
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        
        kde.fit(samples)
        
        fit_end_time = time.time()
        print(f"KDE fitting took {fit_end_time - fit_start_time:.2f} seconds.")
        
        sample_start_time = time.time()
        
        samples_set = []
        
        for i in range(sample_times):  
            sampled = kde.sample(num_samples ,random_state=random_seed + i)  
            sampled = np.clip(sampled, 0, 1)       
            samples_set.append(sampled)
        
        sample_end_time = time.time()
        print(f"Sampling took {sample_end_time - sample_start_time:.2f} seconds.")
    
        self.plot_marginal_distributions(samples, samples_set)
        
        return samples_set
    
    def plot_marginal_distributions(self, original_samples, samples_set):
        num_dimensions = original_samples.shape[1]
        fig, axes = plt.subplots(num_dimensions, 2, figsize=(16, 6 * num_dimensions))
        
        for i in range(num_dimensions):
            # original
            ax_kde = axes[i, 0] if num_dimensions > 1 else axes[0]
            sns.histplot(original_samples[:, i], bins=30, kde=False, label='Original Histogram', color='blue', alpha=0.5, ax=ax_kde)
            sns.kdeplot(original_samples[:, i], fill=True, label='Original KDE', color='blue', ax=ax_kde)
            ax_kde.set_title(f'KDE and Histogram of Dimension {i+1}')
            ax_kde.set_xlabel(f'Dimension {i+1}')
            ax_kde.set_ylabel('Density')
            ax_kde.legend()
            
            # sample
            ax_hist = axes[i, 1] if num_dimensions > 1 else axes[1]
            for j, samples in enumerate(samples_set):
                sns.histplot(samples[:, i], bins=30, kde=False, alpha=0.3, label=f'Sample {j+1}', ax=ax_hist)
            ax_hist.set_title(f'Sampled Histograms of Dimension {i+1}')
            ax_hist.set_xlabel(f'Dimension {i+1}')
            ax_hist.set_ylabel('Density')
            
            
            if i == 0:
                ax_hist.legend()
        
        plt.tight_layout()
        plt.show()
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
     
    
    def get_edge_attributes(self, graph, apply_gene_similarity, apply_AD_weight , is_multi = False):
        unique_groups = sorted(set(node_data['group'] for _, node_data in graph.nodes(data=True)))
        print("Unique groups:", unique_groups)
        group_to_onehot = {group: np.array([1 if i == group else 0 for i in unique_groups], dtype=np.float64) for group in unique_groups}
        
        
        
        if is_multi:
            samples = {group : [] for group in unique_groups}
        
        else:
            samples = []
        
        for u, v in graph.edges():
            group_u = graph.nodes[u]['group']
            group_v = graph.nodes[v]['group']
            
            if group_u == group_v:
                encoding = group_to_onehot[group_u].copy()
            else:
                encoding = np.zeros(len(unique_groups), dtype=np.float64)
            
            if apply_gene_similarity:
                gene_similarity_weight = graph[u][v].get('gene_similarity_weight', 1.0)
                encoding *= gene_similarity_weight

            if apply_AD_weight:
                ad_weight = graph[u][v].get('ad_weight', 1.0)
                encoding *= ad_weight
            
            if is_multi:
                samples[group_u].append(encoding)
                if group_u != group_v:
                    samples[group_v].append(encoding)
            
            else:    
                samples.append(encoding)
       
        if is_multi:
            return {group: np.array(encodings) for group, encodings in samples.items()}
        
        else: 
            return np.array(samples)
        
    def analyze_graph(self):
        
        graph_building_time = time.time()
        
        apply_gene_similarity = self.config['graph_builder']['apply_gene_similarity']
        apply_AD_weight = self.config['graph_builder']['apply_AD_weight']
        
        num_samples = len(self.pred_G.edges())
        sample_times = self.config['graph_analysis']['sample_times']        
        
        is_multi = self.config['is_multi']
        
        samples_truth = self.get_edge_attributes(self.truth_G, apply_gene_similarity, apply_AD_weight , is_multi = is_multi)
        samples_pred = self.get_edge_attributes(self.pred_G, apply_gene_similarity, apply_AD_weight , is_multi = is_multi)
        
        print(f"get_edge_attributes took {time.time() - graph_building_time:.2f} seconds.")

        samples_set_truth = self.fit_kde_and_sample(samples_truth, num_samples , sample_times , bandwidth=0.1, random_seed=42)
        samples_set_pred = self.fit_kde_and_sample(samples_pred, num_samples , sample_times , bandwidth=0.1, random_seed=42)
        
        return samples_set_truth , samples_set_pred