import json
import sys
from graph_builder import GraphBuilder
import time
from mmd_computation import *
from GK_emd import gaussian_emd
from graph_analysis import GraphAnalyzer


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config


# NOT subtyping

def SGD(config_path):
    
    config = load_config(config_path)

    start_time = time.time()
    
    graph_builder = GraphBuilder(config)

    truth_graph , pred_graph = graph_builder.process_graph()
     
    graph_building_time = time.time()
    
    print(f"Graph Building took {graph_building_time - start_time:.2f} seconds.")
    
    graph_analyzer = GraphAnalyzer(config , truth_G = truth_graph , pred_G = pred_graph)
    
    sample_sets_truth , sample_sets_pred = graph_analyzer.analyze_graph()
    
    sigma = config['kernel_parameters']['sigma']
    
    SGD_score = compute_mmd(sample_sets_truth , 
                            sample_sets_pred ,
                            kernel = gaussian_emd , 
                            sigma = sigma ,
                            is_hist = True)
    
    print("SGD :" , SGD_score)
    
    return SGD_score
    
    
# To do: subtyping     

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    SGD(config_path)