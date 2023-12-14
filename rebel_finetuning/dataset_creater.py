import os
import pandas as pd
import numpy as np
import argparse

def create_tokenizer(graph_path):
    save_path = 'data/archive_token_terms.txt'
    if not os.path.exists(save_path):
        database_terms_df = pd.read_csv(graph_path)
        database_terms_df.dropna()
        unique_terms = set(database_terms_df["edge_src"]).union(set(database_terms_df["edge_dst"]))
        terms_arr = np.array(list(unique_terms))
        np.savetxt(save_path, terms_arr, delimiter=" ", fmt="%s") 

def create_dataset_using_snippets(graph_path):
    graph_df = pd.read_csv(graph_path)
    graph_df = graph_df.head(20)
    
    for edge_name, rows in graph_df.groupby('edge_name'):
        print(edge_name, len(rows.index))

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_file', type= str, required = True, help = "The path to the graph file")
    return parser.parse_args()

if __name__ == "__main__":
    args = read_args()
    graph_path = args.graph_file

    create_tokenizer(graph_path)
    create_dataset_using_snippets(graph_path)