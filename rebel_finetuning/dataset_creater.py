import os
import pandas as pd
import numpy as np
import argparse
import requests
from multiprocessing import Pool
import time
import random

def create_tokenizer(graph_path):
    save_path = 'data/archive/archive_token_terms.txt'
    if not os.path.exists(save_path):
        database_terms_df = pd.read_csv(graph_path)
        database_terms_df.dropna()
        unique_terms = set(database_terms_df["edge_src"]).union(set(database_terms_df["edge_dst"]))
        terms_arr = np.array(list(unique_terms))
        np.savetxt(save_path, terms_arr, delimiter=" ", fmt="%s") 

def get_relationships_for_edge(src_edge, dst_edge, edge_type):
    try:
        src_name = src_edge.replace(" ", "%20")
        dst_name = dst_edge.replace(" ", "%20")
        request_url = f"https://xdd.wisc.edu/api/snippets?term={src_name}%2C{dst_name}&inclusive=true&clean=true"
        time.sleep(0.5)
        response = requests.get(request_url)
        relations_json = response.json()
        all_pairs = []
        if 'success' in relations_json and 'data' in relations_json['success']: 
            data = relations_json['success']['data']
            for document in data:
                doc_title = document["title"]
                doc_id = document["_gddid"]
                for curr_sentence in document["highlight"]:
                    for piece in curr_sentence.split("."):
                        piece = piece.strip()
                        if src_edge in piece and dst_edge in piece:
                            all_pairs.append([doc_id, doc_title, piece, src_edge, dst_edge, edge_type])
        return all_pairs
    except Exception as e:
        print("Encountered error", e, "for pair", src_edge, "<->", dst_edge)
        return []

def save_rows_as_df(file_id, results_row):
    save_path = f"data/archive/{file_id}.csv"
    save_df = pd.DataFrame(results_row, columns = ["doc_id", "title", "text", "src", "dst", "type"])
    print("Saving file", save_path, "with", len(save_df.index), "rows")
    save_df.to_csv(save_path, index = None, sep = '\t')

def create_dataset_using_snippets(graph_path, num_workers = 5, pairs_per_file = 1000):
    if os.path.exists("data/archive/0.csv"):
        return

    graph_df = pd.read_csv(graph_path).astype(str)
    file_id = 0
    results_row, prev_count = [], 0

    with Pool(num_workers) as pool:
        results = []
        print("DF has a total of", len(graph_df.index), "rows")
        for _, row in graph_df.iterrows():
            src_txt = str(row["edge_src"])
            dst_txt = str(row["edge_dst"])
            edge_type = str(row["edge_name"])
            if src_txt != 'nan' and dst_txt != 'nan' and edge_type != 'nan':
                results.append(pool.apply_async(get_relationships_for_edge, [src_txt, dst_txt, edge_type]))
        print("Expecting results for", len(results), "rows")

        for result in results:
            pairs = result.get()
            results_row.extend(pairs)
            if (len(results_row) > prev_count):
                print("Incremented count for file", file_id, "to", len(results_row))
                prev_count = len(results_row)

            if len(results_row) >= pairs_per_file:
                save_rows_as_df(file_id, results_row)
                file_id += 1
                results_row = []
                prev_count = 0
    
    # Save the last chunk
    save_rows_as_df(file_id, results_row)

def write_split_to_file(split_files, save_path):
    print("Writing split to", save_path)
    with open(save_path, 'w+') as writer:
        for file_name in split_files:
            writer.write(file_name + "\n")

def create_splits():
    archive_dir = "data/archive"
    train_path = os.path.join(archive_dir, "train.txt")
    test_path = os.path.join(archive_dir, "test.txt")
    valid_path = os.path.join(archive_dir, "valid.txt")

    print(train_path, test_path, valid_path)
    if os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(valid_path):
        return True
    
    # Perorm the split
    all_data_files = []
    for file_name in os.listdir(archive_dir):
        if file_name[0] != '.' and "csv" in file_name:
            all_data_files.append(file_name)
    random.shuffle(all_data_files)

    # Create the split
    num_train, num_test = int(0.75 * len(all_data_files)), int(0.15 * len(all_data_files))
    train_files = all_data_files[ : num_train]
    test_files = all_data_files[num_train : num_train + num_test]
    valid_files = all_data_files[num_train + num_test : ]

    # Write split out to file
    write_split_to_file(train_files, train_path)
    write_split_to_file(test_files, test_path)
    write_split_to_file(valid_files, valid_path)

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_file', type= str, required = True, help = "The path to the graph file")
    return parser.parse_args()

if __name__ == "__main__":
    args = read_args()
    graph_path = args.graph_file

    create_tokenizer(graph_path)
    create_dataset_using_snippets(graph_path)
    create_splits()