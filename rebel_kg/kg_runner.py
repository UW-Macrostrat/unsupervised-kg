from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import math
import torch
from pyvis.network import Network
import os
import random
import numpy
from knowledge_graph import *
import multiprocessing
import argparse
import numpy as np
import pandas as pd

def run_for_file(model, tokenizer, file_path):
    combined_kg = KG()
    with open(file_path, 'r') as reader:
        all_lines = reader.readlines()
    
    # Get the article id
    article_id = os.path.basename(file_path)
    article_id = article_id[ : article_id.rindex(".")]

    # Get the kg for all lines and use it to generate combined kg
    print("Generating kg for file", article_id, "with", len(all_lines), "lines")
    for line in all_lines:
        curr_line = line.strip()
        if len(curr_line) == 0:
            continue

        # Add kg for this line
        line_kg = get_kg_for_line(model, tokenizer, curr_line, article_id)
        combined_kg.merge_with_kb(line_kg)
            
    return combined_kg

# This is run by each process concurrently
def run_for_multiple_files(all_files, share_queue, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    merged_kg = KG()
    for curr_file in all_files:
        file_kg = run_for_file(model, tokenizer, curr_file)
        merged_kg.merge_with_kb(file_kg)
    
    if share_queue is None:
        return merged_kg

    share_queue.put(merged_kg)

def run_for_directory(dir_path, num_process, num_files, model_path):
    # Get the files we want to process
    all_dir_files = []
    for file_name in os.listdir(dir_path):
        if "txt" not in file_name or file_name[0] == '.':
            continue
        
        all_dir_files.append(os.path.join(dir_path, file_name))
    
    # Shuffle the dataset
    random.shuffle(all_dir_files)
    if num_files > 0:
        all_dir_files = all_dir_files[ : num_files]
    print("Processing a total of", len(all_dir_files), "files")

    # Split the files per process 
    num_process = min(num_process, len(all_dir_files))
    files_split = np.array(all_dir_files)
    files_per_process = np.array_split(files_split, num_process)

    # Start a new process for each file
    running_processes = []
    share_queue = multiprocessing.Queue()
    for idx, process_files in enumerate(files_per_process):
        process_files = list(process_files)
        print("Process", idx, "is processing", len(process_files), "files")
        curr_process = multiprocessing.Process(target = run_for_multiple_files, args = (process_files, share_queue, model_path, ))
        curr_process.start()
        running_processes.append(curr_process)
    
    # Combine the kg for all of the files
    all_kg = KG()
    kg_gotten = 0
    while kg_gotten < num_process:
        proc_kg = share_queue.get()
        all_kg.merge_with_kb(proc_kg)
        kg_gotten += 1

    # Join all of the processes
    for curr_process in running_processes:
        curr_process.join()   

    return all_kg 

def read_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--directory', type= str, default = "", help = "The directory containing the text corpus we want to process")
    parser.add_argument('--file', type= str, default = "", help = "The file we want to generate the kg for")
    parser.add_argument('--processes', type = int, default = 1, help = "Number of process we want running")
    parser.add_argument('--num_files', type = int, default = -1, help = "Number of files in the directory we want to save")
    parser.add_argument('--save', type = str, required = True, help = "The html file we want to save the network in")
    parser.add_argument('--model_path', type = str, default = "Babelscape/rebel-large", help = "The model we want to use for generating kg")
    return parser.parse_args()

def save_kg(kg, save_path):
    net = Network(directed=True, width="auto", height="700px", bgcolor="#eeeeee")

    # Create the entities
    color_entity = "#00FF00"
    for e in kg.entities:
        net.add_node(e, shape="circle", color=color_entity)
    
    # Add in the edges
    df_rows = []
    for r in kg.relations:
        head, tail, r_type = r["head"], r["tail"], r["type"]
        sources = r["source"]
        src_articles = list(sources.keys())
        src_articles = ",".join(src_articles)

        # Add in the edge
        net.add_edge(head, tail, title = src_articles, label = r_type)

        # Add in the rows to the csv file
        for article_id in sources:
            for sentence in sources[article_id]:
                df_rows.append([head, r_type, tail, article_id, sentence])
    
    # Save the file
    net.repulsion(
        node_distance=200,
        central_gravity=0.2,
        spring_length=200,
        spring_strength=0.05,
        damping=0.09
    )
    net.set_edge_smooth('dynamic')
    net.show(save_path, notebook=False) 

    # Save the kg as a csv
    csv_save_path = save_path[ : save_path.rindex(".")] + ".csv"
    df = pd.DataFrame(df_rows, columns = ["src", "type", "dst", "article_id", "sentence"])
    df.to_csv(csv_save_path, index = False)

def main():
    args = read_args()
    if ".html" not in args.save:
        raise argparse.ArgumentTypeError('Save path must be a .html file')
    elif len(args.directory) == 0 and len(args.file) == 0:
        raise argparse.ArgumentTypeError('Either a file or directory must be specified')

    if len(args.directory) > 0:
        result_kg = run_for_directory(args.directory, args.processes, args.files, args.model_path)
    else:
        result_kg = run_for_multiple_files([args.file], None, args.model_path)

    save_kg(result_kg, args.save)

if __name__ == "__main__":
    main()