import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import TextBox
import os
import imgkit
import random
import argparse
from kg_runner import *

def generate_dataset(samples_save_dir, dataset_dir, num_trials):
    os.makedirs(samples_save_dir, exist_ok = True)

    # Determine if we have existing trials
    existing_trials = 0
    for file_name in os.listdir(samples_save_dir):
        if file_name[0] == '.' or "txt" not in file_name:
            continue

        existing_trials += 1
    
    if existing_trials >= num_trials:
        return
    
    # Get the files in the dataset
    curr_trial_num = existing_trials
    dataset_files = []
    for file_name in os.listdir(dataset_dir):
        if file_name[0] == '.' or "txt" not in file_name:
            continue
        
        dataset_files.append(os.path.join(dataset_dir, file_name))
    
    # Generate the remaining trials
    while curr_trial_num < num_trials:
        # Read in the lines
        file_path = random.choice(dataset_files)
        with open(file_path, 'r') as reader:
            lines = reader.readlines()
        
        if len(lines) < 2:
            continue
        
        # Determine the lines
        starting_idx = random.randint(0, len(lines) - 1)
        remaining_lines = len(lines) - starting_idx
        num_lines = min(random.randint(1, 6), remaining_lines)

        # Write the trial file
        save_path = os.path.join(samples_save_dir, str(curr_trial_num) + ".txt")
        with open(save_path, 'w+') as writer:
            for line_idx in range(starting_idx, starting_idx + num_lines):
                writer.write(lines[line_idx].strip() + "\n")
        
        curr_trial_num += 1

def get_results_for_samples(sample_dir, model_type, model_path):
    curr_model = get_model(model_type, model_path)
    
    for sample_file in os.listdir(sample_dir):
        if sample_file[0] == '.' or "txt" not in sample_file:
            continue
        
        # Get the paths
        sample_path = os.path.join(sample_dir, sample_file)
        html_save_name = model_type + "_" + sample_file.replace("txt", "html")
        html_save_path = os.path.join(sample_dir, html_save_name)
        image_save_path = html_save_path.replace("html", "jpeg")
        
        # Generate the kg for this sample
        if not os.path.exists(html_save_path):
            sample_kg = run_for_file(curr_model, sample_path)
            save_kg(sample_kg, html_save_path)

def get_better_graph(sample_path, rebel_path, seq2rel_path):
    sample_name = os.path.basename(sample_path)
    sample_id = sample_name[ : sample_name.rindex(".")]

    # Visualize the rebel
    fig, axis = plt.subplots(2, 1, figsize=(12, 5))
    axis[0].imshow(Image.open(rebel_path))
    axis[0].axis('off')
    axis[0].set_title("Rebel KG")

    # Visualize the seq2rel
    axis[1].imshow(Image.open(seq2rel_path))
    axis[1].axis('off')
    axis[1].set_title("Seq2Rel KG")

    fig.suptitle("Comparsion for sample " + str(sample_id))
    plt.tight_layout()
    plt.show()

def get_comparsion(samples_dir):
    # Load the existing results
    existing_results = {}
    results_path = os.path.join(samples_dir, "results.csv")
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        for idx, row in df.iterrows():
            trial_file, result = row["trial"], row["result"]
            existing_results[trial_file] = result
    
    for file_name in os.listdir(samples_dir):
        if file_name[0] == '.' or "txt" not in file_name:
            continue
        
        if file_name in existing_results:
            continue

        sample_path = os.path.join(samples_dir, file_name)
        rebel_path = os.path.join(samples_dir, "rebel_" + file_name.replace("txt", "png"))
        seq2rel_path = os.path.join(samples_dir, "seq2rel_" + file_name.replace("txt", "png"))

        get_better_graph(sample_path, rebel_path, seq2rel_path)
        break 

def read_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--samples_dir', type= str, default = "samples", help = "The directory containing the samples we want to compare models on")
    parser.add_argument('--dataset_dir', type= str, required = True, help = "The directory containing the samples we want to compare models on")
    parser.add_argument('--num_samples', type= int, default = 50, help = "The number of samples we want to compare the model on")
    parser.add_argument('--rebel_path', type= str, required = True, help = "The path to the finetuned rebel model directory")
    parser.add_argument('--seq2rel_path', type= str, required = True, help = "The path to the finetuned seq2rel model")
    return parser.parse_args()

def main():
    args = read_args()
    generate_dataset(args.samples_dir, args.dataset_dir, args.num_samples)
    get_results_for_samples(args.samples_dir, "rebel", args.rebel_path)
    get_results_for_samples(args.samples_dir, "seq2rel", args.seq2rel_path)

if __name__ == "__main__":
    main()