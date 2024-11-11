import os
import json

def perform_extraction(file_name, save_dir):
    os.makedirs(save_dir, exist_ok = True)

    # Load the data
    with open(file_name, "r") as reader:
        data = json.load(reader)
    
    # Save each example as its own file
    save_idx = 0
    for curr_result in data:
        save_path = os.path.join(save_dir, str(save_idx) + ".json")
        with open(save_path, 'w+') as writer:
            json.dump(curr_result, writer, indent = 4)

        save_idx += 1

if __name__ == "__main__":
    perform_extraction("weaviate_sample_outputs.json", "weaviate_example_requests")