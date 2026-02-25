import os
import weaviate
import time
import json
from nltk.tokenize import sent_tokenize

client = weaviate.Client(
    "http://cosmos0001.chtc.wisc.edu:8080",
    auth_client_secret=weaviate.auth.AuthApiKey(os.getenv("HYBRID_API_KEY")),
)

def send_query(formation_name, num_results):
    where_filter = {"operator": "And", "operands": [{
        "path": ["text_content"],
        "operator": "ContainsAll",
        "valueString": formation_name.strip().split(" "),
    }]}
    result = client.query.get("Passage", ["text_content", "paper_id"]).with_where(where_filter).with_limit(num_results).do()
    matching_paragraphs = []

    if "data" in result and "Get" in result["data"] and "Passage" in result["data"]["Get"] and result["data"]["Get"]["Passage"] is not None:
        paragraphs = result["data"]["Get"]["Passage"]
        for paragraph_content in paragraphs:
            paper_id, paragraph_text = paragraph_content["paper_id"], paragraph_content["text_content"]
            if formation_name.lower() in paragraph_text.lower():
                matching_paragraphs.append({ "paper_id" : paper_id, "paragraph" : paragraph_text})
    
    return matching_paragraphs

def create_weave_cache(save_dir, min_paragraphs_needed = 2, num_results_to_fetch = 50):
    with open("formation_to_process.txt", "r") as reader:
        formation_names = reader.readlines()
    
    os.makedirs(save_dir, exist_ok = True)
    print("Total number of formation names is", len(formation_names), "with", len(os.listdir(save_dir)), "files in dir", save_dir)
    total_formations, num_sucessful, para_sum, para_count = 0, 0, 0, 0
    for name in formation_names:
        # Determine if the file already exists
        formation_name = name.strip()
        save_path = os.path.join(save_dir, formation_name.replace(" ", "_") + ".json")
        total_formations += 1
        if os.path.exists(save_path):
            num_sucessful += 1
            continue
        
        # Get the paraagraphs for this formantion
        all_paragraphs = send_query(formation_name, num_results_to_fetch)
        time.sleep(2)
        if len(all_paragraphs) < min_paragraphs_needed:
            continue
        
        num_sucessful += 1
        para_sum += len(all_paragraphs)
        para_count += 1

        # Save the result
        with open(save_path, 'w+', encoding='utf-8') as writer:
            json.dump({ "matching_paragraphs" : all_paragraphs}, writer, ensure_ascii=False, indent=4)
    
    # Print the metrics
    if total_formations > 0:
        print("Have at least", min_paragraphs_needed, "pargraph for", (100.0 * num_sucessful)/total_formations, "formations")
    
    if para_count > 0:
        print("Average new paragraph length of", para_sum/para_count)
    
if __name__ == "__main__":
    create_weave_cache("formation_sample_paragraphs")