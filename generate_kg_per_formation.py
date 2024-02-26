from formation_kg_generator import *
import requests
import os
import numpy as np
import json
import multiprocessing
import weaviate
import time

client = weaviate.Client(
    "http://cosmos0001.chtc.wisc.edu:8080",
    auth_client_secret=weaviate.auth.AuthApiKey(os.getenv("HYBRID_API_KEY")),
)

def get_paras_from_weaviate(entity_name, num_results = 30):
    where_filter = {"operator": "And", "operands": [{
        "path": ["text_content"],
        "operator": "ContainsAll",
        "valueString": entity_name.strip().split(" "),
    }]}
    result = client.query.get("Passage", ["text_content", "paper_id"]).with_where(where_filter).with_limit(num_results).do()
    matching_paragraphs = []

    if "data" in result and "Get" in result["data"] and "Passage" in result["data"]["Get"] and result["data"]["Get"]["Passage"] is not None:
        paragraphs = result["data"]["Get"]["Passage"]
        for paragraph_content in paragraphs:
            paper_id, paragraph_text = paragraph_content["paper_id"], paragraph_content["text_content"]
            if entity_name.lower() in paragraph_text.lower():
                matching_paragraphs.append({ "paper_id" : paper_id, "paragraph" : paragraph_text})
    
    return matching_paragraphs

CACHE_DIR = "formation_paragraphs_cache"
def get_paragraphs_for_entity(entity_name):
    entity_file_name = entity_name.replace(" ", "_") + ".json"
    cache_file_path = os.path.join(CACHE_DIR, entity_file_name)

    # If the file exists in cache load it from there
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'r') as reader:
            cached_paragraphs = json.load(reader)
        return cached_paragraphs["matching_paragraphs"]
    
    # Get the paragraphs for this entity and write to cache
    all_paragraphs = get_paras_from_weaviate(entity_name) 
    with open(cache_file_path, 'w+', encoding='utf-8') as writer:
        json.dump({"matching_paragraphs" : all_paragraphs}, writer, ensure_ascii=False, indent=4)
    return all_paragraphs

def process_some_formations(model_types, model_paths, save_dir, overwrite_existing, entities_to_process):
    models_to_use = None

    for entity_name in entities_to_process:
        # Get the current entity
        entity_name = entity_name.strip()
        if len(entity_name) == 0:
            continue
        
        # See if the result already exists
        save_path = os.path.join(save_dir, entity_name.replace(" ", "_") + ".json")
        if os.path.exists(save_path) and not overwrite_existing:
            continue
        
        if models_to_use is None:
            models_to_use = get_models(model_types, model_paths)

        # Get the paragraphs
        all_paragraphs = get_paragraphs_for_entity(entity_name)
        print("Loaded", len(all_paragraphs), "paragraphs for entity", entity_name)
        entity_kg = KG()
        for paragraph_info in all_paragraphs:
            # Pass each paragraph through the model
            paper_id, paragraph_txt = paragraph_info["paper_id"], paragraph_info["paragraph"]
            paragraph_txt = paragraph_txt.replace("\n", " ").strip()
            for model in models_to_use:
                para_kg = get_kg_for_line(model, paragraph_txt, paper_id)
                entity_kg.merge_with_kb(para_kg)

        # Save the results to disk
        print("Saving", len(entity_kg.relations), "relations for entity", entity_name)
        json_to_save = {
            "knowledge_graph" : entity_kg.get_json_representation()
        }
        with open(save_path, 'w+', encoding='utf-8') as writer:
            json.dump(json_to_save, writer, ensure_ascii=False, indent=4)

def load_json_file(json_file):
    with open(json_file, 'r') as reader:
        data = json.load(reader)
    return data

ID_MAP_DIRS = "id_maps"
def get_id_maps():
    lith_id_map = load_json_file(os.path.join(ID_MAP_DIRS, "lith_id_map.json"))
    lith_att_map = load_json_file(os.path.join(ID_MAP_DIRS, "lith_att_id_map.json"))
    strat_name_map = load_json_file(os.path.join(ID_MAP_DIRS, "strat_names_map.json"))
    return {
        "lith_id_map" : lith_id_map,
        "lith_att_map" : lith_att_map,
        "strat_name_map" : strat_name_map
    }

def extract_for_strat_to_lith(relationship, strat_map, lith_map):
    src, dst = relationship["head"], relationship["tail"]
    src_lower, dst_lower = src.lower(), dst.lower()
    
    # See if src in strat or lith map
    perform_swap = False
    if src_lower in strat_map:
        relationship["strat_name_id"] = strat_map[src_lower]
    elif src_lower in lith_map:
        relationship["lith_id"] = lith_map[src_lower]
        perform_swap = True
    
    # See if dst in strat or lith map
    if dst_lower in strat_map and "strat_name_id" not in relationship:
        relationship["strat_name_id"] = strat_map[dst_lower]
        perform_swap = True
    elif dst_lower in lith_map and "lith_id" not in relationship:
        relationship["lith_id"] = lith_map[dst_lower]
    
    if perform_swap:
        relationship["head"] = dst
        relationship["tail"] = src
    
    return "strat_name_id" in relationship or "lith_id" in relationship

def extract_for_lith(relationship, lith_map):
    src, dst = relationship["head"], relationship["tail"]
    src_lower, dst_lower = src.lower(), dst.lower()
    extracted_relationship = False

    if src_lower in lith_map:
        relationship["lith_id"] = lith_map[src_lower]
        extracted_relationship = True
    elif dst_lower in lith_map: 
        relationship["lith_id"] = lith_map[dst_lower]
        extracted_relationship = True
    
    return extracted_relationship

def extract_for_lith_att(relationship, lith_map, lith_att_map):
    # Get the map for this attribute
    att_type = relationship["type"].replace("att_", "").replace("_", " ").lower()
    att_map = lith_att_map[att_type]

    src, dst = relationship["head"], relationship["tail"]
    src_lower, dst_lower = src.lower(), dst.lower()
    extracted_relationship = False
    perform_swap = False

    # See if the src is a lith or a lith attribute
    if src_lower in lith_map:
        relationship["lith_id"] = lith_map[src_lower]
    elif src_lower in att_map:
        relationship["lith_att_id"] = att_map[src_lower]
        perform_swap = True
    
    # See if the dst is a lith or lith attribute
    if dst_lower in lith_map and "lith_id" not in relationship:
        relationship["lith_id"] = lith_map[dst_lower]
        perform_swap = True
    elif dst_lower in att_map and "lith_att_id" not in relationship:
        relationship["lith_att_id"] = att_map[dst_lower]
    
    if perform_swap:
        relationship["head"] = dst
        relationship["tail"] = src
    
    return "lith_id" in relationship or "lith_att_id" in relationship

def process_json_file(run_id, json_file_path, relationship_rows, sources_map, relationship_extracted, ids_maps):
    with open(json_file_path, 'r') as reader:
        data = json.load(reader)
    
    # Get the search strat id
    search_strat_name = os.path.basename(json_file_path).replace("_", " ")
    search_strat_name = search_strat_name[ : search_strat_name.index(".")].strip()
    knowledge_graph = data["knowledge_graph"]

    for relationship in knowledge_graph:
        # See if this relationship should be processed
        relationship_type = relationship["type"]
        extracted_relationship = False
        if relationship_type.startswith("strat_name"):
            extracted_relationship = extract_for_strat_to_lith(relationship, ids_maps["strat_name_map"], ids_maps["lith_id_map"])
        elif relationship_type.startswith("lith"):
            extracted_relationship = extract_for_lith(relationship, ids_maps["lith_id_map"])
        elif relationship_type.startswith("att"):
            extracted_relationship = extract_for_lith_att(relationship, ids_maps["lith_id_map"], ids_maps["lith_att_map"])
        
        if not extracted_relationship:
            continue

        # Record the relationship
        relationship_id = len(relationship_rows) + 1
        relationship["relationship_id"] = relationship_id
        relationship["run_id"] = run_id
        sources = relationship.pop("sources")
        relationship_rows.append(copy.deepcopy(relationship))
    
        # Record the soruces
        for source in sources:
            article_id = source["article_id"]
            for paragraph_txt in source["txt_used"]:
                # Record this source if it doesn't exist
                src_key = (article_id, paragraph_txt, search_strat_name)
                if src_key not in sources_map:
                    sources_map[src_key] = len(sources_map) + 1
                
                # Record the relationship source row
                relationship_extracted.append({
                    "run_id" : run_id,
                    "relationship_id" : relationship_id,
                    "source_id" : sources_map[src_key]
                })

ID_LENGTH = 15
def main(command_args):
    multiprocessing.set_start_method('spawn')
    run_id = str(time.time()).replace(".", "_")
    start_idx = max(0, len(run_id) - ID_LENGTH)
    run_id = run_id[start_idx : ]
    start_time = time.time()

    if not command_args.process_existing:
        # Read the entities we want to proess
        with open(command_args.formation_file, 'r') as reader:
            entity_lines = reader.readlines()
        
        # Split the work across processes
        entities_arr = np.array([name.strip() for name in entity_lines])
        entities_per_process = np.array_split(entities_arr, command_args.num_process)
        os.makedirs(command_args.save_dir, exist_ok = True)

        # Launch the processes
        launched_processes = []
        for curr_process_entities in entities_per_process:
            curr_proc = multiprocessing.Process(target = process_some_formations, args = (command_args.model_types, command_args.model_paths, 
                command_args.save_dir, command_args.overwrite_existing, curr_process_entities))
            curr_proc.start()
            launched_processes.append(curr_proc)
        
        # Wait for them to finish
        [proc.join() for proc in launched_processes]

    # Convert the knowledge graph into dataframes
    id_maps = get_id_maps()
    relationship_rows, sources_map, relationships_extracted = [], {}, []
    for kg_file in os.listdir(command_args.save_dir):
        if "json" not in kg_file or kg_file[0] == '.':
            continue
        
        file_path = os.path.join(command_args.save_dir, kg_file)
        process_json_file(run_id, file_path, relationship_rows, sources_map, relationships_extracted, id_maps)
    
    # Write the relationships dfs
    relationships_save_path = os.path.join(command_args.save_dir, "relationships.csv")
    relationship_df = pd.DataFrame(relationship_rows)
    relationship_df = relationship_df.drop_duplicates()
    print("Writing relationships to", relationships_save_path)
    relationship_df.to_csv(relationships_save_path, index = False)

    # Write the sources dataframe
    sources_rows = []
    start_name_map = id_maps["strat_name_map"]
    for src_key in sources_map:
        article_id, paragraph_txt, search_strat_name = src_key
        
        # Get the start we searched for
        search_name_lower = search_strat_name.lower()
        src_id = sources_map[src_key]
        search_strat_id = -1
        if search_name_lower in start_name_map:
            search_strat_id = start_name_map[search_name_lower]

        sources_rows.append({
            "run_id" : run_id,
            "src_id" : src_id,
            "search_strat_name" : search_strat_name,
            "search_strat_id" : search_strat_id,
            "article_id" : article_id,
            "paragraph_txt" : paragraph_txt,
        })

    sources_save_path = os.path.join(command_args.save_dir, "sources.csv")
    sources_df = pd.DataFrame(sources_rows)
    sources_df = sources_df.drop_duplicates()
    print("Writing sources to", sources_save_path)
    sources_df.to_csv(sources_save_path, index = False)

    # Write the relationships extracted
    extracted_save_path = os.path.join(command_args.save_dir, "relationships_extracted.csv")
    extracted_df = pd.DataFrame(relationships_extracted)
    extracted_df = extracted_df.drop_duplicates()
    print("Writing relationship and sources linking table to", extracted_save_path)
    extracted_df.to_csv(extracted_save_path, index = False)

    # Write the metadata dfs
    metadata_rows = [[run_id, int(start_time), int(time.time()), command_args.run_description]]
    metadata_cols = ["run_id", "start_time", "end_time", "run_description"]
    metadata_df = pd.DataFrame(metadata_rows, columns = metadata_cols)
    metadata_df = metadata_df.drop_duplicates()
    metadata_save_path = os.path.join(command_args.save_dir, "metadata.csv")
    print("Writing metadata to", metadata_save_path)
    metadata_df.to_csv(metadata_save_path, index = False)

def read_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--formation_file', type= str, required = True, help = "The formation we want to generate the kg for")
    parser.add_argument('--process_existing', action = 'store_true', help = "Just process the existing json for KG extraction")
    parser.add_argument('--save_dir', type = str, required = True, help = "The directory path we want to store the results to")
    parser.add_argument('--run_description', type = str, required = True, help = "The description for this generation run")
    parser.add_argument('--num_process', type = int, default = 1, help = "The number of process we want to split the work across")
    parser.add_argument('--model_types', nargs='+', default = ["rebel"], help = "The type of models we want to use")
    parser.add_argument('--overwrite_existing', action='store_true', help = "Should the knowledge graph be regenerated if it already exists")
    parser.add_argument('--model_paths', nargs = '+', default = ["Babelscape/rebel-large"], help = "The path to the model weights we want to use")
    return parser.parse_args()

if __name__ == "__main__":
    main(read_args())