from coref_resolution import *
from ner_extractor import *
from re_extractor import *
from huggingface_hub import Repository
import datetime

class TreeGenerator:

    def __init__(self, data_dir):
        self.ner_extractor = NERExtractor(data_dir)
        self.coref_resolver = CorefResolver()
        self.re_extractor = RE_Extractor(data_dir)
        self.start_prefixes = ["strat", "lith", "att"]
    
    def get_rock_level(self, rock):
        rock_type = rock.term_type
        for idx, prefix in enumerate(self.start_prefixes):
            if rock_type.startswith(prefix):
                return idx

        raise Exception("Invalid term type of " + rock_type)
    
    def perform_level_merge(self, lower_level, terms_by_level, sentence_words, sentence_spans):
        upper_level = lower_level - 1
        while upper_level > 0 and upper_level not in terms_by_level:
            upper_level -= 1
        
        # The upper level doesn't exist so can't perform merge
        if upper_level not in terms_by_level:
            return
        
        lower_terms, lower_prefix = terms_by_level[lower_level], self.start_prefixes[lower_level]
        upper_terms, upper_prefix = terms_by_level[upper_level], self.start_prefixes[upper_level]

        for lower_term in lower_terms:
            # Determine term to merge with
            upper_merge_idx, upper_merge_probability = -1, -1.0
            for upper_idx, upper_term in enumerate(upper_terms):
                pair_probability = self.re_extractor.get_relationship_probability(sentence_words, lower_term, upper_term, lower_prefix, upper_prefix, sentence_spans)
                if pair_probability > upper_merge_probability:
                    upper_merge_idx = upper_idx
                    upper_merge_probability = pair_probability

            # Perform the merge
            if upper_merge_idx >= 0:
                upper_terms[upper_merge_idx].add_child(lower_term, child_probability = upper_merge_probability)

    def format_relationships(self, paragraph, entity, relationships, just_entities):
       # We have a leaft node
        if len(entity.children) == 0:
            # Only record the leaf strats
            if entity.term_type.startswith("strat"):
                entity_range = entity.text_ranges[0]
                just_entities.append({
                    "entity" : paragraph[entity_range[0] : entity_range[1]],
                    "entity_type" : entity.term_type
                })
            return

        # Determine the entity type
        src_range = entity.text_ranges[0]
        src_txt = paragraph[src_range[0] : src_range[1]].strip()
        relationship_type = ""
        if entity.term_type.startswith("strat"):
            relationship_type = "strat_to_lith"
        elif entity.term_type.startswith("lith"):
            relationship_type = "att_of_lith"
        else:
            raise Exception("No relationship type for entity type", entity.term_type)

        for _, child in entity.children:
            # Record this relationship
            dst_range = child.text_ranges[0]
            dst_txt = paragraph[dst_range[0] : dst_range[1]].strip()
            relationships.append({
                "src" : src_txt,
                "relationship_type" : relationship_type,
                "dst" : dst_txt
            })

            self.format_relationships(paragraph, child, relationships, just_entities)

    def format_result(self, paragraph, entities):
        relationships, just_entities = [], []
        for entity in entities:
            self.format_relationships(paragraph, entity, relationships, just_entities)
        return relationships, just_entities
        
    def run_for_text(self, text):
        # Get the rock terms
        sentence, sentence_words, rock_terms, sentence_spans, word_ranges = self.ner_extractor.extract_terms(text)
        self.coref_resolver.record_cooref_occurences(sentence_words, word_ranges, rock_terms)

        # Get rocks by level
        rocks_by_level = {}
        for curr_rock in rock_terms:
            rock_level = self.get_rock_level(curr_rock)
            if rock_level not in rocks_by_level:
                rocks_by_level[rock_level] = []
            rocks_by_level[rock_level].append(curr_rock)        
        
        # Perform the merging
        for lower_level in range(len(self.start_prefixes) - 1, 0, -1):
            if lower_level not in rocks_by_level:
                continue

            self.perform_level_merge(lower_level, rocks_by_level, sentence_words, sentence_spans)
        
        # Return the results
        entities_arr = []
        highest_exist = 0
        while highest_exist < len(self.start_prefixes) and highest_exist not in rocks_by_level:
            highest_exist += 1
        
        if highest_exist < len(self.start_prefixes):
            entities_arr = [entity for entity in rocks_by_level[highest_exist]]

        relationships, just_entities = self.format_result(sentence, entities_arr)
        return {
            "text" : sentence,
            "relationships" : relationships,
            "just_entities" : just_entities
        }

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help = "The directory containing the model")
    parser.add_argument("--input_path", type=str, required=True, help = "The path to the file we want to process")
    parser.add_argument("--save_path", type=str, required=True, help = "The path to the json where we want to save the result")
    return parser.parse_args()

def main():
    # Load the model
    args = read_args()

    # If the model doesn't exist download it from huggingface
    if not os.path.exists(args.model_dir):
        save_dir = "spanbert_macrostrat_finetuned"
        if not os.path.exists(save_dir):
            repo = Repository(local_dir = save_dir, clone_from = args.model_dir)
        args.model_dir = save_dir

    tree_generator = TreeGenerator(args.model_dir)
        
    # Load the text
    with open(args.input_path, 'r') as reader:
        input_text = reader.read().strip()
        
    # Save the result
    run_id = "run_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f"))
    json_to_save = {
        "run" : run_id,
        "extraction_pipeline_id" : "0",
        "model_id" : "tree_based_span_bert_0",
        "results" : [tree_generator.run_for_text(input_text)]
    }

    with open(args.save_path, "w+") as writer:
        json.dump(json_to_save, writer, indent = 4)

if __name__ == "__main__":
    main()