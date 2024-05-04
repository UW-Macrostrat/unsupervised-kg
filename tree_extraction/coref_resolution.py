from fastcoref import FCoref
import torch
import spacy

class CorefResolver:

    def __init__(self):
        self.model = FCoref(device = "cuda" if torch.cuda.is_available() else "cpu")
    
    def record_cooref_occurences(self, sentence_words, word_ranges, rock_terms):
        prediction = self.model.predict(texts = [sentence_words], is_split_into_words=True)[0]
        all_clusters = prediction.get_clusters(as_strings=False)
        
        for curr_cluster in all_clusters:
            # Check if this cluster overlaps with an existing term
            overlap_idx = -1
            for cluster_term_range in curr_cluster:
                for idx, rock_term in enumerate(rock_terms):
                    does_overlap = rock_term.does_range_overlap(cluster_term_range)
                    if does_overlap:
                        overlap_idx = idx
                        break

                if overlap_idx != -1:
                    break
            
            # Merge the cluster with term
            if overlap_idx == -1:
                continue
            
            # Perform the merge
            rock_to_update = rock_terms[overlap_idx]
            for cluster_term_range in curr_cluster:
                word_start, word_end = cluster_term_range[0], cluster_term_range[1]
                txt_range = (word_ranges[word_start][0], word_ranges[word_end - 1][1])
                rock_to_update.add_in_range(cluster_term_range, txt_range)

def main():
    txt = "he artillery formation was named, mapped and discussed by lasky and webber (1949). the formation ranges up to at least 2500 feet in thickness. "
    nlp = spacy.load("en_core_web_lg") 
    sentence_words = [str(token.text) for token in nlp(txt)]

    resolver = CorefResolver()
    resolver.record_cooref_occurences(sentence_words)

if __name__ == "__main__":
    main()