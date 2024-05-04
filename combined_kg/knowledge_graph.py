import math
import torch

class KG():
    def __init__(self):
        self.entities = set()
        self.relations = []
        
    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)
    
    def merge_with_kb(self, kb2):
        for r in kb2.relations:
            self.add_relation(r)
        
    def add_entity(self, e):
        self.entities.add(e)
    
    def merge_relations(self, r2):
        r1 = [r for r in self.relations if self.are_relations_equal(r2, r)][0]
        existing_srcs = r1["source"]
        
        all_new_sources = r2["source"]
        for article_id in all_new_sources:
            article_sentences = all_new_sources[article_id]
            if article_id not in existing_srcs:
                existing_srcs[article_id] = article_sentences
            else:
                existing_srcs[article_id].extend(article_sentences)

    def add_relation(self, r):
        # manage new entities
        entities = [r["head"], r["tail"]]
        for e in entities:
            self.add_entity(e)

        # manage new relation
        if not self.exists_relation(r):
            self.relations.append(r)
        else:
            self.merge_relations(r)

# extract relations for each span and put them together in a knowledge base
def get_kg_for_line(model, line, article_id, span_length=128):
    # create kg
    kg = KG()

    all_relations = model.get_relations_in_line(line)
    for relation in all_relations:
        relation["source"] = {
            article_id: [line],
        }
        
        kg.add_relation(relation)        

    return kg