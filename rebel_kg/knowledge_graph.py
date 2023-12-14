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

def extract_relations_from_model_output(text):
    relations = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return relations

# extract relations for each span and put them together in a knowledge base
def get_kg_for_line(model, tokenizer, line, article_id, span_length=128):

    # tokenize whole text
    inputs = tokenizer([line], return_tensors="pt")

    # compute span boundaries
    num_tokens = len(inputs["input_ids"][0])
    num_spans = math.ceil(num_tokens / span_length)
    overlap = math.ceil((num_spans * span_length - num_tokens) / 
                        max(num_spans - 1, 1))
    spans_boundaries = []
    start = 0
    for i in range(num_spans):
        spans_boundaries.append([start + span_length * i,
                                 start + span_length * (i + 1)])
        start -= overlap

    # transform input with spans
    tensor_ids = [inputs["input_ids"][0][boundary[0]:boundary[1]]
                  for boundary in spans_boundaries]
    tensor_masks = [inputs["attention_mask"][0][boundary[0]:boundary[1]]
                    for boundary in spans_boundaries]
    inputs = {
        "input_ids": torch.stack(tensor_ids),
        "attention_mask": torch.stack(tensor_masks)
    }

    # generate relations
    num_return_sequences = 3
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": num_return_sequences
    }
    generated_tokens = model.generate(
        **inputs,
        **gen_kwargs,
    )

    # decode relations
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    # create kg
    kg = KG()
    for sentence_pred in decoded_preds:
        current_span_index = i // num_return_sequences
        relations = extract_relations_from_model_output(sentence_pred)
        for relation in relations:
            relation["source"] = {
                article_id: [line],
            }

            kg.add_relation(relation)

    return kg