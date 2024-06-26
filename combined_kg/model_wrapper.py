import math
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from seq2rel import Seq2Rel
from seq2rel.common import util
from huggingface_hub import Repository
import os

class ModelWrapper:

    def __init__(self, model_path):
        raise NotImplementedError("ModelWrapper is an abstract class")
    
    def get_relations_in_line(self, line):
        raise NotImplementedError("ModelWrapper is an abstract class")

class RebelWrapper:

    def __init__(self, model_path):
        self.model_path = model_path
        self.span_length = 128

        print("Loading finetuned REBEL model from", self.model_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        print("Loaded REBEL model with device", self.device)
    
    def extract_relations_from_model_output(self, text):
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
                "model_used" : "rebel",
                'tail': object_.strip()
            })

        return relations
    
    def get_relations_in_line(self, line):
        # tokenize whole text
        inputs = self.tokenizer([line], return_tensors="pt").to(self.device)

        # compute span boundaries
        num_tokens = len(inputs["input_ids"][0])
        num_spans = math.ceil(num_tokens / self.span_length)
        overlap = math.ceil((num_spans * self.span_length - num_tokens) / 
                            max(num_spans - 1, 1))
        spans_boundaries = []
        start = 0
        for i in range(num_spans):
            spans_boundaries.append([start + self.span_length * i,
                                    start + self.span_length * (i + 1)])
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
        generated_tokens = self.model.generate(
            **inputs,
            **gen_kwargs,
        )

        # decode relations
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        all_relations = []
        for sentence_pred in decoded_preds:
            relations = self.extract_relations_from_model_output(sentence_pred)
            all_relations.extend(relations)
        
        return all_relations

class Seq2RelWrapper:

    def __init__(self, model_path):
        self.model_path = model_path
        print("Loading finetuned Seq2rel model from", self.model_path)

        # If the model doesn't exist then download it
        if not os.path.exists(self.model_path):
            save_dir = "seq2rel_macrostrat_finetuned"
            if not os.path.exists(save_dir):
                repo = Repository(local_dir = save_dir, clone_from = self.model_path)
            self.model_path = os.path.join(save_dir, 'model.tar.gz')

        cuda_device = -1
        if torch.cuda.is_available():
            cuda_device = 1
        self.model = Seq2Rel(self.model_path, cuda_device = cuda_device)
        print("Loaded finetuned Seq2rel model using cuda_device", cuda_device)
    
    def get_relations_in_line(self, line):
        output = self.model(line)
        all_results = util.extract_relations(output)

        all_relations = []
        for curr_result in all_results:
            for relationship_type in curr_result:
                # Read in the relationship
                relationship_data = curr_result[relationship_type][0]
                src_name, src_type = relationship_data[0]
                dst_name, dst_type = relationship_data[1]
                if len(src_name) == 0 or len(dst_name) == 0:
                    continue

                # Extract the node name
                head_node = src_name[0].strip()
                dst_node = dst_name[0].strip()
                if "unknown" in head_node or "unknown" in dst_node:
                    continue

                # Record the relationship
                all_relations.append({
                    "head" : head_node,
                    "type" : relationship_type.strip(),
                    "model_used" : "seq2rel",
                    "tail" : dst_node 
                })
        
        return all_relations