import argparse
import logging
import os
import random
import spacy
import time
import json
import pandas as pd

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

CLS = "[CLS]"
SEP = "[SEP]"

class RE_Extractor:

    def __init__(self, data_dir):
        # Load the tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_tokenizer = BertTokenizer.from_pretrained(data_dir, do_lower_case = False)

        # Load the labels
        with open(os.path.join(data_dir, "label_mapping.json"), "r") as reader:
            label_to_id = json.load(reader)["label2id"]
        self.max_seq_length = 512

        self.ids_by_start = {}
        for label_name, label_id in label_to_id.items():
            label_start = label_name.split("_")[0].strip()
            label_id = int(label_id)
            if label_start not in self.ids_by_start:
                self.ids_by_start[label_start] = []
            self.ids_by_start[label_start].append(label_id)

        # Load the model
        self.model = BertForSequenceClassification.from_pretrained(data_dir, num_labels = len(label_to_id)).to(self.device)
    
    def extract_features(self, inputs, special_tokens = {}):

        words = inputs["words"]
        def get_special_token(w):
            if w not in special_tokens:
                special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
            return special_tokens[w]

        # Get the tokens
        num_tokens = 0
        tokens = [CLS]
        SUBJECT_START = get_special_token("SUBJ_START")
        SUBJECT_END = get_special_token("SUBJ_END")
        OBJECT_START = get_special_token("OBJ_START")
        OBJECT_END = get_special_token("OBJ_END")

        # Build the initial tokens
        start_idx = min(inputs["span1"][0], inputs["span2"][0])
        end_idx = max(inputs["span1"][1], inputs["span2"][1])
        for i in range(start_idx, end_idx):
            if i == inputs["span1"][0]:
                tokens.append(SUBJECT_START)
            if i == inputs["span2"][0]:
                tokens.append(OBJECT_START)
            for sub_token in self.bert_tokenizer.tokenize(words[i]):
                tokens.append(sub_token)
            if i == inputs["span1"][1]:
                tokens.append(SUBJECT_END)
            if i == inputs["span2"][1]:
                tokens.append(OBJECT_END)

        while (start_idx >= 0 or end_idx < len(words)) and len(tokens) < self.max_seq_length:
            # Include the left word
            if start_idx >= 0:
                left_tokens = [sub_token for sub_token in self.bert_tokenizer.tokenize(words[start_idx])]
                tokens = left_tokens + tokens
                start_idx -= 1
            
            # Include the right word
            if end_idx < len(words):
                right_tokens = [sub_token for sub_token in self.bert_tokenizer.tokenize(words[end_idx])]
                tokens = tokens + right_tokens
                end_idx += 1
        
        tokens.append(SEP)

        # Extract the features
        num_tokens += len(tokens)
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        
        # Convert the tokens into ids
        segment_ids = [0] * len(tokens)
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        # Return it as tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        input_mask = torch.tensor([input_mask], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long).to(self.device)

        return input_ids, input_mask, segment_ids

    def get_prediction(self, inputs, lower_type, upper_type):
        # Get the probability for each label
        input_ids, input_mask, segment_ids = self.extract_features(inputs)
        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_mask, labels=None)
        probabilities = torch.nn.functional.softmax(logits, dim = -1).detach().cpu().numpy()[0]

        # Get idxs to use
        if upper_type == "strat":
            idxs_to_use = self.ids_by_start[upper_type]
        elif lower_type == "att":
            idxs_to_use = self.ids_by_start[lower_type]
        else:
            return -1.0

        # Get the probabilities for that type
        return None, np.max(probabilities[idxs_to_use])

    def get_interval_distance(self, r1, r2):
     x, y = sorted((r1, r2))
     if x[0] <= x[1] < y[0] and all( y[0] <= y[1] for y in (r1,r2)):
        return y[0] - x[1]
     return 0
    
    def get_sentence_id(self, span, sentence_spans):
        span_start = span[0]
        for idx, (sentence_start, sentence_end) in enumerate(sentence_spans):
            if span_start >= sentence_start and span_start < sentence_end:
                return idx
        
        return len(sentence_spans)

    def get_relationship_probability(self, sentence_words, first_rock, second_rock, lower_type, upper_type, sentence_spans):
        # Find the pair of occurences with the lowest distance
        lowest_first_span, lowest_second_span, lowest_distance = None, None, None
        for curr_first_span in first_rock.occurences:
            for curr_second_span in second_rock.occurences:
                # Verify that the spans are in the same distance
                if self.get_sentence_id(curr_first_span, sentence_spans) != self.get_sentence_id(curr_second_span, sentence_spans):
                    continue

                curr_distance = self.get_interval_distance(curr_first_span, curr_second_span)
                if lowest_distance is None or curr_distance < lowest_distance:
                    lowest_first_span, lowest_second_span = curr_first_span, curr_second_span

        # Get the probability for this span
        if lowest_first_span is None or lowest_second_span is None:
            return -1.0

        _, first_pair_probability = self.get_prediction({
            "words" : sentence_words,
            "span1" : lowest_first_span,
            "span2" : lowest_second_span
        }, lower_type, upper_type)

        _, second_pair_probability = self.get_prediction({
            "words" : sentence_words,
            "span1" : lowest_second_span,
            "span2" : lowest_first_span
        }, lower_type, upper_type)

        return max(first_pair_probability, second_pair_probability)

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help = "The directory containing the model")
    return parser.parse_args()

def main(args):
    # Load the model
    model = RE_Extractor(args.model_dir)

    # Format the input
    nlp = spacy.load("en_core_web_lg") 
    example_text = "the mount galen volcanics consists of basalt, andesite, dacite, and rhyolite lavas and dacite and rhyolite tuff and tuff-breccia. "
    example_text += "The Hayhook formation was named, mapped and discussed by lasky and webber (1949). the formation ranges up to at least 2500 feet in thickness."

    input = {
        "words" : [str(token) for token in  nlp(example_text)],
        "span1" : (1, 4),
        "span2" : (6, 7)
    }
    prediction, probability = model.get_prediction(input, "att", "lith")
    print("Got prediction", prediction, "has probability of", probability)

if __name__ == "__main__":
    main(read_args())