# Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved
"""
Run BERT on several relation extraction benchmarks.
Adding some special tokens instead of doing span pair prediction in this version.
"""

import argparse
import logging
import os
import random
import spacy
import time
import json
import pandas as pd
import traceback
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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for span pair classification."""

    def __init__(self, guid, sentence, span1, span2, ner1, ner2, label):
        self.guid = guid
        self.sentence = sentence
        self.span1 = span1
        self.span2 = span2
        self.ner1 = ner1
        self.ner2 = ner2
        self.label = label
    
    def get_json(self):
        return {
            "guid" : self.guid,
            "sentence" : self.sentence,
            "span1" : self.span1,
            "span2" : self.span2,
            "ner1" : self.ner1,
            "ner2" : self.ner2,
            "label" : self.label
        }


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

nlp = spacy.load("en_core_web_lg") 
def get_span(txt_tokens, search_txt):
    search_tokens = [str(token) for token in nlp(search_txt.lower())]
    
    for start_idx in range(0, len(txt_tokens) - len(search_tokens)):
        found_match = True
        for search_idx in range(0, len(search_tokens)):
            if txt_tokens[start_idx + search_idx] != search_tokens[search_idx]:
                found_match = False
                break
        
        if found_match:
            return (start_idx, start_idx + len(search_tokens))
    
    return None

valid_labels = ["lith_to_attribute", "strat_to_lith", "lith_to_lith_type", "strat_name_to_lith", "lith_to_lith_att"]
class DataProcessor(object):
    """Processor for the custom data set."""

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r", encoding='utf-8') as reader:
            data = json.load(reader)
        return data

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "train.txt"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "valid.txt"), "valid")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "test.txt"), "test")

    def get_labels(self, data_dir, negative_label="no_relation"):
        """See base class."""
        return sorted(valid_labels)

    def _create_examples(self, dataset_path, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        # If the result already exists
        json_path = dataset_path.replace("txt", "json")
        if os.path.exists(json_path):
            print("Loading data from", json_path)
            with open(json_path, 'r') as reader:
                data = json.load(reader)

            for curr_data in data:
                if curr_data["label"] not in valid_labels:
                    continue

                examples.append(InputExample(
                    guid = curr_data["guid"],
                    sentence = curr_data["sentence"],
                    span1 = tuple(curr_data["span1"]),
                    span2 = tuple(curr_data["span2"]),
                    ner1 = curr_data["ner1"],
                    ner2 = curr_data["ner2"],
                    label = curr_data["label"]
                ))
            
            return examples

        # Get the files to load
        data_dir = os.path.dirname(dataset_path)
        with open(dataset_path, 'r') as reader:
            files = reader.readlines()
        
        # Load example from each file
        for file_name in files:
            file_name = file_name.strip()
            if len(file_name) == 0:
                continue
            
            # Generate examples from the curent df
            file_id = file_name[ : file_name.index(".")]
            file_path = os.path.join(data_dir, file_name)
            curr_df = pd.read_csv(file_path, sep = '\t')
            log_rate = max(int(len(curr_df.index)/20), 10)
            for idx, row in curr_df.iterrows():
                relationship_type = row["type"]
                if relationship_type not in valid_labels:
                    continue

                # Get the labels
                example_id = str(row["doc_id"]) + str(file_id) + str(idx)
                if relationship_type.startswith("att"):
                    src_type, dst_type = "lith", "lith_att"
                else:
                    parts = relationship_type.split("_")
                    to_idx = parts.index("to")
                    src_type, dst_type = "_".join(parts[ : to_idx]), "_".join(parts[to_idx + 1 : ])

                # Determine the spans
                tokens = [str(token) for token in nlp(row["text"].strip())]
                src_span = get_span(tokens, str(row["src"]))
                dst_span = get_span(tokens, str(row["dst"]))
                if src_span is None or dst_span is None:
                    continue

                examples.append(InputExample(
                    guid = example_id,
                    sentence = tokens,
                    span1 = src_span,
                    span2 = dst_span,
                    ner1 = src_type, 
                    ner2 = dst_type,
                    label = relationship_type
                ))

                if idx % log_rate == 0:
                    print("Finished processing", idx, "/", len(curr_df.index), "rows from file", file_name)
            
            print("Finished processing file", file_name)

        # Write examples file
        data_to_write = [example.get_json() for example in examples]
        print("Total number of examples is", len(data_to_write))
        with open(json_path, 'w+') as writer:
            json.dump(data_to_write, writer, indent = 4)

        return examples

max_total = 0
def convert_examples_to_features(examples, label2id, max_seq_length, tokenizer, special_tokens, mode='text'):
    global max_total
    """Loads a data file into a list of `InputBatch`s."""

    def get_special_token(w):
        if w not in special_tokens:
            special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
        return special_tokens[w]

    num_tokens = 0
    num_fit_examples = 0
    num_shown_examples = 0
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Create the tokens
        tokens = [CLS]
        SUBJECT_START = get_special_token("SUBJ_START")
        SUBJECT_END = get_special_token("SUBJ_END")
        OBJECT_START = get_special_token("OBJ_START")
        OBJECT_END = get_special_token("OBJ_END")
        SUBJECT_NER = get_special_token("SUBJ=%s" % example.ner1)
        OBJECT_NER = get_special_token("OBJ=%s" % example.ner2)

        # Build the initial tokens
        start_idx = min(example.span1[0], example.span2[0])
        end_idx = max(example.span1[1], example.span2[1])
        for i in range(start_idx, end_idx):
            if i == example.span1[0]:
                tokens.append(SUBJECT_START)
            if i == example.span2[0]:
                tokens.append(OBJECT_START)
            for sub_token in tokenizer.tokenize(example.sentence[i]):
                tokens.append(sub_token)
            if i == example.span1[1]:
                tokens.append(SUBJECT_END)
            if i == example.span2[1]:
                tokens.append(OBJECT_END)

        while (start_idx >= 0 or end_idx < len(example.sentence)) and len(tokens) < max_seq_length:
            # Include the left word
            if start_idx >= 0:
                left_tokens = [sub_token for sub_token in tokenizer.tokenize(example.sentence[start_idx])]
                tokens = left_tokens + tokens
                start_idx -= 1
            
            # Include the right word
            if end_idx < len(example.sentence):
                right_tokens = [sub_token for sub_token in tokenizer.tokenize(example.sentence[end_idx])]
                tokens = tokens + right_tokens
                end_idx += 1

        # Limit to most sequence length
        tokens.append(SEP)
        num_tokens += len(tokens)
        max_total = max(max_total, len(tokens))
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
        else:
            num_fit_examples += 1

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_id = label2id[example.label]
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if num_shown_examples < 20:
            if (ex_index < 5) or (label_id > 0):
                num_shown_examples += 1
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                        [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))

    logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
    logger.info("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
                num_fit_examples * 100.0 / len(examples), max_seq_length))
    return features


def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
            return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_f1(preds, labels):
    n_gold = n_pred = n_correct = 0
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if (pred != 0) and (label != 0) and (pred == label):
            n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0
        return {'precision': prec, 'recall': recall, 'f1': f1}


def evaluate(model, device, eval_dataloader, eval_label_ids, num_labels, verbose=True):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds[0], axis=1)
    result = compute_f1(preds, eval_label_ids.numpy())
    result['accuracy'] = simple_accuracy(preds, eval_label_ids.numpy())
    result['eval_loss'] = eval_loss
    if verbose:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return preds, result


def main(args):
    global max_total

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = 1

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(args)
    logger.info("device: {}, n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, args.fp16))

    processor = DataProcessor()
    label_list = processor.get_labels(args.data_dir, args.negative_label)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # Load the data    
    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
    special_tokens = {}

    try: 
        if args.do_eval:
            eval_examples = processor.get_dev_examples(args.data_dir)
            eval_features = convert_examples_to_features(
                eval_examples, label2id, args.max_seq_length, tokenizer, special_tokens, args.feature_mode)
            logger.info("***** Dev *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
            eval_label_ids = all_label_ids

        if args.do_train:
            train_examples = processor.get_train_examples(args.data_dir)
            train_features = convert_examples_to_features(
                    train_examples, label2id, args.max_seq_length, tokenizer, special_tokens, args.feature_mode)

            if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
                train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
            else:
                random.shuffle(train_features)

            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
            train_batches = [batch for batch in train_dataloader]

            num_train_optimization_steps = \
                len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

            logger.info("***** Training *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)

            best_result = None
            eval_step = max(1, len(train_batches) // args.eval_per_epoch)
            lrs = [args.learning_rate] if args.learning_rate else \
                [1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
            for lr in lrs:
                model = BertForSequenceClassification.from_pretrained(
                    args.model, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE), num_labels=num_labels)
                if args.fp16:
                    model.half()
                model.to(device)
                if n_gpu > 1:
                    model = torch.nn.DataParallel(model)

                param_optimizer = list(model.named_parameters())
                no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in param_optimizer
                                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                    {'params': [p for n, p in param_optimizer
                                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
                if args.fp16:
                    try:
                        from apex.optimizers import FP16_Optimizer
                        from apex.optimizers import FusedAdam
                    except ImportError:
                        raise ImportError("Please install apex from https://www.github.com/nvidia/apex"
                                        "to use distributed and fp16 training.")

                    optimizer = FusedAdam(optimizer_grouped_parameters,
                                        lr=lr,
                                        bias_correction=False,
                                        max_grad_norm=1.0)
                    if args.loss_scale == 0:
                        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
                    else:
                        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

                else:
                    optimizer = BertAdam(optimizer_grouped_parameters,
                                        lr=lr,
                                        warmup=args.warmup_proportion,
                                        t_total=num_train_optimization_steps)

                start_time = time.time()
                global_step = 0
                tr_loss = 0
                nb_tr_examples = 0
                nb_tr_steps = 0
                for epoch in range(int(args.num_train_epochs)):
                    model.train()
                    logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
                    if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                        random.shuffle(train_batches)
                    for step, batch in enumerate(train_batches):
                        batch = tuple(t.to(device) for t in batch)
                        input_ids, input_mask, segment_ids, label_ids = batch
                        loss = model(input_ids, segment_ids, input_mask, label_ids)
                        if n_gpu > 1:
                            loss = loss.mean()
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps

                        if args.fp16:
                            optimizer.backward(loss)
                        else:
                            loss.backward()

                        tr_loss += loss.item()
                        nb_tr_examples += input_ids.size(0)
                        nb_tr_steps += 1

                        if (step + 1) % args.gradient_accumulation_steps == 0:
                            if args.fp16:
                                lr_this_step = lr * \
                                    warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = lr_this_step
                            optimizer.step()
                            optimizer.zero_grad()
                            global_step += 1

                        if (step + 1) % eval_step == 0:
                            logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                                        epoch, step + 1, len(train_batches),
                                        time.time() - start_time, tr_loss / nb_tr_steps))
                            save_model = False
                            if args.do_eval:
                                preds, result = evaluate(model, device, eval_dataloader, eval_label_ids, num_labels)
                                model.train()
                                result['global_step'] = global_step
                                result['epoch'] = epoch
                                result['learning_rate'] = lr
                                result['batch_size'] = args.train_batch_size
                                logger.info("First 20 predictions:")
                                for pred, label in zip(preds[:20], eval_label_ids.numpy()[:20]):
                                    sign = u'\u2713' if pred == label else u'\u2718'
                                    logger.info("pred = %s, label = %s %s" % (id2label[pred], id2label[label], sign))
                                if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                                    best_result = result
                                    save_model = True
                                    logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                                (args.eval_metric, str(lr), epoch, result[args.eval_metric] * 100.0))
                            else:
                                save_model = True

                            if save_model:
                                model_to_save = model.module if hasattr(model, 'module') else model
                                output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                                torch.save(model_to_save.state_dict(), output_model_file)
                                model_to_save.config.to_json_file(output_config_file)
                                tokenizer.save_vocabulary(args.output_dir)
                                if best_result:
                                    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                                    with open(output_eval_file, "w") as writer:
                                        for key in sorted(result.keys()):
                                            writer.write("%s = %s\n" % (key, str(result[key])))

        if args.do_eval:
            if args.eval_test:
                eval_examples = processor.get_test_examples(args.data_dir)
                eval_features = convert_examples_to_features(
                    eval_examples, label2id, args.max_seq_length, tokenizer, special_tokens, args.feature_mode)
                logger.info("***** Test *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
                eval_label_ids = all_label_ids
            model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
            if args.fp16:
                model.half()
            model.to(device)
            preds, result = evaluate(model, device, eval_dataloader, eval_label_ids, num_labels)
            with open(os.path.join(args.output_dir, "predictions.txt"), "w") as f:
                for ex, pred in zip(eval_examples, preds):
                    f.write("%s\t%s\n" % (ex.guid, id2label[pred]))
            with open(os.path.join(args.output_dir, "test_results.txt"), "w") as f:
                for key in sorted(result.keys()):
                    f.write("%s = %s\n" % (key, str(result[key])))
    
    except Exception as e: 
       print("Got exception of", traceback.format_exc())

    # Write the label to id mapping
    data_to_write = {"label2id" : label2id, "id2label" : id2label}
    with open(os.path.join(args.output_dir, "label_mapping.json"), "w+") as writer:
        json.dump(data_to_write, writer, indent = 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_per_epoch", default=10, type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--negative_label", default="no_relation", type=str)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--train_mode", type=str, default='random_sorted', choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_test", action="store_true", help="Whether to evaluate on final test set.")
    parser.add_argument("--feature_mode", type=str, default="text", choices=["text", "ner", "text_ner", "ner_text"])
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_metric", default="f1", type=str)
    parser.add_argument("--learning_rate", default=None, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()
    main(args)
