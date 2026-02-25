import argparse
import os
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

VALID_LABELS = [
    "lith_to_attribute",
    "strat_to_lith",
    "lith_to_lith_type",
    "strat_name_to_lith",
    "lith_to_lith_att",
]

ENTITY_TYPE_MAP = {
    1: "strat",
    2: "lith",
    3: "attribute",
    4: "lith_type"
}


def map_relation(parent_type, child_type):
    if parent_type == "lith" and child_type == "attribute":
        return "lith_to_attribute"
    if parent_type == "strat" and child_type == "lith":
        return "strat_to_lith"
    if parent_type == "lith" and child_type == "lith_type":
        return "lith_to_lith_type"
    if parent_type == "strat" and child_type == "lith":
        return "strat_name_to_lith"
    if parent_type == "lith" and child_type == "attribute":
        return "lith_to_lith_att"
    return None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FeedbackDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_len, label2id):

        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id

        all_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".json")
        ]

        print(f"Loading {len(all_files)} feedback files...")

        for file_path in all_files:
            with open(file_path, "r") as f:
                raw_data = json.load(f)

            for item in raw_data:
                paragraph = item.get("paragraph_text")
                entities = item.get("entities", [])

                if not paragraph:
                    continue

                for parent in entities:
                    parent_type = ENTITY_TYPE_MAP.get(parent.get("type"))
                    parent_name = parent.get("name")

                    for child in parent.get("children", []):
                        child_type = ENTITY_TYPE_MAP.get(child.get("type"))
                        child_name = child.get("name")

                        if not parent_type or not child_type:
                            continue

                        label = map_relation(parent_type, child_type)
                        if label not in VALID_LABELS:
                            continue

                        combined_text = (
                            f"{parent_name} [SEP] {child_name} [SEP] {paragraph}"
                        )

                        self.samples.append({
                            "text": combined_text,
                            "label": label
                        })

        print(f"Generated {len(self.samples)} total training samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        encoding = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.label2id[item["label"]])
        }


def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(VALID_LABELS)
    ).to(device)

    label2id = {l: i for i, l in enumerate(VALID_LABELS)}

    train_dataset = FeedbackDataset(
        args.data_dir,
        tokenizer,
        args.max_seq_length,
        label2id
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    total_steps = len(train_loader) * args.num_train_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * total_steps),
        num_training_steps=total_steps
    )

    scaler = GradScaler()

    print("\nStarting training...\n")

    for epoch in range(int(args.num_train_epochs)):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            with autocast():
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nModel saved to: {args.output_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", required=True,
                        help="Folder containing feedback JSON files")

    parser.add_argument("--model", required=True,
                        help="HF model name OR local checkpoint")

    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--num_train_epochs", default=3, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--no_cuda", action="store_true")

    args = parser.parse_args()
    train(args)