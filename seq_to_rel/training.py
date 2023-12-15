import os
import json
from pathlib import Path
import subprocess

def training():
    train_path = Path("data") / "train.tsv"
    test_path = Path("data") / "test.tsv"
    valid_path = Path("data") / "valid.tsv"
    config_file = "training.jsonnet"

    #Run the train command
    dataset_size = len(Path(train_path).read_text().strip().split("\n"))
    command = (
        f"train_data_path={train_path} "
        f"valid_data_path={valid_path} "
        f"dataset_size={dataset_size} "
        f"allennlp train {config_file} "
        f"--serialization-dir output "
        f"--include-package seq2rel "
        f"-f"
    )

    print("Running the command", command)

    subprocess.run(command, shell=True)

if __name__ == "__main__":
    training()