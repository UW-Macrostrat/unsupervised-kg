# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""REBEL"""

from __future__ import absolute_import, division, print_function

import pandas as pd

import datasets
import os
import re 
import json
import logging
import math
from collections import defaultdict

_DESCRIPTION = """\
REBEL is a silver dataset created for the paper REBEL: Relation Extraction By End-to-end Language generation
"""

class ArchiveConfig(datasets.BuilderConfig):
    """BuilderConfig for REBEL."""

    def __init__(self, **kwargs):
        """BuilderConfig for REBEL.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ArchiveConfig, self).__init__(**kwargs)


class Archive(datasets.GeneratorBasedBuilder):
    """Archive 1.0"""

    BUILDER_CONFIGS = [
        ArchiveConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "triplets": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            # homepage="",
#             citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = {
            "train": self.config.data_files["train"], 
            "dev": self.config.data_files["dev"], 
            "test": self.config.data_files["test"], 
        }

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        filepath = filepath[0]
        logging.info("generating examples from = %s", filepath)
        dir_path = os.path.dirname(filepath)

        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
            for file_idx, curr_line in enumerate(lines):
                # Read in the dataframe
                curr_line = curr_line.strip()
                df_path = os.path.join(dir_path, curr_line)
                df = pd.read_csv(df_path).astype(str)

                for row_idx, row in df.iterrows():
                    data_id = str(row["doc_id"]) + "_" + str(row_idx)
                    title = row["title"]
                    text = row["text"]
                    src_txt, dst_txt, type_txt = row["src"], row["dst"], row["type"]
                    triplet_txt = f"<triplet> {src_txt} <subj> {dst_txt} <obj> {type_txt}"

                    yield data_id, {
                        "title" : title,
                        "context" : text,
                        "id" : data_id,
                        "triplets" : triplet_txt
                    }