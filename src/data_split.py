import numpy as np
import csv
import yaml
import json
import pandas as pd
from pathlib import Path
import os
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split

parent = Path(__file__).resolve().parent.parent

with open(parent / "config.yaml", "r") as f:
    configs = yaml.full_load(f)

def split_data(dataset):
    def transform_data(dataset):
        def create_corpus(batch):
            return {
                "corpus": [
                    f"{pos} {neg}" 
                    for pos, neg in zip(batch['positive'], batch['negative'])
                ]
            }
        
        dataset = dataset.map(create_corpus, batched=True, batch_size=1000)
        return dataset

    dataset = transform_data(dataset)

    train_test_ds = dataset["train"].train_test_split(test_size=0.2, seed=42, shuffle=True)
    test_valid_ds = train_test_ds["test"].train_test_split(test_size=0.5, seed=42, shuffle=True)

    train_dataset = train_test_ds["train"]
    test_dataset = test_valid_ds["test"]
    valid_dataset = test_valid_ds["train"]

    final_ds = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
        "valid": valid_dataset
    })
    return final_ds

if __name__ == "__main__":
    # doc_dataset = load_dataset("parquet", data_files=str(parent / configs["dataset"]["triplet"]["docs"]))
    # final_ds = split_data(doc_dataset)
    # final_ds.save_to_disk(parent / "cranfield/triplet/docs")

    queries_dataset = load_dataset("parquet", data_files=str(parent / configs["dataset"]["mnrl"]))
    final_ds = split_data(queries_dataset)
    final_ds.save_to_disk(parent / "cranfield/mnrl")

