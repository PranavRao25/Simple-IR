from pathlib import Path
import yaml
import json
from datasets import load_from_disk
import torch.nn.functional as F
import wandb
import logging
from sentence_transformers.losses import TripletLoss
from sentence_transformers.evaluation import TripletEvaluator, SimilarityFunction
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
import torch
from datasets import load_dataset

parent = Path(__file__).resolve().parent.parent

with open(parent / "config.yaml", "r") as f:
    configs = yaml.full_load(f)

assert torch.cuda.is_available(), "CUDA is not available. Check your PyTorch installation!"
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

def inference(test_dataset, configs, run):
    """
        MODEL PERFORMANCE ON TEST DATA
    """

    model_path = configs["path"]
    model_id = configs["name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = SentenceTransformer(model_path).to(device)
        model.eval()

        test_evaluator = TripletEvaluator(
            anchors=test_dataset["anchor"],
            positives=test_dataset["positive"],
            negatives=test_dataset["negative"],
            name="test_eval",
        )
        results = test_evaluator(model)
        
        wandb.log(results)

        print("INFERENCE DONE")
        run.alert(
            title=f"Inference Run Complete {model_id}",
            text = f"Inference done",
            level="INFO",
            wait_duration=0
        )
        wandb.finish()
    except Exception as e:
        raise e

def main():
    logging.basicConfig(
        filename=parent/"debug.log",
        format='%(levelname)s: %(message)s',
        filemode='a'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.debug("MODEL TRAIN START")

    # DATA LOADING
    queries_dataset = load_from_disk("cranfield/mnrl")
    test_dataset = queries_dataset["test"]

    sample_fraction = 1
    test_dataset = test_dataset.shuffle(seed=42).select(range(int(len(test_dataset) * sample_fraction)))

    # MODEL CONFIG
    model_details = configs["model"]
    id = model_details["name"]
    name = id.replace("/", "_")  # sentence-transformers/all-mpnet-base-v2
    path = model_details["path"]
    hyperparams = model_details["hyperparams"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # WANDB CONFIG    
    test_configs = {
        "device": device,
        "hyperparams": hyperparams,
        "wandb": "nlp_siamese",
        "path": path,
        "name": name,
        "run": f"Siamese_Inference_{name}_mnrl"
    }
    run = wandb.init(
        project=str(test_configs["wandb"]),
        config=test_configs,
        name=str(test_configs["run"])
    )

    inference(test_dataset, test_configs, run)

if __name__ == "__main__":
    main()

    # doc_dataset = load_dataset("parquet", data_files=str(parent / configs["dataset"]["triplet"]["docs"]))
