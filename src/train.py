from pathlib import Path
import yaml
import json
from datasets import load_from_disk, Dataset
import wandb
import logging
import hashlib
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
import torch

parent = Path(__file__).resolve().parent.parent

with open(parent / "config.yaml", "r") as f:
    configs = yaml.full_load(f)

assert torch.cuda.is_available(), "CUDA is not available. Check your PyTorch installation!"
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

def generate_hash(id):
    return hashlib.md5(id.encode('utf-8')).hexdigest()

def load_evaluator_from_dataset(dataset: Dataset, name = "mnrl_eval") -> InformationRetrievalEvaluator:
    queries = {}
    corpus = {}
    relevant_docs = {}

    for row in dataset:
        q_text = row.get("anchor", "") # type: ignore
        pos_text = row.get("positive", "") # type: ignore
        neg_text = row.get("negative", "") # type: ignore # Fetch negative if it exists

        q_id = generate_hash(q_text)
        print("query")
        print(q_text, q_id)
        pos_id = generate_hash(pos_text)
        print("pos")
        print(pos_text, pos_id)

        queries[q_id] = q_text

        if q_id not in relevant_docs:
            relevant_docs[q_id] = set()
        relevant_docs[q_id].add(pos_id)

        corpus[pos_id] = pos_text
        if neg_text:
            neg_id = generate_hash(neg_text)
            print("neg")
            print(neg_text, neg_id)
            corpus[neg_id] = neg_text

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        show_progress_bar=True
    )

def train(train_dataset, valid_dataset, configs, evaluator, run):
    hyperparams = configs["hyperparams"]
    model_id = configs["id"]
    name = configs["name"]
    model_path = configs["path"]
    run_name = configs["run"]

    print(f"Model {name} training start")
    
    output_dir = parent / "output/mnrl"
    output_dir.mkdir(parents=True, exist_ok=True)

    # PREPARE THE MODEL
    lr = float(hyperparams.get("lr", 2e-5))
    train_batch_size = hyperparams.get("batch_size", 32)
    eval_batch_size = hyperparams.get("eval_batch_size", 16)
    mini_batch_size = hyperparams.get("mini_batch_size", 16)
    epochs = hyperparams.get("epochs", 1)
    weight_decay = float(hyperparams.get("weight_decay", 0.01))
    gradient_accumulation_steps = int(hyperparams.get("gradient_accumulation_steps", 8))
    warmup_ratio = float(hyperparams.get("warmup_ratio", 0.1))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"No of Epochs: {epochs}")

    try:
        model = SentenceTransformer(model_id).to(device)

        loss = CachedMultipleNegativesRankingLoss(model=model, mini_batch_size=mini_batch_size)

        model.train()

        wandb.watch(model, log="all", log_freq=100)

        training_args = SentenceTransformerTrainingArguments(
            output_dir=str(output_dir / run_name),
            learning_rate=lr,
            optim="adamw_torch",
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=eval_batch_size,
            warmup_ratio=warmup_ratio,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir=model_path,
            logging_steps=50,
            load_best_model_at_end=True,
            fp16=torch.cuda.is_available(),   # Mixed precision training for speed
            report_to="wandb",
            run_name=run_name
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset, # type: ignore
            eval_dataset=valid_dataset, # type: ignore
            loss=loss,
            evaluator=evaluator
        )

        trainer.train()

        results = trainer.evaluate()

        wandb.log(results)

        trainer.save_model(model_path)

        print("Training Done")

        run.alert(
            title=f"Training Run Complete {model_id}",
            text = f"Task 1 training done",
            level="INFO",
            wait_duration=0
        )
        wandb.finish()

        return model
    except Exception as e:
        raise e

def main(model_configs, train_dataset, valid_dataset, evaluator):
    # MODEL CONFIG
    model_details = model_configs
    id = model_details["name"]
    name = id.replace("/", "_")  # sentence-transformers/all-mpnet-base-v2
    path = model_details.get("path", "model/mnrl")
    hyperparams = model_details["hyperparams"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_configs = {
        "device": device,
        "hyperparams": hyperparams,
        "wandb": "nlp_siamese",
        "path": path,
        "id": id,
        "name": name,
        "run": f"Siamese_Train_{name}_mnrl"
    }

    # WANDB CONFIG
    wandb.init(
        project=train_configs["wandb"], # type: ignore
        config=train_configs,
        name = train_configs["name"] # type: ignore
    )
    run = wandb.init(
        project=str(train_configs["wandb"]),
        config=train_configs,
        name=str(train_configs["run"])
    )
    train(train_dataset, valid_dataset, train_configs, evaluator, run)

if __name__ == "__main__":
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
    
    train_dataset = queries_dataset["train"]
    valid_dataset = queries_dataset["valid"]

    sample_fraction = 1
    train_dataset = train_dataset.shuffle(seed=42).select(range(int(len(train_dataset) * sample_fraction))) # type: ignore
    valid_dataset = valid_dataset.shuffle(seed=42).select(range(int(len(valid_dataset) * sample_fraction))) # type: ignore

    evaluator = load_evaluator_from_dataset(valid_dataset)
    
    model = configs["model"]
    main(model, train_dataset, valid_dataset, evaluator)

    # doc_dataset = load_dataset("parquet", data_files=str(parent / configs["dataset"]["triplet"]["docs"]))
