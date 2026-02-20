from src.sentenceSegmentation import *
import yaml
from pathlib import Path

if __name__ == "__main__":
    parent = Path(__name__).resolve().parent

    with open(parent / "config.yaml", "r") as f:
        configs = yaml.full_load(f)

    docs = docs_loader()
    seg = SentenceSegmentation()

    sentences = []
    for doc in docs:
        sentences.append(seg.naive(doc))
    
    with open(configs["output"] / "seg.txt", "w") as f:
        f.writelines(sentences)
