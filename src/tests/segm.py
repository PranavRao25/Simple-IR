from src.sentenceSegmentation import *
from src.util import *
import yaml
from pathlib import Path

if __name__ == "__main__":
    parent = Path(__file__).resolve().parent.parent.parent
    with open(parent / "config.yaml", "r") as f:
        configs = yaml.full_load(f)

    seg = SentenceSegmentation()

    queries = queries_loader()
    text = [query["query"] for query in queries]

    segmented_naive = [seg.naive(sent) for sent in text]
    with open(parent / configs["output"] / "seg_naive_query.txt", "w") as f:
        for sent in segmented_naive:
            f.writelines(sent)

    segmented_punkt = [seg.punkt(sent) for sent in text]
    with open(parent / configs["output"] / "seg_punkt_query.txt", "w") as f:
        for sent in segmented_punkt:
            f.writelines(sent)
    
    # segmented_spacy = [seg.spacySegmenter(sent) for sent in text]
    # with open(parent / configs["output"] / "seg_spacy_query.txt", "w") as f:
    #     for sent in segmented_spacy:
    #         f.writelines(sent)

    docs = docs_loader()
    text = [doc["body"] for doc in docs]

    segmented_naive = [seg.naive(sent) for sent in text]
    with open(parent / configs["output"] / "seg_naive_docs.txt", "w") as f:
        for sent in segmented_naive:
            f.writelines(sent)

    segmented_punkt = [seg.punkt(sent) for sent in text]
    with open(parent / configs["output"] / "seg_punkt_docs.txt", "w") as f:
        for sent in segmented_punkt:
            f.writelines(sent)
    
    # segmented_spacy = [seg.spacySegmenter(sent) for sent in text]
    # with open(parent / configs["output"] / "seg_spacy_docs.txt", "w") as f:
    #     for sent in segmented_spacy:
    #         f.writelines(sent)
