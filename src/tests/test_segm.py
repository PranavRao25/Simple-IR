import pytest
from src.sentenceSegmentation import *
import json
from pathlib import Path

current_dir = Path(__file__).resolve().parent  # test

def test_naive_suite():
    s = 0
    for pair in suite:
        sent, output, length = pair["sentence"], pair["output"], pair["output_length"]
        out = seg.naive(sent)
        s += (out == output)
        # print(out)
    return s/len(suite) * 100

def test_punkt_suite():
    s = 0
    for pair in suite:
        sent, output, length = pair["sentence"], pair["output"], pair["output_length"]
        out = seg.punkt(sent)
        s += (out == output)
        # print(out)
    return s/len(suite) * 100

def test_spacy_suite():
    s = 0
    for pair in suite:
        sent, output, length = pair["sentence"], pair["output"], pair["output_length"]
        out = seg.spacySegmenter(sent)
        s += (out == output)
        # print(out)
    return s/len(suite) * 100

if __name__ == "__main__":
    with open(current_dir / "adv_test_suite.json", "r") as f:
        suite = json.load(f)

    seg = SentenceSegmentation()
    docs = docs_loader()

    naive_results = test_naive_suite()
    print(f"Naive performance: {naive_results}")

    punkt_results = test_punkt_suite()
    print(f"Punkt performance: {punkt_results}")

    spacy_results = test_spacy_suite()
    print(f"Spacy performance: {spacy_results}")
