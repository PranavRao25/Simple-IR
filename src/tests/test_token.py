import pytest
from pathlib import Path
from src.tokenization import *
import json

def test_naive():
    s = 0
    for entry in suite:
        sentence, output = entry["text"], entry["expected"]
        out = tok.naive(sentence)
        # print(out)
        s += out == output
    return s / len(suite) * 100

def test_penn():
    s = 0
    for entry in suite:
        sentence, output = entry["text"], entry["expected"]
        out = tok.pennTreeBank(sentence)
        # print(out)
        s += out == output
    return s / len(suite) * 100

def test_spacy():
    s = 0
    for entry in suite:
        sentence, output = entry["text"], entry["expected"]
        out = tok.spacyTokenizer(sentence)
        # print(out)
        s += out == output
    return s / len(suite) * 100

if __name__ == "__main__":
    parent = Path(__file__).resolve().parent

    tok = Tokenization()

    with open(parent / "token_suite.json", "r") as f:
        suite = json.load(f)
    
    naive_results = test_naive()
    print(f"Result of Naive Tokenizer: {naive_results}")

    penn_results = test_penn()
    print(f"Result of Penn Tokenizer: {penn_results}")

    spacy_results = test_spacy()
    print(f"Result of Spacy Tokenizer: {spacy_results}")
