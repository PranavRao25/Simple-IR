import pytest
from src.sentenceSegmentation import *
import json

with open("adv_test_suite.json", "r") as f:
    suite = json.load(f)

seg = SentenceSegmentation()
docs = docs_loader()

def test_naive_suite():
    for pair in suite:
        sent, output = pair["sentence"], pair["output"]
        assert output == seg.naive(sent)

    with pytest.raises(TypeError):
        seg.naive(1)

def test_punkt_suite():
    for pair in suite:
        sent, output = pair["sentence"], pair["output"]
        assert output == seg.punkt(sent)

    with pytest.raises(TypeError):
        seg.punkt(1)

def test_spacy_suite():
    for pair in suite:
        sent, output = pair["sentence"], pair["output"]
        assert output == seg.spacySegmenter(sent)

    with pytest.raises(TypeError):
        seg.spacySegmenter(1)
