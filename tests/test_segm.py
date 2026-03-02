import pytest
from src.sentenceSegmentation import *

seg = SentenceSegmentation()
docs = docs_loader()

sentences = [
    "Hello, There. Where we do we go now? Great!",  # VANILLA
    "",
    "Hello now",
    "....",
    "!!!!",
    ".?!",
    "iitm.ac.in",  # BREAK 1
    "Mrs. Hercules",  # BREAK 2
    "What????"  # BREAK 3
    "This paper was written by Grover et al. and was published in 1992.",  # BREAK 4
    "You owe me $15.32 plus some biscuits."  # BREAK 5
    "And He said, \"What a wonderful world!\".", # BREAK 6
    ".hack//SIGN is such a great show!", # BREAK 7
    "", # BREAK 8
    "", # BREAK 9
    "", # BREAK 10
    "", # BREAK 11
    "", # BREAK 12
    "", # BREAK 13
    "", # BREAK 14
    "", # BREAK 15
]

outputs = [
    ['Hello, There', ' Where we do we go now', ' Great'],
    [],
    ['Hello now'],
    [],
    [],
    [],
    ["iitm.ac.in"],
    ["Mrs. Hercules"],
    ["What"],
    ["This paper was written by Grover et al. and was published in 1992"],
    ["You owe me $15.32 plus some biscuits"],
    ["And He said, \"What a wonderful world!\""],
    [".hack//SIGN is such a great show!"]
]

def test_naive():
    for (sent, output) in zip(sentences, outputs):
        assert output == seg.naive(sent)

    with pytest.raises(TypeError):
        seg.naive(1)

def test_punkt():
    for (sent, output) in zip(sentences, outputs):
        assert output == seg.punkt(sent)

    with pytest.raises(TypeError):
        seg.punkt(1)

def test_spacy():
    for (sent, output) in zip(sentences, outputs):
        assert output == seg.spacySegmenter(sent)

    with pytest.raises(TypeError):
        seg.spacySegmenter(1)
