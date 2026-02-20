import pytest
from src.sentenceSegmentation import *

def test_naive():
    seg = SentenceSegmentation()
    docs = docs_loader()

    sentences = [
        "Hello, There. Where we do we go now? Great!",
        "",
        "Hello now",
        "....",
        "!!!!",
        ".?!"
    ]

    outputs = [
        ['Hello, There', ' Where we do we go now', ' Great'],
        [],
        ['Hello now'],
        [],
        [],
        []
    ]

    for (sent, output) in zip(sentences, outputs):
        assert output == seg.naive(sent)

    with pytest.raises(TypeError):
        seg.naive(1)
    