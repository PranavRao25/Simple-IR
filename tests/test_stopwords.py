import pytest
import yaml
from pathlib import Path
import json

def test_stopwords():
    sentences = [
        "an, am, can, do, we",
        "Roosevelt",
        "????",
        "1 and 2 and 3",
        "A pencil with an eraser.",
        "an_an_an_an_an",
        "We did it ourselves again.",
        ""
    ]
