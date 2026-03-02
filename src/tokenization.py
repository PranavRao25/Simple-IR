from util import *

# Add your import statements here
# (Students may import required libraries such as nltk, spacy, re, etc.)
import nltk
from nltk.tokenize import TreebankWordTokenizer
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import re

"""
from nltk.tokenize import word_tokenize

# Same as TreeBankTokenizer
text = "Hello everyone. Welcome to GeeksforGeeks."
word_tokenize(text)
"""
"""
from nltk.tokenize import WordPunctTokenizer

tokenizer = WordPunctTokenizer()
tokenizer.tokenize("Let's see how it's working.")  # includes punctuation
"""
"""
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\W+')
text = "Let's see how it's working."
tokenizer.tokenize(text)
"""

"""
Types of Tokenization:
1. Word tokenization -
	a. Punctuation-based tokenization
	b. Space-based tokenization
	c. Rule-based tokenization
2. Character tokenization
3. Sentence Tokenization
4. Subword Tokenization -
	a. SentencePiece
	b. Unigram
	c. WordPiece
	d. Byte-Pair Encoding (BPE)

Dictionary based Tokenization
Spacy Tokenizer
Moses Tokenizer
Unigram Language Model
"""


class Tokenization():

	def naive(self, text: list[str]) -> list[list[str]]:
		"""
		Tokenization using a Naive Approach


		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		# Tokenize each sentence
		# Word Tokenization / Sub-word tokenization / N-gram tokenization

		tokenizedText = None

		# Fill in code here

		return tokenizedText

	def pennTreeBank(self, text: list[str]) -> list[list[str]]:
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		

		tokenizer = TreebankWordTokenizer()
		tokenizedText = [tokenizer.tokenize(string) for string in text]  # includes punctuation

		return tokenizedText
	
	def spacyTokenizer(self, text) -> list[list[str]]:
		"""
		Tokenization using spaCy

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		nlp = English()
		tokenizer = Tokenizer(nlp.vocab)

		# Fill in code here
		tokenizedText = [tokenizer(string) for string in text]

		return tokenizedText
