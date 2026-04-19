from util import *

# Add your import statements here
# (Students may import required libraries such as nltk, spacy, re, etc.)
import nltk
from nltk.tokenize import TreebankWordTokenizer, WordPunctTokenizer
import spacy
from spacy.tokenizer import Tokenizer
import re

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

Difference between WordPunctTokenizer and TreeBankTokenizer:
1. WordPunctTokenizer - splits I'm as I / ' / m
2. TreeBankTokenizer - splits I'm as I / 'm
"""


class Tokenization():

	def naive(self, text: list[str]) -> list[list[str]]:
		"""
		Word Level Tokenization using a Naive Approach

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

		tokenizer = WordPunctTokenizer() # uses regex expression : \w+|[^\w\s]+

		tokenizedText = [tokenizer.tokenize(sent) for sent in text]  # includes punctuation

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

		nlp = spacy.load('en_core_web_sm')
		
		# Fill in code here
		tokenizedText = []
		for string in text:
			doc = nlp(string)
			l = [str(token) for token in doc]
			tokenizedText.append(l)

		return tokenizedText
