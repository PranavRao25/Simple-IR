from src.util import *

# Add your import statements here
import re
import nltk
import spacy
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

class SentenceSegmentation():

	def __init__(self):
		# Load spaCy model (students may use this if needed)
		self.nlp = spacy.load("en_core_web_sm")

	def naive(self, text: str)-> list[str]:
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		if not isinstance(text, str):
			raise TypeError("Need String input")
		
		segmentedText = None

		# Fill in code here
		pattern = r'[\.\?\!]+'  # BUG: Fix .gov, etc
		result = re.split(pattern, text)
		segmentedText = [sent for sent in result if sent is not None and len(sent)>0]

		return segmentedText

	def punkt(self, text: str)-> list[str]:
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
		
		if not isinstance(text, str):
			raise TypeError("Need String input")

		# TODO: Add more edge cases
		salutations = ["Mr", "Mrs", "Dr", "Ms", "St", "Col", "Gov"]
		websites = ["gov", "com", "org", "io", "edu"]
		punkt_params = PunktParameters()
		punkt_params.abbrev_types = set(salutations + websites)

		segmentor = PunktSentenceTokenizer(punkt_params)
		segmentedText = segmentor.tokenize(text)

		return segmentedText

	def spacySegmenter(self, text: str)-> list[str]:
		"""
		Sentence Segmentation using spaCy

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		# Fill in code here
		nlp = spacy.load("en_core_web_sm")
		doc = nlp(text)
		segmentedText = list([str(sent) for sent in doc.sents])

		return segmentedText
