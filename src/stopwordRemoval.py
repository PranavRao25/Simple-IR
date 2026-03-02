from util import *

# Add your import statements here
import re
import numpy as np
from nltk.corpus import stopwords
from collections import defaultdict

class StopwordRemoval():

	def curated_list(self, text: list[list[str]]):
		stop_words = stopwords.words('english')

		pattern = r"\b(" + r"|".join(stop_words) + r")\b"
		stop_rem = re.compile(pattern)

		stopwordRemovedText = [[stop_rem.sub("", sent) for sent in doc] for doc in text]
		return stopwordRemovedText
	
	def data_driven(self, text: list[list[str]])-> list[list[str]]:
		"""
			Remove words with low relevance
			Relevance - measure of a word's importance to a document
			in relation to a larger collection of documents

			Assume ```text``` is a list of documents
		"""

		# Accumulate all words
		terms = []
		for doc in text:
			terms += doc

		# Create a frequency table
		freq_table = defaultdict(int)
		for doc in text:
			for term in doc:
				term = term.lower()
				freq_table[term] += 1
		
		# Normalize according to the highest frequency
		most = max(freq_table.values())
		for term in freq_table.keys():
			freq_table[term] = freq_table[term] - most
		
		# convert frequency into probability using softmax
		softmax_table = defaultdict(float)
		for term in freq_table.keys():
			softmax_table[term] = freq_table[term] / sum(freq_table.values())

		threshold = 0.05  # first 5% is termed as stopwords
		stopwords = [entry[0] for entry in softmax_table.items() if entry[-1] <= threshold]

		# remove words
		stopwordRemovedText = [[term for term in doc if term not in stopwords] for doc in text]

		return stopwordRemovedText


	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = None

		#Fill in code here

		return stopwordRemovedText
