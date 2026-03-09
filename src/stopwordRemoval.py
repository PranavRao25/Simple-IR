from util import *

# Add your import statements here
import re
import numpy as np
from nltk.corpus import stopwords
import json
from collections import defaultdict

class StopwordRemoval():

	def curated_list(self, text: list[list[str]]):
		stop_words = set(stopwords.words('english'))

		stopwordRemovedText = [
			[token for token in sent if token not in stop_words]
			for sent in text
		]

		# pattern = r"\b(" + r"|".join(stop_words) + r")\b"
		# stop_rem = re.compile(pattern)

		# stopwordRemovedText = [[stop_rem.sub("", sent) for sent in doc] for doc in text]

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
		freq_table = dict()
		for doc in text:
			for term in doc:
				term = term.strip().lower()  # remove spaces and lowercase
				if len(term) > 0:
					if not term[0].isalpha():  # if first element is not alpha
						term = term[1:]
					if len(term) > 1 and not term[-1].isalpha():  # if last element is not alpha
						term = term[:-1]

					if term not in freq_table:
						freq_table[term] = 1
					else:
						freq_table[term] = freq_table[term] + 1
		
		# Normalize according to the highest frequency
		# most = max(freq_table.values())
		# for term in freq_table.keys():
		# 	freq_table[term] = freq_table[term] - most

		# convert frequency into probability using softmax
		softmax_table = defaultdict(float)
		for term in freq_table.keys():
			softmax_table[term] = freq_table[term] / sum(freq_table.values())
		
		all_stopwords = []
		l = len(softmax_table.keys())
		if l > 0:
			threshold = 0.15
			if l != len(softmax_table.values()):
				print(l, len(softmax_table.values()))
			
			threshold_probab = sorted(list(softmax_table.values()), reverse=True)[int(threshold * l)]
			stopwords = [entry[0] for entry in softmax_table.items() if entry[-1] >= threshold_probab]
			all_stopwords.append(stopwords)
			# remove words
			stopwordRemovedText = [[term for term in doc if term not in stopwords] for doc in text]
		else:
			stopwordRemovedText = text

		with open("stopwords.json", "r") as f:
			k = json.load(f)
		k = k + all_stopwords
		with open("stopwords.json", "w") as f:
			json.dump(k, f)

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
		stopwordRemovedText = self.curated_list(text)
		# stopwordRemovedText = self.data_driven(text)

		return stopwordRemovedText
