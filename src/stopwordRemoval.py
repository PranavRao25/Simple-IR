from util import *

# Add your import statements here
import re
import json


class StopwordRemoval():

	def curated_list(self, text):
		with open("stopwords.json", "r") as f:
			stopwords = json.load(f)
		stopwords = stopwords["stopwords"]

		pattern = r"\b(" + r"|".join(stopwords) + r")\b"
		stop_rem = re.compile(pattern)
		cleaned_text = stop_rem.sub("", text)

		return cleaned_text
	
	def data_driven(self, text):
		


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




	