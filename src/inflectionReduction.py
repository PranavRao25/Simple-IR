from util import *

# Add your import statements here
# (Students may import required libraries such as nltk, WordNetLemmatizer, PorterStemmer, etc.)
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
import re
import spacy


class InflectionReduction:
	def porterStemmer(self, text: list[list[str]]) -> list[list[str]]:
		"""
		Inflection Reduction using Porter Stemmer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed tokens representing a sentence
		"""

		porter_stemmer = PorterStemmer()

		reducedText = []
		for words in text:
			stemmed_words = [porter_stemmer.stem(word) for word in words]
			reducedText.append(stemmed_words)

		return reducedText

	def wordnetLemmatizer(self, text: list[list[str]])-> list[list[str]]:
		"""
		Inflection Reduction using WordNet Lemmatizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			lemmatized tokens representing a sentence
		"""

		"""
			JJ : Adjective
			NN : Noun
			VB: Verb
			R: Adverb
			PRP: 
		"""

		def get_wordnet_tag(tag):
			if tag.startswith("JJ"):
				return 'a'
			elif tag.startswith("N"):
				return 'n'
			elif tag.startswith("R"):
				return 'r'
			elif tag.startswith("V"):
				return 'v'
			else:
				return 'n'
		
		lemmatizer = WordNetLemmatizer()

		reducedText = []
		for words in text:  # words is a list of strs (post-tokenization output)
			tags = pos_tag(words)  # (word, POS-tag) pair
			lemma_words = [lemmatizer.lemmatize(word, get_wordnet_tag(tag)) for word, tag in tags]
			reducedText.append(lemma_words)

		return reducedText

	def reduce(self, text):
		"""
		Wrapper function for inflection reduction.
		Students may choose which method to call
		or extend this function to support both options.
		"""

		reducedText = None

		# Stemming is simple, fast but bad performance
		# Lemmatization is complex, slow but good performance

		# Heuristically decide on the length of text

		if len(text) < 20:
			reducedText = self.porterStemmer(text)
		else:
			reducedText = self.wordnetLemmatizer(text)

		return reducedText
