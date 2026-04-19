from util import *

# Add your import statements here
import math

class InformationRetrieval():

	def __init__(self):
		self.index = dict()
		self.doc_lengths = {}

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		index = dict()

		#Fill in code here
		num_docs = len(docs)
		
		for i in range(num_docs):
			doc = docs[i]
			doc_id = docIDs[i]
			
			num_words = 0
			for sent in doc:
				for word in sent:
					if word not in index:
						index[word] = {}
					
					if doc_id not in index[word]:
						index[word][doc_id] = 0

					index[word][doc_id] += 1
					num_words += 1

			self.doc_lengths[doc_id] = num_words
		
		self.index = index
		self.total_docs = num_docs

	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		#Fill in code here
		
		for query in queries:

			tf_idf_scores = dict()
			for sent in query:
				for word in sent:					
					if word in self.index:
						doc_freqs = self.index[word]
						
						for doc_id, freq in doc_freqs.items():

							doc_length = self.doc_lengths[doc_id]
							tf = freq / doc_length
							idf = math.log(self.total_docs / len(doc_freqs))
							tf_idf = tf * idf

							if doc_id not in tf_idf_scores:
								tf_idf_scores[doc_id] = 0
							tf_idf_scores[doc_id] += tf_idf
						
			sorted_docs = sorted(tf_idf_scores, key=tf_idf_scores.get, reverse=True) # type: ignore
			doc_IDs_ordered.append(sorted_docs)
		return doc_IDs_ordered



