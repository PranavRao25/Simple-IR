from util import *

# Add your import statements here
import numpy as np
import pandas as pd

class Evaluation():
	# DONE
	def queryPrecision(self, query_doc_IDs_ordered: list, query_id: int, true_doc_IDs: list, k: int):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		query_docs_IDs_ordered : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		query_id : int
			The ID of the query in question
		true_doc_IDs : list
			The list of IDs of documents relevant to the query (ground truth)
		k : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here
		total_k = query_doc_IDs_ordered[:k]
		no_relevant = len(set(total_k).intersection(set(true_doc_IDs)))
		precision = no_relevant / k

		return precision

	def meanPrecision(self, doc_IDs_ordered: list, query_ids: list, qrels: list, k: int):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		query_docs_IDs_ordered : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		query_ids : int
			The IDs of the queries in question
		qrels : list
			list[dict]
		k : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""
		meanPrecision = -1

		#Fill in code here
		df = pd.DataFrame(qrels)
		df[df["query_num"].isin(query_ids)]

		return meanPrecision
	
	# DONE
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query
		"""
		recall = -1

		#Fill in code here
		total_k = query_doc_IDs_ordered[:k]
		no_relevant = len(set(total_k).intersection(set(true_doc_IDs)))
		recall = no_relevant / len(true_doc_IDs)

		return recall

	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		meanRecall = -1

		#Fill in code here

		return meanRecall

	# DONE
	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query
		"""
		fscore = -1

		#Fill in code here
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		fscore = 2 * precision * recall / (precision + recall)

		return fscore

	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		meanFscore = -1

		#Fill in code here

		return meanFscore
	
	# DONE
	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs: list, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query
		"""
		nDCG = -1

		#Fill in code here

		# TODO: how to calculate the relevance scores of a doc to the query
		# not there: 0
		# there: 1

		docs_k = query_doc_IDs_ordered[:k]
		true_docs = set(true_doc_IDs)
		present = set(docs_k).intersection(true_docs)

		DCG = np.sum([(2**(int(docs_k[i] in present)) - 1)/np.log(i+1) for i in range(len(docs_k))])
		IDCG = len(present) * np.sum([1/np.log(i+1) for i in range(1, len(present))])
		nDCG = DCG/IDCG

		return nDCG

	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		meanNDCG = -1

		#Fill in code here

		return meanNDCG

	# DONE
	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)
		"""
		avgPrecision = -1

		#Fill in code here
		true_docs = set(true_doc_IDs)

		avgPrecision = 0
		for i in range(0, k):
			precision_i = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i)
			rel_i = int(query_doc_IDs_ordered[i] in true_docs)
			avgPrecision += (rel_i * precision_i)
		avgPrecision /= len(true_doc_IDs)

		return avgPrecision

	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries
		"""
		meanAveragePrecision = -1

		#Fill in code here

		return meanAveragePrecision

	# DONE
	def queryReciprocalRank(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of reciprocal rank for a single query

		Parameters
		----------
		query_doc_IDs_ordered : list
			Ranked list of document IDs
		query_id : int
			Query ID
		true_doc_IDs : list
			List of relevant document IDs
		k : int
			The k value

		Returns
		-------
		float
			Reciprocal rank value
		"""

		reciprocalRank = -1

		#Fill in code here
		present = set(query_doc_IDs_ordered[k]).intersection(set(true_doc_IDs))
		first = min(present)
		reciprocalRank = 1/(query_doc_IDs_ordered.index(first) + 1)

		return reciprocalRank

	def meanReciprocalRank(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of Mean Reciprocal Rank (MRR)
		averaged over all queries

		Parameters
		----------
		arg1 : list
			List of ranked document lists
		arg2 : list
			Query IDs
		arg3 : list
			Relevance judgments
		arg4 : int
			The k value

		Returns
		-------
		float
			MRR value
		"""

		meanReciprocalRank = -1

		#Fill in code here

		return meanReciprocalRank
