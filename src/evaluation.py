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

	# DONE
	def meanPrecision(self, doc_IDs_ordered: dict, query_ids: list, qrels: list, k: int):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		docs_IDs_ordered : list
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

		def map_query(query_id):
			query_doc_IDs_ordered = doc_IDs_ordered[int(query_id)]
			true_doc_IDs = list(true_docs_map.get(query_id, set()))
			return self.queryPrecision(query_doc_IDs_ordered, int(query_id), true_doc_IDs, k)

		#Fill in code here
		mean_Precision = -1
		df = pd.DataFrame(qrels)
		df["query_num"] = df["query_num"].astype(str)
		str_query_ids = [str(q_id) for q_id in query_ids]
		df = df[df["query_num"].isin(str_query_ids)]
		true_docs_map = df.groupby("query_num")["id"].apply(set).to_dict()
		mean_Precision = float(np.average(np.fromiter(map(map_query, str_query_ids), dtype=float)))

		return mean_Precision
	
	# DONE
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query
		"""
		recall = -1

		#Fill in code here

		if len(true_doc_IDs) == 0:
			print(f"{query_id} has 0 true_doc_IDs")
		
		total_k = query_doc_IDs_ordered[:k]
		no_relevant = len(set(total_k).intersection(set(true_doc_IDs)))
		print(no_relevant)
		recall = 0 if no_relevant == 0 else no_relevant / len(true_doc_IDs)

		return recall

	# DONE
	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""

		def map_query(query_id):
			query_doc_IDs_ordered = doc_IDs_ordered[int(query_id)]
			true_doc_IDs = list(true_docs_map.get(query_id, set()))
			return self.queryRecall(query_doc_IDs_ordered, int(query_id), true_doc_IDs, k)

		#Fill in code here
		mean_Recall = -1
		df = pd.DataFrame(qrels)
		df["query_num"] = df["query_num"].astype(str)
		str_query_ids = [str(q_id) for q_id in query_ids]
		df = df[df["query_num"].isin(str_query_ids)]
		true_docs_map = df.groupby("query_num")["id"].apply(set).to_dict()
		mean_Recall = float(np.average(np.fromiter(map(map_query, str_query_ids), dtype=float)))

		return mean_Recall

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

		if (precision + recall) == 0:
			return 0
		
		fscore = 2 * precision * recall / (precision + recall)
		return fscore

	# DONE
	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""

		def map_query(query_id):
			query_doc_IDs_ordered = doc_IDs_ordered[int(query_id)]
			true_doc_IDs = list(true_docs_map.get(query_id, set()))
			return self.queryPrecision(query_doc_IDs_ordered, int(query_id), true_doc_IDs, k)

		#Fill in code here
		mean_Fscore = -1
		df = pd.DataFrame(qrels)
		df["query_num"] = df["query_num"].astype(str)
		str_query_ids = [str(q_id) for q_id in query_ids]
		df = df[df["query_num"].isin(str_query_ids)]
		true_docs_map = df.groupby("query_num")["id"].apply(set).to_dict()
		mean_Fscore = float(np.average(np.fromiter(map(map_query, str_query_ids), dtype=float)))

		return mean_Fscore
	
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
		present = set(docs_k).intersection(set(true_doc_IDs))
		print(len(present))

		DCG = np.sum([(2**(int(docs_k[i] in present)) - 1)/np.log(i+1) for i in range(1, len(docs_k))])
		IDCG = len(present) * np.sum([1/np.log(i+1) for i in range(1, len(present)+1)])

		if IDCG == 0:
			print(f"{query_id} has 0 IDCG at {k}")
			return 0
		
		nDCG = DCG/IDCG
		return nDCG

	# DONE
	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""

		def map_query(query_id):
			query_doc_IDs_ordered = doc_IDs_ordered[int(query_id)]
			true_doc_IDs = list(true_docs_map.get(query_id, set()))
			return self.queryNDCG(query_doc_IDs_ordered, int(query_id), true_doc_IDs, k)

		#Fill in code here
		mean_NDCG = -1
		df = pd.DataFrame(qrels)
		df["query_num"] = df["query_num"].astype(str)
		str_query_ids = [str(q_id) for q_id in query_ids]
		df = df[df["query_num"].isin(str_query_ids)]
		
		true_docs_map = df.groupby("query_num")["id"].apply(set).to_dict()
		mean_NDCG = float(np.average(np.fromiter(map(map_query, str_query_ids), dtype=float)))

		return mean_NDCG

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
		for i in range(1, k):
			precision_i = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i)
			i = min(i-1, len(query_doc_IDs_ordered) - 1)
			rel_i = int(query_doc_IDs_ordered[i] in true_docs)
			avgPrecision += (rel_i * precision_i)

		avgPrecision = 0 if len(true_doc_IDs) == 0 else avgPrecision/len(true_doc_IDs)

		return avgPrecision

	# DONE
	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries
		"""

		def map_query(query_id):
			query_doc_IDs_ordered = doc_IDs_ordered[int(query_id)]
			true_doc_IDs = list(true_docs_map.get(query_id, set()))
			return self.queryAveragePrecision(query_doc_IDs_ordered, int(query_id), true_doc_IDs, k)

		#Fill in code here
		mean_AP = -1
		
		df = pd.DataFrame(q_rels)
		df["query_num"] = df["query_num"].astype(str)
		str_query_ids = [str(q_id) for q_id in query_ids]
		df = df[df["query_num"].isin(str_query_ids)]
		
		true_docs_map = df.groupby("query_num")["id"].apply(set).to_dict()
		mean_AP = float(np.average(np.fromiter(map(map_query, str_query_ids), dtype=float)))

		#Fill in code here
		# m_ap = []
		# 	ap = 0
		# 	for i in range(k):
		# 		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i)
		# 		rel = query_doc_IDs_ordered[i] in q_df[q_df["pos"] == i]["id"]
		# 		ap += precision * rel
		# 	ap /= len(query_doc_IDs_ordered)
		# 	m_ap.append(ap)
		# mean_Precision = np.average(m_ap)

		return mean_AP

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

		if k > len(query_doc_IDs_ordered):
			print(f"k : {k} is bigger than query_docs : {len(query_doc_IDs_ordered)}")
		
		k = min(k-1, len(query_doc_IDs_ordered)-1)
		present = [query_doc_IDs_ordered[k]] if query_doc_IDs_ordered[k] in set(true_doc_IDs) else []

		if len(present) == 0:
			print(f"{query_id} has 0 present at {k}")
			reciprocalRank = 0
		else:
			reciprocalRank = 1/(query_doc_IDs_ordered.index(min(present)) + 1)

		return reciprocalRank

	# DONE
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

		def map_query(query_id):
			query_doc_IDs_ordered = doc_IDs_ordered[int(query_id)]
			true_doc_IDs = list(true_docs_map.get(query_id, set()))
			return self.queryReciprocalRank(query_doc_IDs_ordered, int(query_id), true_doc_IDs, k)

		#Fill in code here
		mRR = -1
		df = pd.DataFrame(qrels)
		df["query_num"] = df["query_num"].astype(str)
		str_query_ids = [str(q_id) for q_id in query_ids]
		df = df[df["query_num"].isin(str_query_ids)]
		
		true_docs_map = df.groupby("query_num")["id"].apply(set).to_dict()
		mRR = float(np.average(np.fromiter(map(map_query, str_query_ids), dtype=float)))

		return mRR
