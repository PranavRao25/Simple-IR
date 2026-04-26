from util import *

# Add your import statements here
import numpy as np
import pandas as pd

class Evaluation():
	"""
		Performs Evaluation on the retrieved documents
	"""

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

		#Fill in code here
		precision = -1
		total_k = query_doc_IDs_ordered[:k]
		no_relevant = len(set(true_doc_IDs).intersection(set(total_k)))
		precision = no_relevant / k

		return precision

	def meanPrecision(self, doc_IDs_ordered: list, query_ids: list, qrels: list, k: int):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		docs_IDs_ordered : dict
			A dictionary of query ids to their list of integers denoting the IDs of documents in
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
			query_doc_IDs_ordered = doc_IDs_ordered[int(query_id)-1]  # get the predicted order
			true_doc_IDs = list(true_docs_map.get(query_id, set()))  # get the true order
			return self.queryPrecision(query_doc_IDs_ordered, int(query_id), true_doc_IDs, k)

		#Fill in code here
		mean_Precision = -1
		df = pd.DataFrame(qrels)  # qrels has true relevant lists
		df["query_num"] = df["query_num"].astype(int)
		df["id"] = df["id"].astype(int)
		true_docs_map = df.groupby("query_num")["id"].apply(set).to_dict()
		mean_Precision = float(np.average(np.fromiter(map(map_query, query_ids), dtype=float)))

		return mean_Precision
	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query
		"""

		recall = -1

		if len(true_doc_IDs) == 0:
			print(f"{query_id} has 0 true_doc_IDs")
		
		total_k = query_doc_IDs_ordered[:k]
		no_relevant = len(set(total_k).intersection(set(true_doc_IDs)))
		# print(no_relevant)
		recall = 0 if no_relevant == 0 else no_relevant / len(true_doc_IDs)

		return recall

	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""

		def map_query(query_id):
			query_doc_IDs_ordered = doc_IDs_ordered[int(query_id)-1]
			true_doc_IDs = list(true_docs_map.get(query_id, set()))
			return self.queryRecall(query_doc_IDs_ordered, int(query_id), true_doc_IDs, k)

		#Fill in code here
		mean_Recall = -1
		df = pd.DataFrame(qrels)  # qrels has true relevant lists
		df["query_num"] = df["query_num"].astype(int)
		df["id"] = df["id"].astype(int)
		true_docs_map = df.groupby("query_num")["id"].apply(set).to_dict()
		mean_Recall = float(np.average(np.fromiter(map(map_query, query_ids), dtype=float)))

		return mean_Recall

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

	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""

		def map_query(query_id):
			query_doc_IDs_ordered = doc_IDs_ordered[int(query_id)-1]
			true_doc_IDs = list(true_docs_map.get(query_id, set()))
			return self.queryPrecision(query_doc_IDs_ordered, int(query_id), true_doc_IDs, k)

		#Fill in code here
		mean_Fscore = -1
		df = pd.DataFrame(qrels)  # qrels has true relevant lists
		df["query_num"] = df["query_num"].astype(int)
		df["id"] = df["id"].astype(int)
		true_docs_map = df.groupby("query_num")["id"].apply(set).to_dict()
		mean_Fscore = float(np.average(np.fromiter(map(map_query, query_ids), dtype=float)))

		return mean_Fscore
	
	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs: list, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query
		"""
		nDCG = -1

		#Fill in code here
		docs_k = query_doc_IDs_ordered[:k]
		present = set(true_doc_IDs).intersection(set(docs_k))

		if len(present) == 0:
			print(f"Present {query_id}: {set(docs_k)} {set(true_doc_IDs)}")

		DCG = np.sum([(2**(int(docs_k[i] in present)) - 1)/np.log(i+1) for i in range(1, len(docs_k))])
		IDCG = len(present) * np.sum([1/np.log(i+1) for i in range(1, len(present)+1)])

		if IDCG == 0:
			print(f"{query_id} has 0 IDCG at {k}")
			return 0
		
		nDCG = DCG/IDCG
		return nDCG

	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""

		def map_query(query_id):
			query_doc_IDs_ordered = doc_IDs_ordered[int(query_id)-1]
			true_doc_IDs = list(true_docs_map.get(query_id, set()))
			return self.queryNDCG(query_doc_IDs_ordered, int(query_id), true_doc_IDs, k)

		#Fill in code here
		mean_NDCG = -1
		df = pd.DataFrame(qrels)  # qrels has true relevant lists
		df["query_num"] = df["query_num"].astype(int)
		df["id"] = df["id"].astype(int)
		true_docs_map = df.groupby("query_num")["id"].apply(set).to_dict()
		mean_NDCG = float(np.average(np.fromiter(map(map_query, query_ids), dtype=float)))

		return mean_NDCG

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

	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries
		"""

		def map_query(query_id):
			query_doc_IDs_ordered = doc_IDs_ordered[int(query_id)-1]
			true_doc_IDs = list(true_docs_map.get(query_id, set()))
			return self.queryAveragePrecision(query_doc_IDs_ordered, int(query_id), true_doc_IDs, k)

		#Fill in code here
		mean_AP = -1
		
		df = pd.DataFrame(qrels)  # qrels has true relevant lists
		df["query_num"] = df["query_num"].astype(int)
		df["id"] = df["id"].astype(int)
		true_docs_map = df.groupby("query_num")["id"].apply(set).to_dict()
		mean_AP = float(np.average(np.fromiter(map(map_query, query_ids), dtype=float)))

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
		
		k = min(k, len(query_doc_IDs_ordered))

		true_doc_IDs_set = set(true_doc_IDs)
		min_index = next((i for i in range(k) if query_doc_IDs_ordered[i] in true_doc_IDs_set), k)

		if min_index == k:
			print(f"{query_id} has 0 present at {k}")
			print(query_doc_IDs_ordered[:k], true_doc_IDs)
			print(set(query_doc_IDs_ordered[:k]).intersection(true_doc_IDs_set))
			reciprocalRank = 0
		else:
			reciprocalRank = 1/(query_doc_IDs_ordered[min_index] + 1)

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

		def map_query(query_id):
			query_doc_IDs_ordered = doc_IDs_ordered[int(query_id)-1]
			true_doc_IDs = list(true_docs_map.get(query_id, set()))
			return self.queryReciprocalRank(query_doc_IDs_ordered, int(query_id), true_doc_IDs, k)

		#Fill in code here
		mRR = -1
		df = pd.DataFrame(qrels)  # qrels has true relevant lists
		df["query_num"] = df["query_num"].astype(int)
		df["id"] = df["id"].astype(int)
		true_docs_map = df.groupby("query_num")["id"].apply(set).to_dict()
		mRR = float(np.average(np.fromiter(map(map_query, query_ids), dtype=float)))

		return mRR