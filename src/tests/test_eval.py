import numpy as np
import pandas as pd
import yaml
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
import random

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
		# print(no_relevant)
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
		# print(len(present))

		DCG = np.sum([(2**(int(docs_k[i] in present)) - 1)/np.log(i+1) for i in range(1, len(docs_k))])
		IDCG = len(present) * np.sum([1/np.log(i+1) for i in range(1, len(present)+1)])

		if IDCG == 0:
			print(f"{query_id} has 0 IDCG  and {len(present)} at {k}")
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
		# present = set(query_doc_IDs_ordered[k]).intersection(set(true_doc_IDs))

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

parent = Path(__file__).resolve().parent.parent.parent

with open(parent / "config.yaml", "r") as f:
    configs = yaml.full_load(f)

queries_json = json.load(open(parent / "cranfield/cran_queries.json", 'r'))[:]
query_ids = [item["query number"] for item in queries_json]
queries = [item["query"] for item in queries_json]

docs_json = json.load(open(parent / "cranfield/cran_docs.json", 'r'))[:]
doc_ids = [item["id"] for item in docs_json]
docs = [item["body"] for item in docs_json]

qrels = json.load(open(parent / "cranfield/cran_qrels.json", 'r'))[:]

precisions, recalls, fscores = [], [], []
MAPs, nDCGs, MRRs = [], [], []

# self.informationRetriever.buildIndex(processedDocs, doc_ids)
# doc_IDs_ordered = self.informationRetriever.rank(processedQueries)

df = pd.DataFrame(qrels)
df["query_num"] = df["query_num"].astype(str)
str_query_ids = [str(q_id) for q_id in query_ids]
df = df[df["query_num"].isin(str_query_ids)]
true_docs_map = df.groupby("query_num")["id"].apply(set).to_dict()

ids = set(df["id"])
# print(ids)

doc_IDs_ordered = dict((int(query_id), []) for query_id in query_ids)
for query_id in str_query_ids:
	# print(query_id)
	true = true_docs_map.get(query_id, set())
	# print(true)
	false = list(ids.difference(true))
	# print(false)
	true_doc_IDs = list(true_docs_map.get(query_id, set()))
	doc_IDs_ordered[int(query_id)] += true_doc_IDs[:5]
	doc_IDs_ordered[int(query_id)] += false[:5]
	# print(doc_IDs_ordered[int(query_id)])

with open("docs_id.json", "w") as f:
	json.dump(doc_IDs_ordered, f)
# print(docs_df)
# print(sorted(df[df["query_num"] == "1"]["id"]))
# print(true_docs_map['1'])

evaluator = Evaluation()

# k = 5
# query_id = random.choice(query_ids)
# print(query_id)
# print(doc_IDs_ordered["52"])
# print(list(true_docs_map.get(query_id, set())))

query_id = random.choice(str_query_ids)
query_doc_IDs_ordered: list = doc_IDs_ordered[int(query_id)]  # dict[int] = list
true_doc_IDs = list(true_docs_map.get(query_id, set()))

print(f"{query_id} related lists:")
print(query_doc_IDs_ordered)
print(true_doc_IDs)

for k in range(1, 11):
    # precision = evaluator.queryPrecision(query_doc_IDs_ordered, int(query_id), true_doc_IDs, k)
    # recall = evaluator.queryRecall(query_doc_IDs_ordered, int(query_id), true_doc_IDs, k)
    # fscore = evaluator.queryFscore(query_doc_IDs_ordered, int(query_id), true_doc_IDs, k)
	
    # print(f"Precision, Recall, F-score @ {k}: {precision}, {recall}, {fscore}")

    # ndcg = evaluator.queryNDCG(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
    # ap = evaluator.queryAveragePrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
    # rr = evaluator.queryReciprocalRank(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

    # print(f"ap, ndcg, rr @ {k}: {ap} {rr} {ndcg}")

    macro_precision = evaluator.meanPrecision(doc_IDs_ordered, query_ids, qrels, k)
    macro_recall = evaluator.meanRecall(doc_IDs_ordered, query_ids, qrels, k)
    macro_fscore = evaluator.meanFscore(doc_IDs_ordered, query_ids, qrels, k)

    precisions.append(macro_precision)
    recalls.append(macro_recall)
    fscores.append(macro_fscore)

    # print(f"Precision, Recall, F-score @ {k}: {macro_precision}, {macro_recall}, {macro_fscore}")

    MAP = evaluator.meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k)
    nDCG = evaluator.meanNDCG(doc_IDs_ordered, query_ids, qrels, k)
    MRR = evaluator.meanReciprocalRank(doc_IDs_ordered, query_ids, qrels, k)

    MAPs.append(MAP)
    nDCGs.append(nDCG)
    MRRs.append(MRR)

    # print(f"MAP, nDCG, MRR @ {k}: {MAP}, {nDCG}, {MRR}")

# Plot

print(precisions)
print(recalls)
print(fscores)

print(MAPs)
print(nDCGs)
print(MRRs)

plt.plot(range(1, 11), precisions, label="Precision")
plt.plot(range(1, 11), recalls, label="Recall")
plt.plot(range(1, 11), fscores, label="F-Score")
plt.plot(range(1, 11), MAPs, label="MAP")
plt.plot(range(1, 11), nDCGs, label="nDCG")
plt.plot(range(1, 11), MRRs, label="MRR")
plt.legend()
plt.title("Evaluation Metrics - Cranfield Dataset")
plt.xlabel("k")
plt.savefig(parent / "output/eval_plot.png")
