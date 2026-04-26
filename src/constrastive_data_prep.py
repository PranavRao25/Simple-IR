import json
import pandas as pd
import itertools
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import yaml

parent = Path(__file__).resolve().parent.parent
with open(parent / "config.yaml", "r") as f:
    configs = yaml.full_load(f)

qrels = json.load(open(parent / configs["dataset"]["qrels"], 'r'))[:]
queries = json.load(open(parent / configs["dataset"]["queries"], 'r'))[:]
docs = json.load(open(parent / configs["dataset"]["docs"], 'r'))[:]

queries = {query["query number"]: query["query"] for query in queries}
docs = {doc["id"]: doc["body"] + doc["author"] + doc["bibliography"] for doc in docs}
qrel_df = pd.DataFrame(qrels)

def create_constrastive_queries_dataset():
    negative_queries = []
    positive_queries = []
    grouped = qrel_df.groupby('id')
    for doc_id, group in grouped:
        pos_groups = group.groupby('position')['query_num'].apply(list).to_dict()
        
        # similar queries
        for p_id, grp_queries in pos_groups.items():
            if len(grp_queries) > 1:
                for q1, q2 in itertools.combinations(grp_queries, 2):  # itertools.permutations(queries, 2)
                    q1, q2, doc_id = int(q1), int(q2), int(doc_id) # type: ignore
                    positive_queries.append({
                        "q": queries[q1],
                        "sim_q": queries[q2],
                        "sim_doc": docs[doc_id],
                        "sim_pos": p_id,
                        "qid": q1,
                        "sim_qid": q2,
                        "sim_doc_id": doc_id
                    })
                    
        # dissimiliar queries
        positions = list(pos_groups.keys())
        if len(positions) > 1:
            for pos1, pos2 in itertools.combinations(positions, 2):  # itertools.permutations(queries, 2)
                queries1 = pos_groups[pos1]
                queries2 = pos_groups[pos2]
                
                for q1, q2 in itertools.product(queries1, queries2):
                    q1, q2, doc_id = int(q1), int(q2), int(doc_id) # type: ignore
                    negative_queries.append({
                        "q": queries[q1],
                        "dsim_q": queries[q2],
                        "dsim_doc": docs[doc_id],
                        "dsim_pos": pos1,
                        "dsim_pos": pos2,
                        "qid": q1,
                        "dsim_qid": q2,
                        "dsim_doc_id": doc_id
                    })

    pos_df = pd.DataFrame(positive_queries)
    neg_df = pd.DataFrame(negative_queries).drop("q", axis=1)
    
    on = "qid"
    return negative_queries, positive_queries, pd.merge(pos_df, neg_df, on=on)

def create_constrastive_docs_dataset():
    negative_docs = []
    positive_docs = []
    grouped = qrel_df.groupby('query_num')
    for q_id, group in grouped:
        pos_groups = group.groupby('position')['id'].apply(list).to_dict()
        
        # similar docs
        for p_id, grp_docs in pos_groups.items():
            if len(grp_docs) > 1:
                for d1, d2 in itertools.combinations(grp_docs, 2):  # itertools.permutations(queries, 2)
                    d1, d2, q_id = int(d1), int(d2), int(q_id) # type: ignore
                    positive_docs.append({
                        "doc": docs[d1],
                        "sim_doc": docs[d2],
                        "sim_q": queries[q_id],
                        "sim_pos": p_id,
                        "did": d1,
                        "sim_did": d2,
                        "sim_qid": q_id
                    })
                    
        # dissimiliar docs
        positions = list(pos_groups.keys())
        if len(positions) > 1:
            for pos1, pos2 in itertools.combinations(positions, 2):  # itertools.permutations(queries, 2)
                docs1 = pos_groups[pos1]
                docs2 = pos_groups[pos2]
                
                for d1, d2 in itertools.product(docs1, docs2):
                    d1, d2, q_id = int(d1), int(d2), int(q_id) # type: ignore
                    negative_docs.append({
                        "doc": docs[d1],
                        "dsim_doc": docs[d2],
                        "dsim_q": queries[q_id],
                        "dsim_pos": pos1,
                        "dsim_pos": pos2,
                        "did": d1,
                        "dsim_did": d2,
                        "dsim_qid": q_id
                    })

    pos_df = pd.DataFrame(positive_docs)
    neg_df = pd.DataFrame(negative_docs).drop("doc", axis=1)
    on = "did"
    return negative_docs, positive_docs, pd.merge(pos_df, neg_df, on=on)

if __name__ == "__main__":
    negative_queries, positive_queries, triplet_queries = create_constrastive_queries_dataset()

    with open(parent / configs["dataset"]["contrastive"]["queries"]["negative"], "w") as f:
        json.dump(negative_queries, f)
    with open(parent / configs["dataset"]["contrastive"]["queries"]["positive"], "w") as f:
        json.dump(positive_queries, f)
    
    negative_docs, positive_docs, triplet_docs = create_constrastive_docs_dataset()

    with open(parent / configs["dataset"]["contrastive"]["docs"]["negative"], "w") as f:
        json.dump(negative_docs, f)
    with open(parent / configs["dataset"]["contrastive"]["docs"]["positive"], "w") as f:
        json.dump(positive_docs, f)
    
    triplet_queries.to_csv(parent / configs["dataset"]["triplet"]["queries"], index=False)
    triplet_docs.to_csv(parent / configs["dataset"]["triplet"]["docs"], index=False)
