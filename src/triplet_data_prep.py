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

def create_triplet_queries_dataset():
    """
        Create a Triplet dataset with
        Anchor: Question
        Positive: Related Document
        Negative: Unrelated Document
        PositivePosition: How related
        NegativePosition: How unrelated
    """

    total_docs = set(qrel_df["id"])
    grouped_qrels = qrel_df.groupby("query_num")

    for query_num, group in grouped_qrels:
        query_id = int(query_num) # type: ignore
        q_text = queries[query_id]
        pos_groups = group.groupby("position")["id"].apply(list).to_dict()
        positions = list(pos_groups.keys())

        # get the pairs within the ranked
        if len(positions) > 1:
            for pos1, pos2 in itertools.combinations(positions, 2):  # itertools.permutations(queries, 2)
                doc1 = pos_groups[pos1]
                doc2 = pos_groups[pos2]
                
                for d1, d2 in itertools.product(doc1, doc2):
                    d1, d2 = int(d1), int(d2)
                    yield {
                        "q": q_text,
                        "pos": docs[d1],
                        "neg": docs[d2],
                        "qid": query_id,
                        "r_id": d1,
                        "u_id": d2,
                        "pos_r": pos1,
                        "pos_u": pos2
                    }

        # get the pairs which are not related
        related_docs = set(group["id"])
        unrelated_docs = total_docs - related_docs

        if related_docs and unrelated_docs:
            for u_doc, r_doc in itertools.product(unrelated_docs, related_docs):
                u_doc, r_doc = int(u_doc), int(r_doc)
                yield {
                    "q": q_text,
                    "pos": docs[r_doc],
                    "neg": docs[u_doc],
                    "qid": query_id,
                    "r_id": r_doc,
                    "u_id": u_doc,
                    "pos_r": -1,
                    "pos_u": -1
                }

def create_triplet_docs_dataset():
    """
        Create a Triplet dataset with
        Anchor: Doc
        Positive: Related Query
        Negative: Unrelated Query
        PositivePosition: How related
        NegativePosition: How unrelated
    """

    total_queries = set(qrel_df["query_num"])
    grouped_qrels = qrel_df.groupby("id")

    for doc_num, group in grouped_qrels:
        doc_id = int(doc_num) # type: ignore
        d_text = docs[doc_id]
        pos_groups = group.groupby("position")["query_num"].apply(list).to_dict()
        positions = list(pos_groups.keys())

        # get the pairs within the ranked
        if len(positions) > 1:
            for pos1, pos2 in itertools.combinations(positions, 2):  # itertools.permutations(queries, 2)
                qs1 = pos_groups[pos1]
                qs2 = pos_groups[pos2]
                
                for q1, q2 in itertools.product(qs1, qs2):
                    q1, q2 = int(q1), int(q2)
                    yield {
                        "doc": d_text,
                        "pos": queries[q1],
                        "neg": queries[q2],
                        "doc_id": doc_id,
                        "r_id": q1,
                        "u_id": q2,
                        "pos_r": pos1,
                        "pos_u": pos2
                    }

        # get the pairs which are not related
        related_qs = set(group["query_num"])
        unrelated_qs = total_queries - related_qs

        if related_qs and unrelated_qs:
            for u_q, r_q in itertools.product(unrelated_qs, related_qs):
                u_q, r_q = int(u_q), int(r_q)
                yield {
                    "doc": d_text,
                    "pos": queries[r_q],
                    "neg": queries[u_q],
                    "doc_id": doc_id,
                    "r_id": r_q,
                    "u_id": u_q,
                    "pos_r": -1,
                    "pos_u": -1
                }

def dump_d_data(gen, file_path, chunk_size=500):
    schema = pa.schema([
        ('doc', pa.string()),
        ('pos', pa.string()),
        ('neg', pa.string()),
        ('doc_id', pa.int64()),
        ('r_id', pa.int64()),
        ('u_id', pa.int64()),
        ('pos_r', pa.int64()),
        ('pos_u', pa.int64())
    ])
    
    with pq.ParquetWriter(file_path, schema) as writer:
        chunk = []
        for triplet in gen:
            chunk.append(triplet)
            if len(chunk) >= chunk_size:
                table = pa.Table.from_pylist(chunk, schema=schema)
                writer.write_table(table)
                chunk = []
        if chunk:
            table = pa.Table.from_pylist(chunk, schema=schema)
            writer.write_table(table)

def dump_q_data(gen, file_path, chunk_size=500):
    schema = pa.schema([
        ('q', pa.string()),
        ('pos', pa.string()),
        ('neg', pa.string()),
        ('qid', pa.int64()),
        ('r_id', pa.int64()),
        ('u_id', pa.int64()),
    ])
    
    with pq.ParquetWriter(file_path, schema) as writer:
        chunk = []
        for triplet in gen:
            chunk.append(triplet)
            if len(chunk) >= chunk_size:
                table = pa.Table.from_pylist(chunk, schema=schema)
                writer.write_table(table)
                chunk = []
        if chunk:
            table = pa.Table.from_pylist(chunk, schema=schema)
            writer.write_table(table)

if __name__ == "__main__":
    triplet_qs = create_triplet_queries_dataset()
    triplet_docs = create_triplet_docs_dataset()

    dump_d_data(triplet_docs, parent / configs["dataset"]["triplet"]["docs"], chunk_size=1000)
    dump_q_data(triplet_qs, parent / configs["dataset"]["triplet"]["queries"], chunk_size=1000)
