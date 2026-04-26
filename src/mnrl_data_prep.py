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

def create_mnrl_queries_dataset():
    """
        Create a Triplet dataset with
        Anchor: Question
        Positive: Related Document with a position
        Negative(s): Unrelated Document with no position
    """

    total_docs = set(qrel_df["id"])
    grouped_qrels = qrel_df.groupby("query_num")

    for query_num, group in grouped_qrels:
        query_id = int(query_num) # type: ignore
        q_text = queries[query_id]
        pos_groups = group.groupby("position")["id"].apply(list).to_dict()
        positions = list(pos_groups.keys())

        # get the pairs which are not related
        related_docs = set(group["id"])
        unrelated_docs = total_docs - related_docs

        if related_docs and unrelated_docs:
            for r_doc, u_doc in itertools.product(related_docs, unrelated_docs):
                u_doc, r_doc = int(u_doc), int(r_doc)
                yield {
                    "q": q_text,
                    "pos": docs[r_doc],
                    "neg": docs[u_doc],
                    "qid": query_id,
                    "r_id": r_doc,
                    "u_id": u_doc
                }

def create_hard_mnrl_queries_dataset():
    """
        Create a Triplet dataset with
        Anchor: Question
        Positive: Related Document with a position
        Negative(s): Unrelated Document with no position
        PositivePosition: How related
        NegativePosition: How unrelated
    """

    total_docs = set(qrel_df["id"])
    grouped_qrels = qrel_df.groupby("query_num")

    for query_num, group in grouped_qrels:
        query_id = int(query_num) # type: ignore
        q_text = queries[query_id]
        pos_groups = group.groupby("position")["id"].apply(list).to_dict()

        # get the pairs which are not related
        related_docs = set(group["id"])
        most_relevant_docs = set(pos_groups.get(1, []))
        unrelated_docs = total_docs - related_docs

        if most_relevant_docs and unrelated_docs:
            for r_doc, u_doc in itertools.product(most_relevant_docs, unrelated_docs):
                u_doc, r_doc = int(u_doc), int(r_doc)
                yield {
                    "q": q_text,
                    "pos": docs[r_doc],
                    "neg": docs[u_doc],
                    "qid": query_id,
                    "r_id": r_doc,
                    "u_id": u_doc
                }

if __name__ == "__main__":
    mnrl_qs = create_mnrl_queries_dataset()
    dump_q_data(mnrl_qs, parent / configs["dataset"]["infonce"], chunk_size=1000)

    hard_mnrl_qs = create_hard_mnrl_queries_dataset()
    dump_q_data(hard_mnrl_qs, parent / configs["dataset"]["hard_infonce"], chunk_size=1000)
