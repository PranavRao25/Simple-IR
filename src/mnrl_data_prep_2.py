import json
import random
import pandas as pd
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple

random.seed(42)

def load_json(filepath: str) -> List[Dict]:
    with open(filepath, 'r') as f:
        return json.load(f)

def tokenize(text: str) -> List[str]:
    """Basic whitespace tokenization for BM25. 
    For production, consider a lightweight stemmer (e.g., PorterStemmer)."""
    return text.lower().split()

def generate_triplets(
    docs_path: str, 
    queries_path: str, 
    qrels_path: str, 
    num_hard_negatives: int = 5,
    top_k_bm25: int = 50
) -> pd.DataFrame:
    
    raw_docs = load_json(docs_path)
    raw_queries = load_json(queries_path)
    raw_qrels = load_json(qrels_path)

    doc_index = {
        str(doc['id']): f"{doc.get('title', '')} {doc.get('body', '')}".strip() 
        for doc in raw_docs
    }
    
    query_index = {
        str(q['query number']): q['query'].strip() 
        for q in raw_queries
    }

    positive_mapping = {}
    for qrel in raw_qrels:
        qid = str(qrel['query_num'])
        did = str(qrel['id'])
        
        if qid not in positive_mapping:
            positive_mapping[qid] = set()
        positive_mapping[qid].add(did)

    doc_ids = list(doc_index.keys())
    tokenized_corpus = [tokenize(doc_index[did]) for did in doc_ids]
    
    print("Building BM25 Index...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    triplets: List[Tuple[str, str, str]] = []
    
    print("Mining Hard Negatives...")
    for qid, positive_doc_ids in positive_mapping.items():
        if qid not in query_index:
            continue
            
        anchor_text = query_index[qid]
        tokenized_query = tokenize(anchor_text)
        
        top_k_scores = bm25.get_scores(tokenized_query)
        
        top_k_indices = sorted(
            range(len(top_k_scores)), 
            key=lambda i: top_k_scores[i], 
            reverse=True
        )[:top_k_bm25]
        
        retrieved_doc_ids = [doc_ids[i] for i in top_k_indices]
        
        hard_negative_candidates = [
            did for did in retrieved_doc_ids 
            if did not in positive_doc_ids
        ]
        
        for pos_id in positive_doc_ids:
            if pos_id not in doc_index:
                continue
                
            positive_text = doc_index[pos_id]
            
            sampled_negatives = random.sample(
                hard_negative_candidates, 
                min(num_hard_negatives, len(hard_negative_candidates))
            )
            
            for neg_id in sampled_negatives:
                negative_text = doc_index[neg_id]
                triplets.append((anchor_text, positive_text, negative_text))

    df = pd.DataFrame(triplets, columns=['anchor', 'positive', 'negative'])
    return df

if __name__ == "__main__":
    df_triplets = generate_triplets(
        docs_path="cranfield/cran_docs.json",
        queries_path="cranfield/cran_queries.json",
        qrels_path="cranfield/cran_qrels.json",
        num_hard_negatives=2,
        top_k_bm25=50
    )
    
    print(f"Dataset generated with {len(df_triplets)} triplets.")
    
    df_triplets.to_parquet("cranfield/cranfield_infonce_triplets.parquet", index=False)

