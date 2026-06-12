import hnswlib
from Config import Config
import json
import os
from text2vec import SentenceModel
from create_faiss import bm25_keyword
import numpy as np
from Qwen3_rerank import qwen3_rerank

args = Config()
with open(os.path.join(args.model_document, args.data_json), 'r', encoding='utf-8') as f:
    documents = json.load(f)

sentence_model = SentenceModel('text2vec-base-chinese')
hnsw_index = hnswlib.Index(space='l2', dim=args.model_dim)
hnsw_index.load_index(os.path.join(args.model_document, args.hnsw_index))
hnsw_index.set_ef(200)

'''
Embeddding模型和BM25关键词联合搜索，使用RRF和重排模型进行搜索。
'''


# Embedding模型搜索知识，并返回前TopK的知识
def search_embedding_answer(input_text):
    query_embeddings = sentence_model.encode([input_text], device='cuda')
    query_embeddings /= np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    labels, distances = hnsw_index.knn_query(query_embeddings, k=int(args.topk))
    embedding_score_pair = list(zip([documents[i] for i in labels[0]], distances[0]))
    return embedding_score_pair


'''
RRF对BM25关键词搜索和Embedding搜索答案权重后重排
其中bm25_weight是BM25关键词搜索答案权重
embedding_weight是Embedding搜索答案权重
'''


def rrf_fusion(input_text, bm25_weight=0.4, embedding_weight=0.6, k=60, top_n=10):
    bm25_results = bm25_keyword(input_text, top_n)  # BM25关键词搜索前topn
    embedding_results = search_embedding_answer(input_text)  # Embedding搜索前topn

    scores = {}
    for rank, (text, _) in enumerate(bm25_results):
        key = text
        if key not in scores:
            scores[key] = {'rrf': 0.0, 'text': text}
        scores[key]['rrf'] += bm25_weight * 1 / (k + rank + 1)
    for rank, (text, _) in enumerate(embedding_results):
        key = text
        if key not in scores:
            scores[key] = {'rrf': 0.0, 'text': text}
        scores[key]['rrf'] += embedding_weight * 1 / (k + rank + 1)
    faused = sorted(scores.values(), key=lambda x: x['rrf'], reverse=True)
    faused = qwen3_rerank(input_text, [item['text'] for item in faused], 5)
    return [line[0] for line in faused]


if __name__ == '__main__':
    args = Config()
    input_text = '中长期交易边界条件是什么'
    faused = rrf_fusion(input_text)
    print(faused)
