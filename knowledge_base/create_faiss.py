from text2vec import SentenceModel
import hnswlib
import numpy as np
import json
import glob
from Config import Config
import os
import chardet
import logging
import asyncio
import aiofiles
from typing import Optional
from rank_bm25 import BM25Okapi
import jieba
import jieba.analyse
'''
程序只对txt文件进行处理
'''

# shibing624/text2vec-base-chinese
# GanymedeNil/text2vec-large-chinese
# python create_faiss.py data model_document

args = Config()
# args.update({'data_path': sys.argv[1],'model_document': sys.argv[2]})
logging.basicConfig(filename=os.path.join(args.model_document, args.create_faiss_log), filemode='w', encoding='utf-8',
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def detect_encoding(file_path):
    async with aiofiles.open(file_path, 'rb') as f:
        raw_data = await f.read()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, chardet.detect, raw_data)
        return result['encoding']


async def read_file(path):
    try:
        encoding = await detect_encoding(path)
        async with aiofiles.open(path, 'r', encoding=encoding) as read:
            line = await read.readlines()
            line = ''.join(line)
            return line
    except Exception as e:
        logging.error(f'数据读取异常:{e}')
        return None


async def read_data(path_list, max_concurrent=50):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def read_limit_data(path):
        async with semaphore:
            return await read_file(path)

    tasks = [read_limit_data(path) for path in path_list]
    texts = await asyncio.gather(*tasks)
    return texts


# 创建Embedding向量知识库
def faiss_model(args, texts):
    model = SentenceModel(args.model)
    text_embeddings = model.encode(texts)
    text_embeddings = np.array(text_embeddings).astype('float32')
    dim = text_embeddings.shape[1]  # 向量的维度（例如：768维）
    num_elements = len(texts)
    text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    index = hnswlib.Index(space='l2', dim=dim)  # 使用 L2 距离
    index.init_index(max_elements=num_elements, ef_construction=200, M=32)
    index.add_items(text_embeddings)
    # 保存索引
    index.save_index(os.path.join(args.model_document, args.hnsw_index))


# 创建结巴分词方法，其中包括停用词库
class ChineseTokenizer:
    """基于 jieba 的中文分词器，支持停用词过滤"""
    with open('model_document/default_stopwords.txt', 'r', encoding='utf-8') as r:
        DEFAULT_STOPWORDS = [word.strip('\n') for word in r.readlines()]

    def __init__(self, stopwords: Optional[set] = None, min_len: int = 2):
        self.stopwords = stopwords if stopwords is not None else self.DEFAULT_STOPWORDS
        self.min_len = min_len

    def tokenize(self, text: str) -> list[str]:
        tokens = jieba.cut(text)
        return [
            t.strip() for t in tokens
            if t.strip()
               and len(t.strip()) >= self.min_len
               and t.strip() not in self.stopwords
        ]

    def extract_keywords(self, text: str, topk: int = 20) -> list[str]:
        """用 TF-IDF 提取关键词"""
        return jieba.analyse.extract_tags(text, topK=topk)


# 保存关键词库，后期对知识库搜索时，不用对文档进行分词。
def save_keyword_index(args):
    path = os.path.join(args.data_path, '*.txt')
    path_list = glob.glob(path)
    texts = asyncio.run(read_data(path_list))
    chinesetokenizer = ChineseTokenizer()
    docs = [(sentence, chinesetokenizer.extract_keywords(sentence)) for i, sentence in enumerate(texts)]
    with open('model_document/keyword_index.json', 'w', encoding='utf-8') as f:
        json.dump(docs, f, indent=4)


# BM25关键词搜索前TopN的知识文档
def bm25_keyword(query, top_n=5):
    with open('model_document/keyword_index.json', 'r', encoding='utf-8') as r:
        data = json.load(r)
    tokenized_corpus = [line[1] for line in data]
    chinesetokenizer = ChineseTokenizer()
    tokenized_query = chinesetokenizer.extract_keywords(query)
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    # scores = np.clip(scores, 0, None)
    # max_s = scores.max()
    # normalized_scores = scores / max_s if max_s > 0 else scores
    doc_score_pair = list(zip([line[0] for line in data], scores))
    results = sorted(doc_score_pair, key=lambda x: x[1], reverse=True)[:top_n]
    return results


def main(args):
    try:
        path = os.path.join(args.data_path, '*.txt')
        path_list = glob.glob(path)
        texts = asyncio.run(read_data(path_list))
        faiss_model(args, texts)
        save_keyword_index(args)  # 关键词索引
        # 保存文本数据
        with open(os.path.join(args.model_document, args.data_json), "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.info(f"运行情况：{e}")


if __name__ == '__main__':
    main(args)
# 运行命令：python create_faiss.py data model_document
