# vim ~/mineru.json
'''
{
    "models-dir": {
        "pipeline": "/root/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1___0",
        "vlm": "/root/.cache/modelscope/hub/models/OpenDataLab/MinerU2___5-Pro-2604-1___2B"
    }
}
'''
# export MINERU_MODEL_SOURCE=local
# mineru-api --host 0.0.0.0 --port 8515
import requests
import argparse
from text2vec import SentenceModel
import hnswlib
import numpy as np
import json
import glob9
import os
import chardet
import logging
from typing import Optional
from rank_bm25 import BM25Okapi
import jieba
import jieba.analyse


# MinerU识别pdf文件的，并将识别结果以md文件形式保存到result文件夹中

def pdf_mineru_md(mineru_url, pdf_path):
    basename = os.path.basename(pdf_path)
    # 2. 准备要上传的文件
    # 使用字典格式，键为表单字段名 'files'，值为一个元组 (文件名, 文件对象)
    files = {'files': (basename, open(pdf_path, 'rb'), 'application/pdf')}
    # 3. 准备其他的表单字段数据
    data = {'return_md': 'true'}
    # 4. 发送 POST 请求
    response = requests.post(mineru_url, files=files, data=data)
    # 5. 处理响应结果
    os.makedirs('result', exist_ok=True)
    result = json.loads(response.text)
    name = result['file_names'][0]
    result_text = result['results'][name]['md_content']
    return result_text
    # with open(f"result/{name}.md", "w", encoding='utf-8') as out:
    #     out.write(result_text)


# 按照标题切分到文本
def split_paragraph(content):
    content_list = [line for line in content.split('\n') if line]
    data_list = []
    data = []
    content_len = len(content_list)
    for i, line in enumerate(content_list):
        data.append(line.replace('#', ''))
        if i + 1 < content_len and not line.startswith('#') and content_list[i + 1].startswith('#'):
            data_list.append(data.copy())
            data = []
    if data:
        data_list.append(data)
    return data_list


# 按一定行数切分文件
def split_line(content, count):
    content_list = [line.replace('#', '') for line in content.split('\n') if line]
    data_list = []
    for i in range(0, len(content_list), count):
        data_list.append(content_list[i:i + count])
    return data_list


# 对pdf文件进行批量解析
def mineru_pdf_txt(args):
    pdf_path_list = glob.glob(os.path.join(args.pdf_path, '*.pdf'))
    txt_list = []
    for pdf_path in pdf_path_list:
        text = pdf_mineru_md(args.mineru_url, pdf_path)
        print(os.path.basename(pdf_path),'解析完成')
        txt_list.append(text)

    data = []
    for md_txt in txt_list:
        data += split_paragraph(md_txt)
        data += split_line(md_txt, args.spilt_line_count)

    return [''.join(text) for text in data]


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
def save_keyword_index(texts):
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
        texts = mineru_pdf_txt(args)
        print('pdf全部解析完毕')
        faiss_model(args, texts)
        save_keyword_index(texts)  # 关键词索引
        print('知识库构建完毕')
        # 保存文本数据
        with open(os.path.join(args.model_document, args.data_json), "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.info(f"运行情况：{e}")


if __name__ == '__main__':
    args = argparse.ArgumentParser('MinerU识别pdf,并将识别后的文本按页和行数两种方式切分')
    args.add_argument('--model',type=str,default="text2vec-base-chinese",required=False, help='model path')
    args.add_argument('--model_dim',type=int,default=768,required=False, help='model dim')
    args.add_argument('--topk',type=int,default=5,required=False, help='top k')
    args.add_argument('--emantic_weight',type=float,default=0.4,required=False, help='emantic weight')
    args.add_argument('--threshold',type=float,default=0.2,required=False, help='threshold')
    args.add_argument('--keyword_weight',type=float,default=0.4,required=False, help='keyword weight')
    args.add_argument('--exact_match_weight',type=float,default=0.2,required=False, help='exact match weight')
    args.add_argument('--model_document',default='model_document',type=str, required=False, help='model document')
    args.add_argument('--data_json',default='documents.json',type=str, required=False, help='data json')
    args.add_argument('--hnsw_index',default='hnsw_index.bin',type=str, required=False, help='hnsw_index')
    args.add_argument('--create_faiss_log',default='create_faiss_log.log',type=str, required=False, help='create_faiss_log')
    args.add_argument('--mineru_url', type=str, default="http://192.168.0.1:8515/file_parse")
    args.add_argument('--md_path', default='result', type=str, required=False, help='md path')
    args.add_argument('--pdf_path', default='o_data', type=str, required=False, help='pdf path')
    args.add_argument('--spilt_line_count', default=22, type=int, required=False, help='spilt line count')
    args = args.parse_args()
    main(args)

