# Requires transformers>=4.51.0
import torch
from modelscope import AutoModel, AutoTokenizer, AutoModelForCausalLM


'''
加载重排模型
'''
tokenizer = AutoTokenizer.from_pretrained("Qwen3_Reranker_06B", padding_side='left')
# model = AutoModelForCausalLM.from_pretrained("Qwen3_Reranker_06B").eval()
# We recommend enabling flash_attention_2 for better acceleration and memory saving.
model = AutoModelForCausalLM.from_pretrained("Qwen3_Reranker_06B", torch_dtype=torch.float16).cuda().eval()
token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 8192
prefix = "<|im_start|>system\n根据查询和指令判断文档是否符合要求。 注意，答案只能是 \"是\" 或 \"否\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = '检索相关段落以回答该查询'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,
                                                                                     query=query, doc=doc)
    return output


def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs


@torch.no_grad()
def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

#使用重排模型对结果进行重排。
def qwen3_rerank(query, documents,top_n):
    task = '检索相关段落以回答该查询'
    pairs = [format_instruction(task, query, doc) for doc in documents]
    # Tokenize the input texts
    inputs = process_inputs(pairs)
    scores = compute_logits(inputs)
    doc_score_pairs= list(zip(documents,scores))
    doc_score_pairs.sort(key=lambda x:x[1],reverse=True)
    top_results = doc_score_pairs[:top_n]

    return top_results

if __name__=='__main__':
    query='中国首都在哪？'
    documents=['北京是中国首都。','中国首都有八达岭长城。','天津市是直辖市。']
    top_results = qwen3_rerank(query, documents,2)
    print(top_results)