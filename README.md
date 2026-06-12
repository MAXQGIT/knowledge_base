# 知识库使用方法
## 整体介绍

1. 实现了批量pdf文件自动识别，自动解析，自动构建知识库的流程，具体运行create_knowledge_faiss.py程序即可实现这些功能。

2. 实现了智能问答的全过程，并且默认支持5轮的多轮对话的。

## 技术架构流程图

![技术架构流程图](https://github.com/MAXQGIT/knowledge_base/blob/master/knowledge_base/%E5%9F%BA%E4%BA%8E%E5%90%91%E9%87%8F%E6%99%BA%E8%83%BD%E4%BD%93%E6%8A%80%E6%9C%AF%E6%B5%81%E7%A8%8B.png)

## 使用到的模型

1.自然语言大模型:Qwen/Qwen3.5-35B-A3B

下载链接：https://www.modelscope.cn/models/Qwen/Qwen3.5-35B-A3B

2.处理pdf模型: MinerU

下载链接：https://github.com/opendatalab/MinerU

启动命令:

(1) export MINERU_MODEL_SOURCE=local      

(2) mineru-api --host 0.0.0.0 --port 8515
更多启动方式介绍: https://opendatalab.github.io/MinerU/usage/quick_usage/#quick-usage-via-command-line

3.Embedding向量模型:text2vec-base-chinese

下载链接：https://www.modelscope.cn/models/Jerry0/text2vec-base-chinese/files

4.rerank模型：Qwen/Qwen3-Reranker-0.6B

下载地址: https://www.modelscope.cn/models/Qwen/Qwen3-Reranker-0.6B

## 启动方式

1.将需要pdf文件，都放在o_data文件夹中，然后运行create_knowledge_faiss.py，程序将自动构建知识库。

2. 运行app.py便能在页面打开，进行问答。

## 环境配置

1.由于都是比较新的模型，运行程序时，缺少什么库直接pip install ** 即可。

## 硬件配置

1.Qwen_35B_A3B 需要100G显存。

2.MinerU运行起来峰值现存占用是11G

3.Embedding模型和Rerank模型一张8G显存的显卡就可部署使用。

## 内网部署方式

1.docker方式部署。

2.裸金属部署，conda pack 打包环境即可。

3.MinerU部署需要配置一个文件，下面是配置文件操作方式

vim ~/mineru.json

'''
{
    "models-dir": {
        "pipeline": "/root/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1___0",
        "vlm": "/root/.cache/modelscope/hub/models/OpenDataLab/MinerU2___5-Pro-2604-1___2B"
    }
}
'''
