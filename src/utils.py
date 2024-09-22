from datasets import Dataset, load_dataset
from transformers import RagTokenizer, RagRetriever
import torch

def load_knowledge_data(file_path):
    """加载本地知识库数据"""
    from datasets import load_dataset
    dataset = load_dataset('json', data_files=file_path)
    return dataset

def prepare_retriever(model_name):
    """初始化 RAG 检索器"""
    tokenizer = RagTokenizer.from_pretrained(model_name)
    retriever = RagRetriever.from_pretrained(
        model_name,
        index_name="exact",
        use_dummy_dataset=True,
        #passages_path="data/psgs_w100.tsv",  # 您的本地文档路径
        #index_path="data/my_index.faiss",    # 您的本地索引路径
        dataset_kwargs={"trust_remote_code": True}
    )
    return tokenizer, retriever
