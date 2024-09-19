from datasets import Dataset, load_dataset
from transformers import RagTokenizer, RagRetriever
import torch

def load_knowledge_data(file_path):
    """加载本地知识库数据"""
    dataset = load_dataset('json', data_files=file_path)
    return dataset

def prepare_retriever(model_name, index_name="faiss"):
    """初始化 RAG 检索器"""
    tokenizer = RagTokenizer.from_pretrained(model_name)
    retriever = RagRetriever.from_pretrained(
            model_name, index_name=index_name, use_dummy_dataset=True
    )
    return tokenizer, retriever

