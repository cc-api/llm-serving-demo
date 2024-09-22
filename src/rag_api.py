from fastapi import FastAPI, Request
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
import torch

# 加载微调后的模型和检索器
model = RagTokenForGeneration.from_pretrained("./model/finetuned_model")
tokenizer = RagTokenizer.from_pretrained("./model/finetuned_model")

# 设置为 CPU 或 GPU
device = torch.device("cpu")
model.to(device)

# 创建 FastAPI 实例
app = FastAPI()

# 定义生成文本的路由
@app.post("/generate")
async def generate_answer(request: Request):
    data = await request.json()
    question = data['question']
    
    # Tokenize 输入
    inputs = tokenizer(question, return_tensors="pt").to(device)
    
    # 生成答案
    with torch.no_grad():
        output = model.generate(**inputs)
    
    # 解码生成的文本
    generated_answer = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    
    return {"answer": generated_answer}

# 启动服务器
# 运行命令: uvicorn rag_api:app --reload
