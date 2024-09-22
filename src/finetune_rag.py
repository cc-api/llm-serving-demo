from transformers import RagTokenForGeneration, Trainer, TrainingArguments
import torch
from datasets import load_dataset
from utils import load_knowledge_data, prepare_retriever

# 加载数据集
data_path = "data/knowledge_dataset.json"
dataset = load_knowledge_data(data_path)


# 如果数据集没有分割，进行分割
if "train" not in dataset:
    dataset = dataset.train_test_split(test_size=0.1)

# 加载模型和检索器
model_name = "facebook/rag-token-base"
tokenizer, retriever = prepare_retriever(model_name)

# 定义模型
model = RagTokenForGeneration.from_pretrained(model_name, retriever=retriever)

# Tokenization
def tokenize_function(examples):
    inputs = tokenizer(
            examples["question"],
            padding="max_length",
            truncation=True,
            max_length=512
            )
    labels = tokenizer(
            examples["answer"],
            padding="max_length",
            truncation=True,
            max_length=512
            ).input_ids
    inputs["labels"] = labels
    return inputs

# 数据 tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./model/finetuned_model",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
)

# Trainer 设置
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

# 开始微调
trainer.train()

# 保存微调模型
model.save_pretrained("./model/finetuned_model")
tokenizer.save_pretrained("./model/finetuned_model")
