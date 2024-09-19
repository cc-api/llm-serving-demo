#!/bin/bash

# Step 1: Fine-tuning RAG model
echo "Fine-tuning the RAG model..."
python src/finetune_rag.py

# Step 2: Serving the model using FastAPI
echo "Starting FastAPI server..."
uvicorn src.rag_api:app --reload

