from transformers import AutoTokenizer, AutoModel

import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
base_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)


def embed_text(text, embedding_technique="mean"):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = base_model.embeddings(input_ids=input_ids)
    
    if embedding_technique == "mean":
        outputs = torch.mean(outputs, dim=1)
    elif embedding_technique == "cls":
        outputs = outputs[:, 0, :]
    
    return outputs