from transformers import AutoTokenizer, AutoModel
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
base_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)


def embed_text(text, embedding_technique="cls", batch_size=400):
    """
    Embeds the given text using a specified embedding technique.
    Args:
        text Iterable(str): The input text to be embedded. Can be a list of texts.
        embedding_technique (str, optional): The technique to use for embedding. 
            Options are "mean" for mean pooling and "cls" for using the [CLS] token embedding. 
            Defaults to "mean".
    Returns:
        torch.Tensor: The embedded representation of the input text (bs, 768).
    """

    if isinstance(text, str):
        text = [text]
    
    all_outputs = []
    
    for i in range(0, len(text), batch_size):
        batch_texts = text[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = base_model.embeddings(input_ids=input_ids)
    
        if embedding_technique == "mean":
            all_outputs.append(torch.mean(outputs, dim=1))
        elif embedding_technique == "cls":
            all_outputs.append(outputs[:, 0, :])

    return torch.cat(all_outputs, dim=0).cpu().numpy()