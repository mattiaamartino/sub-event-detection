from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F

import torch

# Set up device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizer and the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("model_output/checkpoint-288718", use_fast=True)
finetuned_model = DistilBertForSequenceClassification.from_pretrained("model_output/checkpoint-288718").to(device)

# Define a function to extract outputs up to the pre-classification layer
def get_pre_classifier_output(text, batch_size=400):
    """
    Extracts the output of the model up to the pre-classification layer with batch processing.
    
    Args:
        text (str or List[str]): Input text or a batch of texts
        batch_size (int): Size of batches to process, default 700
    
    Returns:
        torch.Tensor: The output of the pre-classification layer (bs, dim)
    """
    # Handle single string input
    if isinstance(text, str):
        text = [text]
    
    # Initialize list to store batch outputs
    all_outputs = []
    
    # Process in batches
    for i in range(0, len(text), batch_size):
        batch_texts = text[i:i + batch_size]
        
        # Tokenize the batch
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Forward pass through the encoder
        with torch.no_grad():
            distilbert_output = finetuned_model.distilbert(**inputs)
            hidden_state = distilbert_output.last_hidden_state
            pooled_output = hidden_state[:, 0, :]
            pre_classified_output = finetuned_model.pre_classifier(pooled_output)
            
        all_outputs.append(pre_classified_output)
    
    # Concatenate all batch outputs
    return torch.cat(all_outputs, dim=0)


def run_second_part_of_model(pre_classified_output):
    """
    Executes the second part of the model starting from the pre-classification output.

    Args:
        pre_classified_output (torch.Tensor): The output from the pre-classification layer (bs, dim).
        model (DistilBertForSequenceClassification): The fine-tuned model.
    
    Returns:
        torch.Tensor: The logits from the classification layer (bs, num_labels).
    """
    # Ensure no gradients are calculated
    with torch.no_grad():
        # Apply ReLU activation
        activated_output = F.relu(pre_classified_output)  # (bs, dim)

        # Apply dropout
        dropped_output = finetuned_model.dropout(activated_output)  # (bs, dim)

        # Pass through the final classifier layer
        logits = finetuned_model.classifier(dropped_output)  # (bs, num_labels)
    
    return logits