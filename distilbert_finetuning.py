import pandas as pd
from datasets import Dataset, DatasetDict
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def create_huggingface_dataset(train_data_folder):
    train_data = []
    test_data = []

    for file_name in tqdm(os.listdir(train_data_folder)):
        df = pd.read_csv(os.path.join(train_data_folder, file_name))
        df.drop(columns=['MatchID', 'PeriodID', 'Timestamp'], inplace=True, errors='ignore')
        #Here we split the dataset so to take 20% of the data for each period in each match
        for id_val, group in df.groupby("ID"):
            train_group, test_group = train_test_split(group, test_size=0.2, random_state=42)
            train_data.append(train_group)
            test_data.append(test_group)
    
    train_df = pd.concat(train_data).reset_index(drop=True)
    test_df = pd.concat(test_data).reset_index(drop=True)

    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Create DatasetDict
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    return dataset


dataset = create_huggingface_dataset("cleaned_data/train_data")

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import numpy as np

import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["Tweet"], truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

training_args = TrainingArguments(
    output_dir="model_output",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    use_mps_device=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()