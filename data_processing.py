import json
from datasets import Dataset
from sklearn.model_selection import train_test_split
from utils import format_examples, tokenize_function

def load_data(file_path):
    """
    Load dataset from a JSON file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def prepare_datasets(data, tokenizer, test_size=0.1, seed=42):
    """
    Prepare train and validation datasets from raw data
    """
    # Format examples for training
    formatted_examples = format_examples(data)
    
    # Create dataset
    dataset = Dataset.from_list(formatted_examples)
    
    # Split dataset into train and validation
    train_val_datasets = dataset.train_test_split(test_size=test_size, seed=seed)
    train_data = train_val_datasets['train']
    val_data = train_val_datasets['test']
    
    # Tokenize the datasets
    train_tokenized = train_data.map(
        lambda examples: tokenize_function(tokenizer, examples),
        batched=True,
        remove_columns=["text"]
    )
    
    val_tokenized = val_data.map(
        lambda examples: tokenize_function(tokenizer, examples),
        batched=True,
        remove_columns=["text"]
    )
    
    return train_tokenized, val_tokenized 