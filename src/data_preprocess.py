import os
import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, random_split, DataLoader


def load_data(file_path):
    """ Load CSV or Excel data and return a dataframe """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or XLSX.")

    return df



def create_corpus_file(file_path, text_column, corpus_file_name):
    """
    Given a file path, text column name, and output corpus file path, this function
    extracts text data and saves it as a corpus file.
    
    Args:
        file_path (str): Path to the input CSV or XLSX file.
        text_column (str): Name of the column containing the text data.
        corpus_file_path (str): Path to save the generated corpus file.
    """
    df = load_data(file_path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the file.")

    text_data = df[text_column].dropna().tolist() 

    corpus_file_path = f"experiments/{corpus_file_name}.txt"
    with open(corpus_file_path, 'w', encoding='utf-8') as f:
        for text in text_data:
            f.write(f"{text.strip()}\n")

    print(f"Corpus file saved at: {corpus_file_path}")



class ReportDataset(Dataset):
    """ Custom Dataset for loading accident/maintenance reports """
    def __init__(self, report, labels, tokenizer, max_length=128):
        self.report = report
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.report)

    def __getitem__(self, idx):
        report = self.report[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            report, 
            padding='max_length', 
            max_length=self.max_length, 
            truncation=True, 
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0) if 'token_type_ids' in encoding else torch.zeros(self.max_length, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


def preprocess_data(file_path, text_column, class_column, tokenizer, label_map, test_split=0.1, val_split=0.1):
    """
    Loads data, preprocesses it, and splits it into train, validation, and test datasets.
    """
    df = load_data(file_path)
    
    if f"{text_column}" not in df.columns or f"{class_column}" not in df.columns:
        raise ValueError("Dataset must contain 'description' and 'class' columns.")

    descriptions = df[text_column].fillna("No description").tolist()
    labels = df[class_column].map(label_map).tolist()

    full_dataset = ReportDataset(descriptions, labels, tokenizer)

    test_size = int(test_split * len(full_dataset))
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - (test_size + val_size)

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset



def get_tokenizer(tokenizer_type="pretrained", vocab_file=None):
    """
    Returns a tokenizer:
    - "pretrained" uses BertTokenizer from Hugging Face (for pre-trained BERT)
    - "custom" uses BertWordPieceTokenizer (for custom BERT training).
      If vocab_file is not provided, it trains the tokenizer using the provided corpus.
    """
    if tokenizer_type == "pretrained":
        return BertTokenizer.from_pretrained("bert-base-uncased")
    
    elif tokenizer_type == "custom":   # If vocab file is not provided, train the tokenizer on the corpus
        if vocab_file is None:
            raise ValueError("You must provide a vocab file for custom tokenizer.")
        
        return BertTokenizer(vocab_file)

    else:
        raise ValueError("Invalid tokenizer type. Choose 'pretrained' or 'custom'.")




def create_dataloader(train_dataset, val_dataset, test_dataset, batch_size=int):
    """
    Creates DataLoaders for the train, validation, and test datasets.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

