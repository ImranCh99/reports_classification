import os
import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, random_split, DataLoader



def load_data(file_path=None, df=None):
    """ 
    Load CSV or Excel data into a DataFrame.
    If a file path is provided, it loads from the file.
    If a DataFrame is provided, it returns the given DataFrame.
    """
    if df is not None:
        return df  # Use provided DataFrame directly

    if file_path is None:
        raise ValueError("Either file_path or df must be provided.")

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, skiprows=3)  # Handles aviation dataset properly
    else:
        raise ValueError("Unsupported file format. Use CSV or XLSX.")

    return df



def create_corpus_file(file_path=None, df=None, text_column=None, corpus_file_name=None):
    """
    Extracts text data from a dataset (CSV/XLSX file or DataFrame) and saves it as a corpus file.

    Args:
        file_path (str, optional): Path to the input CSV or XLSX file.
        df (pd.DataFrame, optional): Preloaded DataFrame.
        text_column (str): Name of the column containing the text data.
        corpus_file_name (str): Name of the output corpus file (without extension).
    """
    df = load_data(file_path=file_path, df=df)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the dataset.")

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


def preprocess_data(file_path=None, df=None, text_column=None, class_column=None, tokenizer=None, label_map=None, test_split=0.1, val_split=0.1):
    """
    Loads data (from file or provided DataFrame), preprocesses it, and splits into train, validation, and test datasets.
    """
    df = load_data(file_path=file_path, df=df)

    if text_column not in df.columns or class_column not in df.columns:
        raise ValueError(f"Dataset must contain '{text_column}' and '{class_column}' columns.")

    descriptions = df[text_column].fillna("No description").tolist()
    labels = df[class_column].map(label_map).astype(int).tolist()

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

