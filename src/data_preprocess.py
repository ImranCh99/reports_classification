import os
import pandas as pd
import torch
from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
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
        text = str(self.report[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }



def preprocess_data(file_path, tokenizer, test_split=0.1, val_split=0.1):
    """
    Loads data, preprocesses it, and splits it into train, validation, and test datasets.
    """
    df = load_data(file_path)
    
    # Ensure columns exist
    if "description" not in df.columns or "class" not in df.columns:
        raise ValueError("Dataset must contain 'description' and 'class' columns.")

    # Extract descriptions and labels
    descriptions = df["description"].fillna("No description").tolist()
    labels = df["class"].astype(int).tolist()

    full_dataset = ReportDataset(descriptions, labels, tokenizer)

    test_size = int(test_split * len(full_dataset))
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - (test_size + val_size)

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset



def get_tokenizer(tokenizer_type="pretrained", vocab_file=None, corpus_file=None, vocab_size=int, min_frequency=2):
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
            if corpus_file is None:
                raise ValueError("Corpus file must be provided to train the custom tokenizer.")
            print(f"Training custom tokenizer on corpus: {corpus_file}")
            train_tokenizer(corpus_file, vocab_size=vocab_size, min_frequency=min_frequency)
            vocab_file = os.path.join("models/custom_tokenizer", "vocab.txt")

        if not os.path.exists(vocab_file):
            raise ValueError(f"Vocab file not found: {vocab_file}")
        
        return BertWordPieceTokenizer(vocab_file)

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

