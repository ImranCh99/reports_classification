import os
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer

def train_tokenizer(corpus_file, vocab_size=int, min_frequency=2, save_dir="models/custom_tokenizer"):
    """
    Train a custom BertWordPieceTokenizer on a given corpus and save the model.
    
    Args:
        corpus_file (str): Path to the corpus file for training the tokenizer.
        vocab_size (int): Desired vocabulary size (default: 30000).
        min_frequency (int): Minimum frequency of words to be included in the vocabulary (default: 2).
        save_dir (str): Directory where the trained tokenizer will be saved (default: "models/custom_tokenizer").
    """
    tokenizer = BertWordPieceTokenizer()

    tokenizer.train(files=[corpus_file], vocab_size=vocab_size, min_frequency=min_frequency,
                    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])


    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_model(save_dir)

    print(f"Tokenizer trained and saved in '{save_dir}'")

