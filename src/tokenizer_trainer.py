from tokenizers import BertWordPieceTokenizer

def train_tokenizer(corpus_file, vocab_size=int, min_frequency=2, vocab_name=str):
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


    tokenizer.save_model("experiments", vocab_name)

    print(f"Tokenizer trained and saved in experiments/{vocab_name}")

