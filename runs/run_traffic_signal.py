import argparse
import torch
import os
from src.tokenizer_trainer import train_tokenizer
from src.utils import set_seed, training_curves
from src.data_preprocess import get_tokenizer, preprocess_data, create_dataloader, create_corpus_file
from src.models import BertClassifier, get_bert_config
from src.train import Trainer
from src.predict import make_predictions



data_path = r"C:\Users\imran\OneDrive\Desktop\codes\data\text\traffic_signal_mwo\Traffic_Signal_Work_Orders.csv"
text_column = "Job Description"
class_column = "Work Type"
label_map = {
    "Scheduled Work": 0,
    "Trouble Call": 1
}
classes = len(label_map)
model_name = "traffic_signal_classifier"
tokenizer_name = "traffic_singal"
corpus_name = "traffic_signal_corpus"


def main():
    parser = argparse.ArgumentParser(description="Run BERT-based Text Classification")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="Mode: 'train' or 'test'")
    parser.add_argument("--model", type=str, choices=["pretrained", "custom"], default="pretrained", help="Model type")
    parser.add_argument("--model_path", type=str, help="Path to saved model")
    parser.add_argument("--vocab_path", type=str, help="Path to saved vocab file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size for custom tokenizer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")

    args = parser.parse_args()

    set_seed(42)

    if args.mode == "train":

        if not os.path.exists(f"experiments/{corpus_name}.txt"):
            create_corpus_file(file_path=data_path, text_column=text_column, corpus_file_name=corpus_name)
        else:
            print(f"Corpus file already exists at experiments/{corpus_name}.txt. Skipping creation.")

        if not os.path.exists(f"experiments/{tokenizer_name}-vocab.txt"):
            train_tokenizer(f"experiments/{corpus_name}.txt", vocab_size=args.vocab_size, vocab_name=tokenizer_name)
        else:
            print(f"Vocabulary file already exists at experiments/{tokenizer_name}.txt. Skipping training.")

        tokenizer = get_tokenizer(tokenizer_type="custom" if args.model == "custom" else "pretrained", vocab_file=f"experiments/{tokenizer_name}-vocab.txt" if args.model == "custom" else None)

        train_dataset, val_dataset, test_dataset = preprocess_data(file_path=data_path, text_column=text_column, class_column=class_column, tokenizer=tokenizer, label_map=label_map)
        train_loader, val_loader, _ = create_dataloader(train_dataset, val_dataset, test_dataset, batch_size=args.batch_size)

        config = get_bert_config(args.vocab_size if args.model == "custom" else 30522, model_type=args.model)

        classifier = BertClassifier(config, num_labels=classes, load_pretrained=(args.model == "pretrained"))

        if args.model == "pretrained":
            for param in classifier.bert.parameters():
                param.requires_grad = False

        path = f"experiments/{model_name}.pth" if args.model == "custom" else f"experiments/{model_name}_pretrained.pth"
        
        bert_trainer = Trainer(classifier, train_loader, val_loader, device=args.device, learning_rate=args.lr, num_epochs=args.epochs, model_type=args.model, path=path)
        results = bert_trainer.train()
        training_curves(results)

    elif args.mode == "test":

        tokenizer = get_tokenizer(tokenizer_type="custom" if args.model == "custom" else "pretrained", vocab_file=f"experiments/{tokenizer_name}-vocab.txt" if args.model == "custom" else None)
        train_dataset, val_dataset, test_dataset = preprocess_data(file_path=data_path, text_column=text_column, class_column=class_column, tokenizer=tokenizer, label_map=label_map)
        _, _, test_loader = create_dataloader(train_dataset, val_dataset, test_dataset, batch_size=args.batch_size)

        config = get_bert_config(args.vocab_size if args.model == "custom" else 30522, model_type=args.model)

        classifier = BertClassifier(config, num_labels=classes, load_pretrained=(args.model == "pretrained"))
        if args.model == "pretrained":
            classifier.classifier.load_state_dict(torch.load(f"{args.model_path}.pth"))
        else:
            classifier.load_state_dict(torch.load(f"{args.model_path}.pth"))
        
        make_predictions(classifier, test_loader, device=args.device)

        

if __name__ == "__main__":
    main()
