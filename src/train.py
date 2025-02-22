import torch
from torch import nn
from torch.optim import AdamW
#from transformers import get_scheduler
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path=str, model_type=str):
        """
        Args:
            patience (int): How long to wait after the last time validation loss improved.
            verbose (bool): If True, prints messages when saving the best model.
            delta (float): Minimum change to qualify as an improvement.
            path (str): Path to save the best model weights.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.model_type = model_type
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model weights when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model weights...")
        if self.model_type == "pretrained":
            torch.save(model.classifier.state_dict(), self.path)
        else:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, device, learning_rate=float, num_epochs=int, model_type=str, path=str):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.num_epochs = num_epochs

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        # self.scheduler = get_scheduler(
        #     "linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * num_epochs
        # )
        self.early_stopper = EarlyStopping(patience=5, verbose=True, path=path, model_type=model_type)

    def train_step(self, batch):
        self.model.train()
        
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)
        labels = batch["label"].to(self.device)

        self.optimizer.zero_grad()
        logits = self.model(input_ids, attention_mask, token_type_ids)
        train_loss = self.loss(logits, labels)
        train_loss.backward()
        self.optimizer.step()
        #self.scheduler.step()

        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        train_accuracy = (preds == labels).float().mean().item()

        return train_loss.item(), train_accuracy

    def validation_step(self, batch):
        self.model.eval()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)
        labels = batch["label"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, token_type_ids)
            val_loss = self.loss(logits, labels)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            val_accuracy = (preds == labels).float().mean().item()

        return val_loss.item(), val_accuracy

    def train(self):

        results = {"train_loss": [],
                    "train_acc": [],
                    "val_loss": [],
                    "val_acc": []}    

        for epoch in tqdm(range(self.num_epochs)):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")

            # Training loop
            train_loss, train_accuracy = 0, 0
            for batch in tqdm(self.train_dataloader, desc="Training"):
                loss, accuracy = self.train_step(batch)
                train_loss += loss
                train_accuracy += accuracy
            train_loss /= len(self.train_dataloader)
            train_accuracy /= len(self.train_dataloader)

            # Validation loop
            val_loss, val_accuracy = 0, 0
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                loss, accuracy = self.validation_step(batch)
                val_loss += loss
                val_accuracy += accuracy
            val_loss /= len(self.val_dataloader)
            val_accuracy /= len(self.val_dataloader)

            print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_accuracy)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_accuracy)

            self.early_stopper(val_loss, self.model)
            if self.early_stopper.early_stop:
                print("Early stopping")
                break

        return results
