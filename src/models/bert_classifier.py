from bert import BERTModel
from transformers import BertModel as HFBertModel
from torch import nn

class BertClassifier(nn.Module):
    def __init__(self, config, num_labels, load_pretrained=False):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BERTModel(config)

        if load_pretrained:
            hf_model = HFBertModel.from_pretrained("bert-base-uncased")
            pretrained_bert_weights = hf_model.state_dict()
            self.bert.load_state_dict(pretrained_bert_weights, strict=False) 
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        _, pooled_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

