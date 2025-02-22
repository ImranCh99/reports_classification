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

def get_bert_config(vocab_size, model_type="pretrained"):
    """Returns the configuration for BERT based on model type."""
    class BertConfig:
        def __init__(self, model_type):
            if model_type == "custom":
                self.hidden_size = 128  
                self.num_hidden_layers = 6 
                self.hidden_dropout_prob = 0.1
                self.attention_probs_dropout_prob = 0.1
                self.layer_norm_eps = 1e-12
                self.intermediate_size = 512 
                self.vocab_size = vocab_size
                self.max_position_embeddings = 512
                self.type_vocab_size = 2
                self.pad_token_id = 0
                self.num_attention_heads = 4  
            else:
                self.hidden_size = 768  
                self.num_hidden_layers = 12  
                self.hidden_dropout_prob = 0.1
                self.attention_probs_dropout_prob = 0.1
                self.layer_norm_eps = 1e-12
                self.intermediate_size = 3072 
                self.vocab_size = vocab_size
                self.max_position_embeddings = 512
                self.type_vocab_size = 2
                self.pad_token_id = 0
                self.num_attention_heads = 12  

    return BertConfig(model_type)
