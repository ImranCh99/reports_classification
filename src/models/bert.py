import torch
from torch import nn

class BertConfig:
    def __init__(self):
        self.hidden_size = 384
        self.num_hidden_layers = 12
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.layer_norm_eps = 1e-12
        self.intermediate_size = 1536
        self.vocab_size = 30522
        self.max_position_embeddings = 512
        self.type_vocab_size = 2
        self.pad_token_id = 0
        self.num_attention_heads = 12

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        

    def forward(self, input_ids=None, token_type_ids=None):
        b, t = input_ids.size()
        device = input_ids.device
        # Use the position_ids created in the constructor if none are provided
        position_ids = torch.arange(0, t, dtype=torch.long, device=device)

        # Use the buffered token_type_ids if none are provided
        if token_type_ids is None:
            token_type_ids = torch.zeros(self.position_ids.size(), dtype=torch.long).expand(b, t)

        # If no input embeddings are provided, transform input_ids into embeddings

        inputs_embeds = self.word_embeddings(input_ids)

        # Create token type embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # Sum word embeddings, position embeddings, and token type embeddings
        embeddings = inputs_embeds + token_type_embeddings
        embeddings += self.position_embeddings(position_ids)

        # Normalize and apply dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, config, attention_dropout=0.1, is_decoder=False):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_dropout)
        self.is_decoder = is_decoder

    def transpose_for_scores(self, x):
        # Reshape and permute for multi-head attention
        bsz, seq_len, hidden_size = x.size()
        x = x.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states if encoder_hidden_states is not None else hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states if encoder_hidden_states is not None else hidden_states))

        # Scaled dot-product attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32))

        if attention_mask is not None:
            attention_scores += attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Compute context layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(hidden_states.size(0), -1, self.all_head_size)

        return context_layer

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    
class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask = None):
        
        attention_outputs = self.self(hidden_states=hidden_states, attention_mask=attention_mask)
        attention_output = self.output(input_tensor=hidden_states, hidden_states=attention_outputs)
        
        outputs = attention_output
        
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask = None):
        
        self_attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask
        )
        attention_output = self_attention_outputs
        
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states=hidden_states, attention_mask=attention_mask)

        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BERTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
    
    def get_extended_attention_mask(self, attention_mask, input_shape):
        # Convert attention mask to binary:
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None):
        
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("You have to specify either input_ids")

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, dtype=torch.long, device=input_ids.device)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask)

        sequence_output = encoder_outputs
        pooled_output = self.pooler(sequence_output)
        
        return sequence_output, pooled_output

