
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from dataclasses import dataclass

@dataclass
class Config:

    heads: int
    dropout: float
    embedding_size: int
    dict_size: int
    attention_dropout: float
    hidden_layer_size: int
    N_blocks: int


    @staticmethod
    def from_json_file(filepath: str) -> 'Config':
        with open(filepath, 'r') as file:
          data = json.load(file)
          return Config(**data)

config = Config.from_json_file('config/config.json')

class MultiHeadAttention(nn.Module):
  def __init__(self, config: Config):

    super().__init__()
    self.heads = config.heads
    self.dropout = config.dropout
    self.embedding_size = config.embedding_size
    self.attention_dropout = config.attention_dropout

    self.query = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
    self.key = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
    self.value = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
    self.att_dropout = nn.Dropout(self.attention_dropout)

    self.linear = nn.Linear(self.embedding_size, self.embedding_size)
    self.after_linear_dropout = nn.Dropout(self.dropout)


  def forward(self, x):
    # x shaape is (B, L, E)

    # B - batch_size,
    # L - length,
    # H - heads,
    # E - embedding_size,
    # Q - query_size == (E // H)

    B, L, E = x.shape
    H = self.heads

    query = self.query(x)  #(B, L, E)
    key = self.key(x) #(B, L, E)
    value = self.key(x) #(B, L, E)




    query = query.view(B, L, H, E // H).transpose(2, 1) #(B, H, L, Q)
    key = key.view(B, L, H, E // H).transpose(2, 1) #(B, H, L, Q)
    value = value.view(B, L, H, E // H).transpose(2, 1) #(B, H, L, Q)

    att = (query @ key.transpose(-2, -1)) / (key.shape[-1] ** 0.5) #(B, H, L, Q) @ (B, H, Q, L) -> (B, H, L, L)

    traingle_mask = torch.tril(torch.ones(L, L, device=x.device))
    att = att.masked_fill(traingle_mask[None, None, :, :] == 0, float('-inf')) # (B, H, L, L)

    att = F.softmax(att, dim=-1) # (B, H, L, L)
    att = self.att_dropout(att)
    output = att @ value #(B, H, L, L) @ (B, H, L, Q) -> (B, H, L, Q)
    output = output.transpose(1, 2).contiguous().view(B, L, E) #(B, L, E)
    output = self.linear(output) #(B, L, E) -> (B, L, E)
    output = self.after_linear_dropout(output)

    return output

class FullyConnected(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.hidden_layer_size = config.hidden_layer_size
    self.embedding_size = config.embedding_size
    self.dropout = config.dropout

    self.linear1 = nn.Linear(self.embedding_size, self.hidden_layer_size)
    self.activation = nn.GELU(approximate='tanh')
    self.linear2 = nn.Linear(self.hidden_layer_size, self.embedding_size)
    self.dropout = nn.Dropout(self.dropout)


  def forward(self, x):

    # x shaape is (B, L, E)

    # B - batch_size,
    # L - length,
    # E - embedding_size
    # H - hidden_size

    out = self.linear1(x)  #(B, L, E) -> (B, L, H)
    out = self.activation(out)  #(B, L, H)
    out = self.dropout(out) #(B, L, H)
    out = self.linear2(out) #(B, L, E)

    return out

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.layernorm1 = nn.LayerNorm(config.embedding_size)
        self.mlp = FullyConnected(config)
        self.layernorm2 = nn.LayerNorm(config.embedding_size)


    def forward(self, x):
        out = x + self.att(self.layernorm1(x)) #(B, L, E)
        out = out + self.mlp(self.layernorm2(out)) #(B, L, E)
        return out

class Za_kar_yan_GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.dict_size = config.dict_size
        self.embedding_size = config.embedding_size
        self.N_blocks = config.N_blocks

        self.embedding = nn.Embedding(self.dict_size, self.embedding_size)
        self.position_emb = nn.Embedding(self.dict_size, self.embedding_size)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(self.N_blocks)])
        self.linear = nn.Linear(self.embedding_size, self.dict_size)


    def forward(self, x):
        # x shaape is (B, L)

        # B - batch_size,
        # L - length,
        # E - embedding_size
        # V - dict_size

        B, T = x.shape
        for_position_emb_tensor = torch.arange(T, dtype=torch.long, device=x.device) #(T, )
        out = self.embedding(x) + self.position_emb(for_position_emb_tensor) #(B, T, E)

        for block in self.blocks:
            out = block(out) #(B, T, E)
        

        out = self.linear(out) #(B, T, E) -> (B, T, V)
        return out

