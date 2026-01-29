import torch
from torch import nn
from mamba_ssm import Mamba
from recbole.model.loss import BPRLoss
import torch.nn as nn

class Mamba4Rec(nn.Module):
    def __init__(self, config):
        super(Mamba4Rec, self).__init__()

        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        
        # Hyperparameters for Mamba block
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]

        self.n_items = config["n_items"]
        self.n_users = config["n_users"]

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.user_embedding = nn.Embedding(
            self.n_users, self.hidden_size, padding_idx=0
        )
        self.fuse_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
            
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)
        ])
        
        self.loss_type = config["loss_type"]
        if self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len, user_ids):
        '''
        item_seq: [B, L], padded to the max length L in the batch
        user_ids: [B]
        '''
        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)
        
        # extract user embedding and concatenate to item embeddings then process with an mlp
        user_emb = self.user_embedding(user_ids).unsqueeze(1)  # [B, 1, H]

        # TODO: try other fusion methods
        # item_emb = torch.cat([item_emb, user_emb.repeat(1, item_emb.size(1), 1)], dim=-1)  # [B, L, 2H]
        # item_emb = self.fuse_mlp(item_emb)  # [B, L, H]

        item_emb = item_emb + user_emb  # [B, L, H] 

        for i in range(self.num_layers):
            item_emb = self.mamba_layers[i](item_emb)
        
        # get item embeddings according to the actual lengths
        batch_size = item_seq.size(0)
        seq_output = item_emb[torch.arange(batch_size), item_seq_len - 1]  # [B, H]

        return seq_output
    
    def compute_loss(self, current_query, item_ids, return_logits=False):
        '''
        current_query: [B, H]
        positive_item: [B]
        negative_item: [B, N]
        Assume the postive item is the first one in item_ids
        '''
        items_emb = self.item_embedding(item_ids)  # [B, N+1, H]
        logits = torch.matmul(current_query.unsqueeze(1), items_emb.transpose(1, 2)).squeeze(1)  # [B, N+1]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # [B], positive item is the first one
        loss = self.loss_fct(logits, labels)
        if return_logits:
            return loss, logits
        return loss
    
class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model*4, dropout=dropout)
    
    def forward(self, input_tensor):
        hidden_states = self.mamba(input_tensor)
        if self.num_layers == 1:        # one Mamba layer without residual connection
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:                           # stacked Mamba layers with residual connections
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        hidden_states = self.ffn(hidden_states)
        return hidden_states

class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states