from typing import Dict, List, Optional, Tuple
import copy
import torch
from torch import nn, Tensor
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
import numpy as np
import random
import json
from metakg_guide_transformer import TransformerEncoderLayer, TransformerEncoder
import math
import pandas as pd
import torch.nn as nn

class GELU(nn.Module):
    """
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len=30):
        super(PositionalEncoding, self).__init__()

        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        pad_row = torch.zeros([1, d_model], dtype=torch.float)
        position_encoding = torch.tensor(position_encoding, dtype=torch.float)

        position_encoding = torch.cat((pad_row, position_encoding), dim=0)

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=True)

    def forward(self, batch_len, start, seq_len):
        """
        :param batch_len: scalar
        :param seq_len: scalar
        :return: [batch, time, dim]
        """
        input_pos = torch.tensor([list(range(start + 1, start + seq_len + 1)) for _ in range(batch_len)]).cuda()
        return self.position_encoding(input_pos).transpose(0, 1)

class PositionalEncoding1(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, args, dictionary, true_triples=None):
        super(TransformerModel, self).__init__()
        try:
            from metakg_guide_transformer import TransformerEncoder, TransformerEncoderLayer, Transformer
        except:
            raise ImportError('Transformer module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.ninp = args.embedding_dim
        self.args = args
        self.pos_encoder = PositionalEncoding(self.ninp)
        encoder_layers = TransformerEncoderLayer(d_model=args.embedding_dim, nhead=4, dim_feedforward=args.hidden_size, dropout=args.dropout)
        self.encoder = TransformerEncoder(encoder_layers, args.num_layers)
        self.ntoken = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed = nn.Embedding(self.ntoken, self.ninp)
        self.fc = torch.nn.Linear(self.ninp, self.ninp)
        self.dictionary = dictionary
        self.glue = GELU()
        self.label_smooth = args.label_smooth
        self.output_proj = nn.Linear(2 * self.ninp, self.ninp)
        self.shape_bias = nn.Linear(self.ninp, 1)
        self.device = "gpu"
        self.path = "/root/metakg/data/MetaKG.json"

        self.init_weights()

    def graph_encoding(self, device):
        with open(self.path, 'r') as file:
            meta_dic = json.load(file)
        updated_vectors = torch.tensor([], device=device)

        for center_node, sub_dict in meta_dic.items():
            keys = torch.tensor([], device=device)
            values = torch.tensor([], device=device)
            biases = torch.tensor([], device=device)
            for b, bias_elements in sub_dict.items():
                if b in self.dictionary:
                    key_index = self.dictionary.index(b)
                    key = self.embed(torch.tensor([key_index], dtype=torch.long).to(device))
                    keys = torch.cat((keys, key.unsqueeze(0)), dim=0)
                    values = torch.cat((values, key.unsqueeze(0)), dim=0)

                    bias_vecs = torch.tensor([], device=device)
                    bias_indexs = [self.dictionary.index(elem) for elem in bias_elements if elem in self.dictionary]
                    if not bias_indexs:
                        print ('it is empty and the key is ',b,"and node is",center_node)
                    for i in bias_indexs:

                        bias_vec = self.embed(torch.tensor([i], dtype=torch.long).to(device))
                        bias_vecs = torch.cat((bias_vecs, bias_vec.unsqueeze(0)), dim=0)
                    bias_vecs = bias_vecs.squeeze(1)
                    bias_vecs = torch.mean(bias_vecs, dim=0)
                    biases = torch.cat((biases, bias_vecs.unsqueeze(0)), dim=0)

            if center_node in self.dictionary:
                node_index = self.dictionary.index(center_node)
                query = self.embed(torch.tensor([node_index], dtype=torch.long).to(device))
                biases = biases.squeeze(1)
                keys = keys.squeeze(1)  
                values = values.squeeze(1)

                sp_bias = self.shape_bias(biases).transpose(0,1)
                atten_value = torch.matmul(query, keys.T)
                attn_weights = torch.softmax(
                    atten_value / (self.ninp ** 0.5) + sp_bias, dim=-1)
                new_vector = torch.matmul(attn_weights, values)
                add_vector = torch.cat((query, new_vector), dim=-1)
                updated_vector = self.output_proj(add_vector)

                updated_vectors = torch.cat((updated_vectors, updated_vector.unsqueeze(0)), dim=0)
        updated_vectors=updated_vectors.squeeze(1)
        return updated_vectors

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def init_weights(self):
        xavier_normal_(self.embed.weight.data)

    def logits(self, source, type_id, **unused):
        bsz, src_len = source.shape
        out_len = source.size(1)
        device = source.device
        mask = self._generate_square_subsequent_mask(source.size(-1),device)
        source = source.transpose(0, 1)
        source = self.embed(source)

        type_id = type_id.transpose(0, 1)
        source_type = self.embed(type_id)
        source += source_type
        source += self.pos_encoder(bsz, 0, src_len)
        meta_prompt = self.graph_encoding(device)

        if self.args.encoder:
            output = self.encoder(source, meta_prompt, mask=mask).transpose(0, 1)
        else:
            mask = mask.to(device)
            output = self.encoder(source, tgt_mask=mask).transpose(0, 1)
        logits = torch.mm(self.glue(self.fc(output)).view(-1, self.ninp), self.embed.weight.transpose(0, 1)).view(bsz, out_len, -1)
        return logits

    def forward(self, source, target, type_id, mask, **unused):
        logits = self.logits(source,type_id)
        lprobs = F.log_softmax(logits, dim=-1)
        loss = -(self.label_smooth * torch.gather(input=lprobs, dim=-1, index=target.unsqueeze(-1)).squeeze() \
            + (1 - self.label_smooth) / (self.ntoken - 1) * lprobs.sum(dim=-1)) * mask
        loss = loss.sum() / mask.sum()
        return loss
