import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomAttention(nn.Module):
    def __init__(self, args, dictionary):
        super(CustomAttention, self).__init__()
        self.ninp = args.embedding_dim
        self.dictionary = dictionary
        self.output_proj = nn.Linear(2 * self.ninp, self.ninp)

    def forward(self, dict_a):
        
        updated_vectors = {}

        for a, sub_dict in dict_a.items():
            keys = []
            values = []
            biases = []
            for b, bias_elements in sub_dict.items():
                key = self.encoder(b)
                keys.append(key)
                values.append(key)
                bias_vectors = [self.encoder(elem) for elem in bias_elements]
                biases.append(torch.mean(torch.stack(bias_vectors), dim=0))
            query = self.encoder(a)
            attn_weights = torch.softmax((query @ torch.stack(keys).T) / (self.ninp ** 0.5) + torch.mean(torch.stack(biases), dim=0))
            new_vector = torch.sum(attn_weights.unsqueeze(-1) * torch.stack(values), dim=0)
            updated_vector = self.output_proj(torch.cat((query, new_vector), dim=-1))
            updated_vectors[a] = updated_vector
        return updated_vectors
