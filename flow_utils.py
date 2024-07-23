import torch
import torch.nn as nn

class DataRescale(nn.Module):
    def __init__(self, input_dim):
        super(DataRescale, self).__init__()
        self.parameter_tensor = torch.ones(input_dim, 1)
        self.weight = nn.Parameter(self.parameter_tensor)
    
    def forward(self, x):
        return self.weight * x

def create_checkerboard_mask(batch_size, width):
    checkerboard = []
    for h in range(batch_size):
        row = []
        for w in range(width):
            value = ((h % 2) + w) % 2
            row.append(value)
        checkerboard.append(row)
    checkerboard_tensor = torch.tensor(checkerboard, dtype=torch.float32)
    return checkerboard_tensor

def create_half_split_mask(batch_size, dim, reverse_mask):
    mask = torch.zeros(dim)
    mask[:dim // 2] = 1
    if reverse_mask:
        mask = 1 - mask
    mask = mask.repeat(batch_size, 1).float()
    return mask

def apply_mask(x, mask):
    return x * mask

def normalize_column(column):
    min_val = column.min()
    max_val = column.max()
    if min_val < 0 or max_val > 1:
        column = (column - min_val) / (max_val - min_val)
    return column

def normalize_all_columns(x):
    for i in range(x.size(1)):
        x[:, i] = normalize_column(x[:, i])
    return x
