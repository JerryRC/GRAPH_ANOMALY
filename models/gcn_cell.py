import torch
import torch.nn as nn


class GCNCell(nn.Module):
    
    def __init__(self, input_dim, output_dim, dropout):
        super(GCNCell, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        
    def forward(self, A, X):
        return self.relu(self.dropout(self.linear(torch.matmul(A, X))))
