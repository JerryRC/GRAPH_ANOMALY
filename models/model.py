import torch
import torch.nn as nn
from models.lstm_cell import LSTMCell
from models.gcn_cell import GCNCell


class MyModel(nn.Module):
    
    def __init__(self, args):
        super(MyModel, self).__init__()
        
        self.args = args
        
        self.N1 = nn.randn(args.num_attributes, args.embedding_dim)
        self.W1 = nn.Parameter(torch.FloatTensor(args.embedding_dim, args.embedding_dim))
        
        self.N2 = nn.randn(args.num_attributes, args.embedding_dim)
        self.W2 = nn.Parameter(torch.FloatTensor(args.embedding_dim, args.embedding_dim))
        
        self.param_reset()

        # 属性先经过lstm所以输入维度为embedding_dim（属性值与列嵌入相乘）
        self.lstm_cell = LSTMCell(args.embedding_dim, args.lstm_hidden_size)
        self.gcn_inside = GCNCell(args.lstm_hidden_size, args.inner_gcn_hidden_size)
        self.gcn_outside = GCNCell(args.inner_gcn_hidden_size, args.outer_gcn_hidden_size)

        self.output_layer = nn.Linear(args.outer_gcn_hidden_size, 1, dropout=args.dropout)


    def forward(self, adjacency_matrix, input_timestamp, hidden_state, cell_state):
        
        # 为属性生成一个邻接矩阵
        N1_hat = torch.tanh(self.alpha * torch.mm(self.N1, self.W1))
        N2_hat = torch.tanh(self.alpha * torch.mm(self.N2, self.W2))
        A_attr = torch.relu(torch.tanh(self.alpha * (torch.mm(N1_hat, N2_hat.T) - torch.mm(N2_hat, N1_hat.T))))
        # top-k 稀疏化邻接矩阵,但不影响反向传播
        A_attr = self.sparse(A_attr, self.topk)
        
        # 属性值映射到嵌入 TODO：未确定用哪个矩阵作为列嵌入
        input_embedding = input_timestamp.unsqueeze(-1) * N1_hat
        lstm_hidden_state, lstm_cell_state = self.lstm_cell(input_embedding, hidden_state, cell_state)
        # 节点内部属性GCN  TODO：但是多出来了列向量的维度怎么办？((time, node), attr, embed)  A_attr扩展？
        gcn_hidden_state = self.gcn_inside(A_attr, lstm_hidden_state)

        # 要把每个属性单独拿出来，所以要维度转换
        gcn_hidden_state = gcn_hidden_state.permute(0, 2, 1, 3) # ((time, attr), node, embed)
        gcn_hidden_state = self.gcn_outside(adjacency_matrix, gcn_hidden_state)
        
        # 输出层，重构
        output = self.output_layer(gcn_hidden_state)    

        return output, lstm_hidden_state, lstm_cell_state
    
    
    def param_reset(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        
        
    def sparse(self, A, k):
        # TODO: 如何实现？
        return A