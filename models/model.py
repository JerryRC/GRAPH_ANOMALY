import torch
import torch.nn as nn
from models.gcn_cell import GCNCell


class MyModel(nn.Module):
    
    def __init__(self, args):
        super(MyModel, self).__init__()
        
        self.args = args
           
        self.N1 = nn.Parameter(torch.FloatTensor(args.num_attributes, args.embedding_dim))
        self.N2 = nn.Parameter(torch.FloatTensor(args.num_attributes, args.embedding_dim))
        # 节点嵌入的映射矩阵
        self.Node_emb = nn.Parameter(torch.FloatTensor(args.num_attributes, args.embedding_dim))
        
        self.param_reset()

        # 属性先经过lstm所以输入维度为embedding_dim（属性值与列嵌入相乘）
        self.lstm = nn.LSTM(args.embedding_dim, args.lstm_hidden_size, args.num_lstm_layers, batch_first=True)
        self.gcn_inside = GCNCell(args.lstm_hidden_size, args.inner_gcn_hidden_size, args.dropout)
        self.gcn_outside = GCNCell(args.inner_gcn_hidden_size, args.outer_gcn_hidden_size, args.dropout)

        self.output_layer = nn.Linear(args.outer_gcn_hidden_size, 1)


    def forward(self, adjacency_matrix, input_timestamp):
        
        # 为属性生成一个邻接矩阵
        N1_hat = torch.tanh(self.N1)
        N2_hat = torch.tanh(self.N2)
        A_attr = torch.relu(torch.tanh(torch.mm(N1_hat, N2_hat.T) - torch.mm(N2_hat, N1_hat.T)))
        # # TODO: top-k 稀疏化邻接矩阵,但不影响反向传播
        # A_attr = self.sparse(A_attr, self.topk)
        
        # 在原来的属性值这个维度扩展一个维度，然后数乘映射成嵌入 => (batch, time, node, attr, embed)
        input_embedding = input_timestamp.unsqueeze(-1) * self.Node_emb
        # 置换维度 => (batch, node, attr, time, embed)
        input_embedding = input_embedding.permute(0, 2, 3, 1, 4)
        # 合并前三个维度 => (batch*node*attr, time, embed)
        input_embedding = input_embedding.reshape(-1, input_embedding.shape[-2], input_embedding.shape[-1])
        
        # 返回值有两个：1. 所有时刻的hidden_state输出；2. 最后一个时刻的(hidden_state, cell_state)
        # 在2.中的两种状态都是(num_lstm_layers, batch, lstm_hidden_size)的形式，且lstm层的最后一层才是1.里面那个最终输出
        # 如：lstm_layers=3，那么结果里面 hidden_state 的[2]（其实就取-1也行）才是输出的那层
        lstm_seq_out, _ = self.lstm(input_embedding)
        # 恢复维度 (batch*node*attr, time, lstm_hidden_size) => (batch*node, attr, lstm_hidden_size)
        lstm_seq_out = lstm_seq_out[:,-1,:].reshape(-1, self.args.num_attributes, self.args.lstm_hidden_size)
        
        gcn_hidden_state = self.gcn_inside(A_attr, lstm_seq_out)

        # 要把每个属性单独拿出来，所以要维度转换 (batch*node, attr, inner_gcn_hidden_size) => (attr, batch, node, inner_gcn_hidden_size)
        gcn_hidden_state = gcn_hidden_state.reshape(-1, self.args.num_nodes, self.args.num_attributes, self.args.inner_gcn_hidden_size).permute(2, 0, 1, 3)
        # 换成这个维度以便于GCN的矩阵乘计算能够自动广播 (batch, node, node) * (attr, batch, node, inner_gcn_hidden_size)
        gcn_hidden_state = self.gcn_outside(adjacency_matrix, gcn_hidden_state)
        
        # 恢复维度 (attr, batch, node, outer_gcn_hidden_size) => (batch, node, attr, outer_gcn_hidden_size)
        gcn_hidden_state = gcn_hidden_state.permute(1, 2, 0, 3)
        # 输出层，重构 (batch, node, attr, 1)
        output = self.output_layer(gcn_hidden_state)

        return output.squeeze(-1)
    
    
    def param_reset(self):
        nn.init.xavier_uniform_(self.N1)
        nn.init.xavier_uniform_(self.N2)
        nn.init.xavier_uniform_(self.Node_emb)
        
        
    def sparse(self, A, k):
        # TODO: 如何实现？
        return A
