# 

import torch
import torch.nn as nn
from lstm_cell import LSTMCell
from gcn_cell import GCNCell


class LSTMGCNCell(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, gcn_hidden_size):
        super(LSTMGCNCell, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm_cell = LSTMCell(input_size, hidden_size)
        self.gcn_cell = GCNCell(input_size, gcn_hidden_size)

        self.output_layer = nn.Linear(hidden_size + gcn_hidden_size, output_size)

    def forward(self, input_data, hidden_state, cell_state, adj):
        # LSTM
        lstm_hidden_state, lstm_cell_state = self.lstm_cell(input_data, hidden_state, cell_state)

        
        # 在时间维度上使用 LSTM 处理输入数据
=====================for input_t in time_series_data:
            lstm_hidden_state = self.lstm(input_t, lstm_hidden_state)

        # GCN
        gcn_hidden_state = self.gcn_cell(input_data, adj)

        # 拼接
        combined = torch.cat((lstm_hidden_state, gcn_hidden_state), dim=1)

        # 输出层
        output = self.output_layer(combined)

        return output, lstm_hidden_state, lstm_cell_state, gcn_hidden_state