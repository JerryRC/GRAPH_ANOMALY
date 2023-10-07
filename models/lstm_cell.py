import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size

        # 输入门
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate_activation = nn.Sigmoid()

        # 遗忘门
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.forget_gate_activation = nn.Sigmoid()

        # 输出门
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate_activation = nn.Sigmoid()

        # 单元状态
        self.cell_state = nn.Linear(input_size + hidden_size, hidden_size)
        self.tanh = nn.Tanh()


    def forward(self, input_data, hidden_state, cell_state):
        combined = torch.cat((input_data, hidden_state), dim=1)

        # 计算输入门、遗忘门、输出门和新的单元状态
        input_gate = self.input_gate_activation(self.input_gate(combined))
        forget_gate = self.forget_gate_activation(self.forget_gate(combined))
        output_gate = self.output_gate_activation(self.output_gate(combined))
        cell_candidate = self.tanh(self.cell_state(combined))

        # 更新单元状态和隐藏状态
        new_cell_state = forget_gate * cell_state + input_gate * cell_candidate
        new_hidden_state = output_gate * torch.tanh(new_cell_state)

        return new_hidden_state, new_cell_state
