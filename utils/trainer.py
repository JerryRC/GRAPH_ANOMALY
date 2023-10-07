import torch
import torch.nn as nn


class MyTrainer:
    
    def __init__(self, device, args):
        
        self.args = args
        self.device = device
        
        self.criterion = nn.MSELoss()
        
        
    def _init_model(self, model):
        
        if self.args.model_path:
            model.load_state_dict(torch.load(self.args.model_path))
        # for name, param in my_model.named_parameters():
        #     if 'attn' in name:
        #         param.requires_grad = False
        #         print (name, param.requires_grad)

        #     else:
        #         # param.requires_grad = False
        #         print (name, param.requires_grad)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.coef)
                
        return model, optimizer
        
    
    def train(self, model, train_loader, val_loader):
        
        self.model, self.optimizer = self._init_model(model)
        
        for epoch in range(self.args.epochs):
            print(f"Epoch {epoch + 1}/{self.args.epochs}")
            print("-" * 10)
            
            # 初始化隐藏状态和单元状态（是否需要batch？）
            hidden_state = torch.zeros(self.args.batch_size, self.args.lstm_hidden_size).to(self.device)
            cell_state = torch.zeros(self.args.batch_size, self.args.lstm_hidden_size).to(self.device)
        
            for _, data_timestamp in enumerate(train_loader):
                hidden_state, cell_state = self._train_epoch(data_timestamp, hidden_state, cell_state)

            self._test_epoch(val_loader)
            
            print()
            
    
    def _train_epoch(self, data_timestamp, hidden_state, cell_state):
        
        self.model.train()
        
        # 从数据集中获取邻接矩阵、节点属性和存在表格数据
        adjacency_matrix = data_timestamp['adjacency_matrix'].to(self.device)
        node_attribute = data_timestamp['node_attribute'].to(self.device)
        exist_table = data_timestamp['exist_table'].to(self.device)
        
        # 将邻接矩阵和节点属性输入模型
        output, hidden_state, cell_state = self.model(adjacency_matrix, node_attribute, hidden_state, cell_state)
        
        # TODO：如何计算损失？
        loss = 
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        print(f"Loss: {loss.item():.4f}")
        
        return hidden_state, cell_state