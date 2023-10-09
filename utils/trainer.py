import torch
import torch.nn as nn


class MyTrainer:
    
    def __init__(self, device, args):
        
        self.args = args
        self.device = device
        
        self.criterion = nn.MSELoss()
        
        
    def init_model(self, model):
        
        if self.args.model_path:
            model.load_state_dict(torch.load(self.args.model_path))
        # for name, param in my_model.named_parameters():
        #     if 'attn' in name:
        #         param.requires_grad = False
        #         print (name, param.requires_grad)

        #     else:
        #         # param.requires_grad = False
        #         print (name, param.requires_grad)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        model.to(self.device)
        
        return model, optimizer
        
    
    def train(self, model, train_loader, val_loader):
        
        self.model, self.optimizer = self.init_model(model)
        
        for epoch in range(self.args.epochs):
            print(f"Epoch {epoch + 1}/{self.args.epochs}")
            print("-" * 10)
            
            for _, data_timestamp in enumerate(train_loader):
                self.train_epoch(data_timestamp)
            
            print()
            
    
    def train_epoch(self, data_timestamp):
        
        self.model.train()
        
        # 从数据集中获取(batch)邻接矩阵、节点属性和存在表格数据
        adjacency_matrix = data_timestamp['adjacency_matrix'].to(self.device)
        node_attribute = data_timestamp['node_attribute'].to(self.device)
        target = data_timestamp['target'].to(self.device)
        
        # 将邻接矩阵和节点属性输入模型
        output = self.model(adjacency_matrix, node_attribute)
        
        loss = self.criterion(output, target)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        print(f"Loss: {loss.item():.4f}")
