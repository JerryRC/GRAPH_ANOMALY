import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from time import time


class NBADataset(Dataset):
    # TODO: 可变时间步长有用？设置膨胀参数（每隔几个时间步取一次）有用？
    def __init__(self, args):
        # 设置数据目录
        self.adjacency_dir = args.adjacency_dir
        self.node_attribute_dir = args.node_attribute_dir
        self.exist_table_dir = args.exist_table_dir
        self.window_size = args.window_size

        # 读取邻接矩阵、节点属性和存在表格数据，分开先读后转能执行更快（TODO：原因不明）
        self.adjacency_matrices = [np.load(f"{self.adjacency_dir}/adjacency{i}.npy") for i in range(68)]
        self.adjacency_matrices = np.array(self.adjacency_matrices)
        
        self.node_attributes =  [np.load(f"{self.node_attribute_dir}/node_attribute{i}.npy")for i in range(68)]
        self.node_attributes = np.array(self.node_attributes)
        
        self.exist_table = np.load(f"{self.exist_table_dir}/exist_table.npy")
        self.exist_table = torch.FloatTensor(self.exist_table)

        # 计算数据集长度，通常为时间步的数量
        self.length = self.node_attributes.shape[0] - self.window_size + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 切片不取最后一个时间步，因为最后一个时间步的输出是下一个时间步的输入
        # time_start = time()
        adjacency_matrix = torch.FloatTensor(
            self.adjacency_matrices[idx : idx + self.window_size - 1]
        )
        node_attribute = torch.FloatTensor(
            self.node_attributes[idx : idx + self.window_size - 1]
        )
        # 窗口内最后一个时间步的节点属性作为标签
        target = torch.FloatTensor(self.node_attributes[idx + self.window_size - 1])

        # 返回一个包含所有数据的字典
        sample = {
            "adjacency_matrix": adjacency_matrix,
            "node_attribute": node_attribute,
            "exist_table": self.exist_table,
            "target": target,
        }
        # time_end = time()
        # print (time_end - time_start)

        return sample


def collate_fn(batch):
    # 从batch中取出所有数据
    # time_start = time()
    adjacency_matrix = [sample["adjacency_matrix"] for sample in batch]
    adjacency_matrix = torch.stack(adjacency_matrix)
    
    node_attribute = [sample["node_attribute"] for sample in batch]
    node_attribute = torch.stack(node_attribute)
    
    target = [sample["target"] for sample in batch]
    target = torch.stack(target)

    # 返回一个包含所有数据的字典
    sample = {
        "adjacency_matrix": adjacency_matrix,
        "node_attribute": node_attribute,
        "exist_table": sample["exist_table"],
        "target": target,
    }
    # time_end = time()
    # print (time_end - time_start)
    return sample


def get_dataloader(args, num_workers=os.cpu_count()):
    dataset = NBADataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader
