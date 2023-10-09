import argparse
from utils.dataloader import get_dataloader
import torch
from utils.trainer import MyTrainer
from models.model import MyModel
import os


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()

parser.add_argument('--num_workers', type=int, default=os.cpu_count())

parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--adjacency_dir', type=str, default='./data/NBA/data/graph')
parser.add_argument('--node_attribute_dir', type=str, default='./data/NBA/data/graph')
parser.add_argument('--exist_table_dir', type=str, default='./data/NBA/data/graph')

parser.add_argument('--embedding_dim', type=int, default=20)
parser.add_argument('--lstm_hidden_size', type=int, default=20)
parser.add_argument('--num_lstm_layers', type=int, default=1)
parser.add_argument('--window_size', type=int, default=10)

parser.add_argument('--inner_gcn_hidden_size', type=int, default=20)
parser.add_argument('--outer_gcn_hidden_size', type=int, default=20)

parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=0.0001)

args, unknown = parser.parse_known_args()
print("注意，以下的传入参数未经定义：\n", unknown) if unknown else print("所有传入参数解析完成！")


if __name__ == '__main__':
    
    dataloader = get_dataloader(args=args)
    model = MyModel(args=args)

    trainer = MyTrainer(device=DEVICE ,args=args)
    trainer.train(model, dataloader, dataloader)
