import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=2,help='batch_size')
parser.add_argument('--adjacency_dir', type=str, default='./data/NBA/data/graph',help='batch_size')
parser.add_argument('--node_attribute_dir', type=str, default='./data/NBA/data/graph',help='batch_size')
parser.add_argument('--exist_table_dir', type=str, default='./data/NBA/data/graph',help='batch_size')
parser.add_argument('--window_size', type=int, default=10,help='batch_size')
parser.add_argument('--lr', type=float, default=0.001)

args, unknown = parser.parse_known_args()


from utils.dataloader import get_dataloader

dataloader = get_dataloader(args=args)

for i, data_timestamp in enumerate(dataloader):
    print (data_timestamp['adjacency_matrix'].shape)
    print (data_timestamp['node_attribute'].shape)
    print (data_timestamp['exist_table'].shape)
    print (data_timestamp['target'].shape)
    if i == 1:
        break