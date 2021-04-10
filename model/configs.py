import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--max_len',default=256)
parser.add_argument('--batch_size',default=32)
parser.add_argument('--epochs',default=10)
parser.add_argument('--learning_rate',default=1e-3)
parser.add_argument('--d_model',default=768)
parser.add_argument('--head_num',default=12)
parser.add_argument('--dropout',default=0.3)
parser.add_argument('--layer_norm_epsilon',default=1e-3) # 수정 해야됨
parser.add_argument('--no_mask_ratio',default=0.4)
parser.add_argument('--mask_ratio',default=0.3)
parser.add_argument('--random_mask_ratio',default=0.1)
parser.add_argument('--multiple_mask_ratio',default=0.2)


args=parser.parse_args([])