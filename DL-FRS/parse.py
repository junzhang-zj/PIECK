import argparse
import torch.cuda as cuda


def parse_args():
    parser = argparse.ArgumentParser(description="Run Recommender Model.")
    parser.add_argument('-d', '--defense', default='NoDefense', choices=['NoDefense', 'NormBound', 'Bulyan', 'TrimmedMean', 'Krum', 'MultiKrum', 'Median']) # 增
    parser.add_argument('--grad_limit', type=float, default=0.5, help='Limit of l2-norm of item gradients.') #增
    parser.add_argument('--attack', nargs='?', default='NoAttack', help="Specify a attack method") # NoAttack,A-pop,A-hum
    parser.add_argument('--dim', type=int, default=8, help='Dim of latent vectors.')
    parser.add_argument('--layers', nargs='?', default='[8,8]', help="Dim of mlp layers.")
    parser.add_argument('--num_neg', type=int, default=1, help='Number of negative items.')
    parser.add_argument('--path', nargs='?', default='Data/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ML-100K', help='Choose a dataset.') # ML-1M, ML-100K, AZ
    parser.add_argument('--device', nargs='?', default='cuda:7' if cuda.is_available() else 'cpu',
                        help='Which device to run the model.')

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--items_limit', type=int, default=30, help='Limit of items.')
    parser.add_argument('--clients_limit', type=float, default=0.0, help='Limit of proportion of malicious clients.')

    return parser.parse_args()


args = parse_args()
