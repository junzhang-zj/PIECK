import argparse
import torch.cuda as cuda

def parse_args():
    parser = argparse.ArgumentParser(description="Run Recommender Model.")
    parser.add_argument('-d', '--defense', default='NoDefense', choices=['NoDefense', 'NormBound','NormBoundCum', 'Bulyan', 'TrimmedMean', 'Krum', 'MultiKrum', 'Median','Regula'])
    parser.add_argument('--attack', nargs='?', default='PIECKUEA', help="Specify a attack method")
    parser.add_argument('--dim', type=int, default=32, help='Dim of latent vectors.')
    parser.add_argument('--path', nargs='?', default='Data/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?',default='ML-100K', help='Choose a dataset.')
    parser.add_argument('--device', nargs='?', default='cuda:2' if cuda.is_available() else 'cpu',
                        help='Which device to run the model.')

    parser.add_argument('--lr', type=float, default=1, help='Learning rate.') # ml-1m=0.01
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')

    parser.add_argument('--grad_limit', type=float, default=1e-3, help='Limit of l2-norm of item gradients.')
    parser.add_argument('--clients_limit', type=float, default=0.05, help='Limit of proportion of malicious clients.')
    parser.add_argument('--items_limit', type=int, default=60, help='Limit of number of non-zero item gradients.')
    parser.add_argument('--size', type=int, default=150, help='mined popular items number')
    
    parser.add_argument('--part_percent', type=int, default=0, help='Proportion of attacker\'s prior knowledge.')
    parser.add_argument('--attack_lr', type=float, default=0.10, help='Learning rate on FedRecAttack.')
    parser.add_argument('--attack_batch_size', type=int, default=256, help='Batch size on FedRecAttack.')
    parser.add_argument('--attack_bal', type=float, default=0.5, help='loss balance.')

    parser.add_argument('--regula_size', type=int, default=10, help='Regula size.')
    parser.add_argument('--ipe_mu', type=float, default=1e-0, help='Learning rate on FedRecAttack.')
    parser.add_argument('--uea_mu', type=float, default=1e-0, help='Learning rate on FedRecAttack.')
    return parser.parse_args()

args = parse_args()
