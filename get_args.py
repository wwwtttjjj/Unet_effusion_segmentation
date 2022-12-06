import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',
                        type=str,
                        default='./data/train',
                        help='the path of training data')

    parser.add_argument('--val_path',
                        type=str,
                        default='./data/val',
                        help='the path of val data')
    parser.add_argument('--save_path',
                        type=str,
                        default='./checkpoints',
                        help='the path of save_model')
    parser.add_argument('--deterministic',
                        type=int,
                        default=1,
                        help='whether use deterministic training')
    parser.add_argument('--max_iterations',
                        type=int,
                        default=6000,
                        help='maximum epoch number to train')

    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    parser.add_argument('--base_lr',
                        type=float,
                        default=0.01,
                        help='maximum epoch number to train')
    parser.add_argument('--batch_size',
                        type=int,
                        default=2,
                        help='the batch_size of training size')
    parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
    parser.add_argument('--beta', type=float, default=5.0, help='beta')

    parser.add_argument('--consistency',
                        type=float,
                        default=0.1,
                        help='consistency')
    parser.add_argument('--consistency_rampup',
                        type=float,
                        default=40.0,
                        help='consistency_rampup')
    parser.add_argument('--consistency_loss',
                        type=bool,
                        default=True,
                        help='add or not add consistency_loss')
    parser.add_argument('--weak_loss',
                        type=bool,
                        default=True,
                        help='add or not add weak_loss')
    args = parser.parse_args()
    return args