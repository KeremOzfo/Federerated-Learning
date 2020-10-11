import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=0, help='cuda:No')

    # dataset related
    parser.add_argument('--dataset_name', type=str, default='mnist', help='mnist, fmnist, cifar10')
    parser.add_argument('--nn_name', type=str, default='mnist', help='mnist, fmnist, simplecifar, resnet18')
    parser.add_argument('--dataset_dist', type=str, default='iid', help='distribution of dataset; iid or non_iid')
    parser.add_argument('--numb_cls_usr', type=int, default=4, help='number of class per user if non_iid2 selected')

    # Federated params
    parser.add_argument('--alg_type', type=str, default='alg3', help='alg1,alg2,al3 (blue,red,green)')
    parser.add_argument('--beta', type=float, default=0.9, help='drift control constant')
    parser.add_argument('--sigma', type=float, default=0.9, help='drift control constant')
    parser.add_argument('--gamma', type=float, default=0.9, help='drift control constant')
    parser.add_argument('--num_client', type=int, default=50, help='number of clients')
    parser.add_argument('--bs', type=int, default=64, help='batchsize')
    parser.add_argument('--lr', type=float, default=0.1, help='learning_rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--W_decay', type=float, default=1e-4, help='weight decay Value')
    parser.add_argument('--comm_rounds', type=int, default=1000, help='number of epochs')
    parser.add_argument('--worker_blocks', type=int, default=10, help='communication workers')
    parser.add_argument('--LocalIter', type=int, default=5, help='communication workers')
    # Quantization params
    parser.add_argument('--quantization', type=bool, default=False, help='apply quantization or not')
    parser.add_argument('--num_groups', type=int, default=16, help='Number Of groups')
    args = parser.parse_args()
    return args