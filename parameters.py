import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=0, help='cuda:No')

    parser.add_argument('--debug', type=bool, default=False, help='iid mnist with mnsit networkk')
    # dataset related
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='mnist, fmnist, cifar10')
    parser.add_argument('--nn_name', type=str, default='simplecifar_nobn', help='mnist, fmnist, simplecifar, resnet18')
    parser.add_argument('--dataset_dist', type=str, default='non_iid2', help='distribution of dataset; iid or non_iid')
    parser.add_argument('--numb_cls_usr', type=int, default=4, help='number of class per user if non_iid2 selected')

    # Federated params
    parser.add_argument('--mode', type=str, default='fedADCp', help='normal,slowmo,benchmark,fed_avg,AFL')
    parser.add_argument('--comm_rounds', type=int, default=1000, help='number of epochs')
    parser.add_argument('--LocalIter', type=int, default=8, help='communication workers')
    parser.add_argument('--Lmomentum', type=float, default=0, help='momentum')
    parser.add_argument('--alg_type', type=str, default='alg3', help='alg1,alg2,al3 (blue,red,green)')
    parser.add_argument('--l_update_ver', type=int, default=1, help='1,2')
    parser.add_argument('--P_M_ver', type=int, default=1, help='1,2')
    parser.add_argument('--alfa', type= float, default=1, help= 'slowmo constant')
    parser.add_argument('--beta', type=float, default=0.9, help='drift control constant')
    parser.add_argument('--beta_warm', type=float, default=0.5, help='beta for the warm up phase')
    parser.add_argument('--phi', type=float, default=0.1, help='fedADC local momentum constant')
    parser.add_argument('--sigma', type=float, default=0.95, help='learning_rate')
    parser.add_argument('--gamma', type=float, default=1, help='learning_rate')
    parser.add_argument('--num_client', type=int, default=50, help='number of clients')
    parser.add_argument('--cl', type=float, default=0.2, help='selected client ratio')
    parser.add_argument('--bs', type=int, default=64, help='batchsize')
    parser.add_argument('--lr', type=float, default=0.1, help='learning_rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--W_decay', type=float, default=1e-4, help='weight decay Value')
    # Quantization params
    parser.add_argument('--quantization', type=bool, default=False, help='apply quantization or not')
    parser.add_argument('--num_groups', type=int, default=16, help='Number Of groups')
    args = parser.parse_args()
    return args

def args_parser_loop():
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', type=bool, default=False, help='iid mnist with mnsit networkk')
    # dataset related
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='mnist, fmnist, cifar10')
    parser.add_argument('--nn_name', type=str, default='simplecifar_nobn', help='mnist, fmnist, simplecifar, resnet18')
    parser.add_argument('--dataset_dist', type=str, default='non_iid2', help='distribution of dataset; iid or non_iid')
    parser.add_argument('--numb_cls_usr', type=list, default=[2,3,4], help='number of class per user if non_iid2 selected')

    # Federated params
    parser.add_argument('--mode', type=str, default='fedADCp', help='normal,slowmo,benchmark,fed_avg,AFL')
    parser.add_argument('--comm_rounds', type=int, default=500, help='number of epochs')
    parser.add_argument('--LocalIter', type=int, default=8, help='communication workers')
    parser.add_argument('--Lmomentum', type=float, default=0, help='momentum')
    parser.add_argument('--alg_type', type=str, default='alg3', help='alg1,alg2,al3 (blue,red,green)')
    parser.add_argument('--l_update_ver', type=int, default=1, help='1,2')
    parser.add_argument('--P_M_ver', type=int, default=2, help='1,2')
    parser.add_argument('--alfa', type= float, default=1, help= 'slowmo constant')
    parser.add_argument('--beta', type=list, default=[0.6,0.7,0.8,0.9], help='drift control constant')
    parser.add_argument('--beta_warm', type=float, default=0.5, help='beta for the warm up phase')
    parser.add_argument('--phi', type=float, default=0.1, help='fedADC local momentum constant')
    parser.add_argument('--sigma', type=float, default=0.95, help='learning_rate')
    parser.add_argument('--gamma', type=float, default=1, help='learning_rate')
    parser.add_argument('--num_client', type=int, default=100, help='number of clients')
    parser.add_argument('--cl', type=float, default=0.2, help='selected client ratio')
    parser.add_argument('--bs', type=int, default=64, help='batchsize')
    parser.add_argument('--lr', type=list, default=[0.01, 0.025, 0.05, 0.1], help='learning_rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--W_decay', type=float, default=1e-4, help='weight decay Value')


    args = parser.parse_args()
    return args
