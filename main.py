from train_funcs import *
import numpy as np
from parameters import *
import torch
import random
import datetime
import os
import argparse
import numpy as np
import multiprocessing as mp
import itertools
import torch.multiprocessing as mpcuda
from torch.multiprocessing import set_start_method

def main_treaded(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    simulation_ID = int(random.uniform(1, 999))
    print('device:', device)
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))
    x = datetime.datetime.now()
    date = x.strftime('%b') + '-' + str(x.day)
    if args.mode == 'AFL':
        newFile = '{}-{}-VER:{}-{}-cls_{}-H_{}-A:{}-B:{}-ID_{}'.format(date, args.mode, args.l_update_ver,
                                                                       args.P_M_ver, args.numb_cls_usr, args.LocalIter,
                                                                       args.alfa, args.beta, simulation_ID)
    else:
        newFile = '{}-{}-cls_{}-H_{}-A:{}-B:{}-ID_{}'.format(date, args.mode, args.numb_cls_usr,
                                                             args.LocalIter, args.alfa, args.beta, simulation_ID)
    if not os.path.exists(os.getcwd() + '/Results'):
        os.mkdir(os.getcwd() + '/Results')
    n_path = os.path.join(os.getcwd(), 'Results', newFile)
    for i in range(5):
        accs = None
        if args.mode == 'normal':
            accs = train(args, device)
        elif args.mode == 'slowmo':
            accs = train_slowmo(args, device)
        elif args.mode == 'normal2':
            accs = train_2(args, device)
        elif args.mode == 'nesterov':
            accs = train_nesterov(args, device)
        elif args.mode == 'fed_avg':
            accs = train_fedavg(args, device)
        elif args.mode == 'fedADC':
            accs = train_fedADC(args, device)
        elif args.mode == 'AFL':
            accs = train_AFL(args, device)
        elif args.mode == 'fedADCp':
            accs = train_fedADCp(args, device)
        elif args.mode == 'fedADCn':
            accs = train_fedADCn(args, device)
        if i == 0:
            os.mkdir(n_path)
            f = open(n_path + '/simulation_Details.txt', 'w+')
            f.write('simID = ' + str(simulation_ID) + '\n')
            f.write('############## Args ###############' + '\n')
            for arg in vars(args):
                line = str(arg) + ' : ' + str(getattr(args, arg))
                f.write(line + '\n')
            f.write('############ Results ###############' + '\n')
            f.close()
        s_loc = date + f'federated_prototype_{args.mode}' + '--' + str(i)
        s_loc = os.path.join(n_path, s_loc)
        np.save(s_loc, accs)
        f = open(n_path + '/simulation_Details.txt', 'a+')
        f.write('Trial ' + str(i) + ' results at ' + str(accs[len(accs) - 1]) + '\n')
        f.close()

if __name__ == '__main__':
    args = args_parser_loop()
    total_gpu = torch.cuda.device_count()
    worker_per_gpu = args.worker_per_gpu
    max_active_user = np.min([total_gpu * worker_per_gpu, mp.cpu_count()])
    combinations = []
    work_load=[]
    w_parser = argparse.ArgumentParser()
    for arg in vars(args):
        arg_type = type(getattr(args,arg))
        if arg_type == list and arg != 'lr_change' and arg != 'excluded_gpus':
            work_ = [n for n in getattr(args, arg)]
            work_load.append(work_)
    for t in itertools.product(*work_load):
        combinations.append(t)

    jobs = []
    for combination in combinations:
        w_parser = argparse.ArgumentParser()
        listC = 0
        for arg in vars(args):
            arg_type = type(getattr(args, arg))
            if arg_type == list and arg != 'lr_change' and arg != 'excluded_gpus':
                new_type = type(combination[listC])
                w_parser.add_argument('--{}'.format(arg), type=new_type, default=combination[listC], help='')
                listC +=1
            else:
                val = getattr(args, arg)
                new_type = type(getattr(args, arg))
                w_parser.add_argument('--{}'.format(arg), type=new_type, default=val, help='')

        selected_gpu = int(len(jobs) % total_gpu) ## assign gpu for the work
        while selected_gpu in args.excluded_gpus: ##check gpu
            selected_gpu = int((selected_gpu +1) % total_gpu)
        w_parser.add_argument('--gpu_id', type=int, default=selected_gpu, help='')
        w_args = w_parser.parse_args()
        model = get_net(w_args)

        try:
            set_start_method('spawn')
        except RuntimeError:
            pass

        if torch.cuda.is_available():
            if len(jobs) < max_active_user:
                model.share_memory()
                p = mpcuda.Process(target=main_treaded, args=(w_args,))
                p.start()
                jobs.append(p)
            else:
                for job in jobs:
                    job.join()
                jobs = []
        else:
            if len(jobs) < max_active_user:
                p = mp.Process(target=main_treaded, args=(w_args,))
                jobs.append(p)
                p.start()
            else:
                for job in jobs:
                    job.join()
                jobs = []

