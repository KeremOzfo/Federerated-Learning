from train_funcs import *
import numpy as np
from parameters import *
import torch
import random
import datetime
import os

device = torch.device("cpu")
args = args_parser()

if __name__ == '__main__':
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    simulation_ID = int(random.uniform(1,999))
    print('device:',device)
    args = args_parser()
    for arg in vars(args):
       print(arg, ':', getattr(args, arg))
    x = datetime.datetime.now()
    date = x.strftime('%b') + '-' + str(x.day)
    if args.mode == 'AFL':
        newFile = '{}-{}-VER:{}-{}-cls_{}-H_{}-A:{}-B:{}-ID_{}'.format(date, args.mode, args.l_update_ver,
        args.P_M_ver, args.numb_cls_usr, args.LocalIter,args.alfa,args.beta,simulation_ID)
    else:
        newFile = '{}-{}-cls_{}-H_{}-A:{}-B:{}-ID_{}'.format(date,args.mode,args.numb_cls_usr,
                                                             args.LocalIter,args.alfa,args.beta,simulation_ID)
    if not os.path.exists(os.getcwd() + '/Results'):
        os.mkdir(os.getcwd() + '/Results')
    n_path = os.path.join(os.getcwd(), 'Results', newFile)
    for i in range(5):
        accs = None
        if args.mode == 'normal':
            accs = train(args, device)
        elif args.mode == 'slowmo':
            accs = train_slowmo(args,device)
        elif args.mode == 'normal2':
            accs = train_2(args, device)
        elif args.mode == 'nesterov':
            accs = train_nesterov(args,device)
        elif args.mode == 'fed_avg':
            accs = train_fedavg(args,device)
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
        s_loc = os.path.join(n_path,s_loc)
        np.save(s_loc,accs)
        f = open(n_path + '/simulation_Details.txt', 'a+')
        f.write('Trial ' + str(i) + ' results at ' + str(accs[len(accs)-1]) + '\n')
        f.close()