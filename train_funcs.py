import torch
from torch.utils.data import DataLoader
# custom modules
import data_loader as dl
from nn_classes import *
import server_functions as sf
import math
from parameters import *
import time
import numpy as np
from tqdm import tqdm


def evaluate_accuracy(model, testloader, device):
    """Calculates the accuracy of the model"""
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def train(args, device):

    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds = dl.get_indices(trainset, args)
    # PS model
    net_ps = get_net(args).to(device)
    net_ps_prev = get_net(args).to(device)
    sf.initialize_zero(net_ps_prev)
    prev_models = [get_net(args).to(device) for u in range(num_client)]
    [sf.initialize_zero(prev_models[u]) for u in range(num_client)]



    net_users = [get_net(args).to(device) for u in range(num_client)]

    optimizers = [torch.optim.SGD(net_users[cl].parameters(), lr=args.lr, weight_decay = 1e-4) for cl in range(num_client)]
    criterions = [nn.CrossEntropyLoss() for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)

    # synch all clients models models with PS
    [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]

    net_sizes, net_nelements = sf.get_model_sizes(net_ps)
    ind_pairs = sf.get_indices(net_sizes, net_nelements)
    N_s = (50000 if args.dataset_name == 'cifar10' else 60000)
    modelsize = sf.count_parameters(net_ps)
    accuracys = []
    ps_model_mask = torch.ones(modelsize).to(device)

    acc = evaluate_accuracy(net_ps, testloader, device)
    accuracys.append(acc * 100)
    localIter = 0
    worker_block_comm = 0
    group_selection = args.num_client / args.worker_blocks
    runs = args.comm_rounds * args.LocalIter


    for run in range(runs):

        for cl in range(num_client):

            trainloader = DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                                     shuffle=True)
            for data in trainloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                sf.zero_grad_ps(net_users[cl])
                predicts = net_users[cl](inputs)
                loss = criterions[cl](predicts, labels)
                loss.backward()
                break
        localIter += 1
        if run <5:
            sf.zero_grad_ps(net_ps)
            [sf.push_grad(net_users[cl], net_ps, num_client) for cl in range(num_client)]
            sf.update_model(net_ps, net_ps_prev, lr=0.1, momentum=0.9, weight_decay=1e-4)
            [sf.pull_model(prev_models[cl], net_ps_prev) for cl in range(num_client)]
            [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]
            localIter = 0
        else:
            [sf.update_model_2(net_users[cl], prev_models[cl], device, args) for cl in
             range(num_client)]
        if localIter == args.LocalIter: ## Communication
            localIter = 0
            worker_vec = np.zeros(args.num_client)
            selected_group_no = worker_block_comm % group_selection
            worker_block_comm += 1
            stpoint = int(selected_group_no*args.worker_blocks)
            worker_vec[stpoint:stpoint+args.worker_blocks] = 1
            ps_model_flat =sf.get_model_flattened(net_ps, device)
            selected_avg = torch.zeros_like(ps_model_flat).to(device)
            for cl in (np.where(worker_vec == 1)[0]): ## update model of the selected group of workers
                model_flat = sf.get_model_flattened(net_users[cl], device)
                dif_model = model_flat.sub(1, ps_model_flat)
                selected_avg.add_(1/args.worker_blocks, dif_model)
            ps_model_flat.add_(1, selected_avg)
            sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)
            [sf.pull_model(net_users[cl], net_ps) for cl in (np.where(worker_vec == 1)[0])]
            for cl in (np.where(worker_vec == 0)[0]): ## update model of the other workers
                model_flat = sf.get_model_flattened(net_users[cl], device)
                dif_model = model_flat.sub(1, ps_model_flat)
                modified_dif = selected_avg.mul(10)
                modified_dif.add_(1,dif_model)
                model_flat.add_(1/(args.worker_blocks +1), modified_dif)
                sf.make_model_unflattened(net_users[cl], model_flat, net_sizes, ind_pairs)
            new_momentum = selected_avg.mul(args.gamma/(args.LocalIter* args.lr)) ## update momentum
            new_momentum.add_(1-args.gamma, sf.get_model_flattened(net_ps_prev,device))  ## update momentum
            sf.make_model_unflattened(net_ps_prev, new_momentum, net_sizes, ind_pairs) ## broadcast momentum to workers
            [sf.pull_model(prev_models[cl], net_ps_prev) for cl in range(args.num_client)] ## broadcast model to workers

        # if run %10 == 5: ##debug
        #     acc = evaluate_accuracy(net_ps, testloader, device)
        #     print('accuracy:', acc * 100,run)


        if run % args.LocalIter == 0:
            acc = evaluate_accuracy(net_ps, testloader, device)
            accuracys.append(acc * 100)
            print('accuracy:',acc*100)
    return accuracys

def train_fedavg(args, device):

    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds = dl.get_indices(trainset, args)
    # PS model
    net_ps = get_net(args).to(device)
    net_ps_prev = get_net(args).to(device)
    sf.initialize_zero(net_ps_prev)
    prev_models = [get_net(args).to(device) for u in range(num_client)]
    [sf.initialize_zero(prev_models[u]) for u in range(num_client)]



    net_users = [get_net(args).to(device) for u in range(num_client)]

    criterions = [nn.CrossEntropyLoss() for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)

    # synch all clients models models with PS
    [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]

    net_sizes, net_nelements = sf.get_model_sizes(net_ps)
    ind_pairs = sf.get_indices(net_sizes, net_nelements)
    N_s = (50000 if args.dataset_name == 'cifar10' else 60000)
    accuracys = []

    acc = evaluate_accuracy(net_ps, testloader, device)
    accuracys.append(acc * 100)

    for run in range(args.comm_rounds):
        worker_vec = np.zeros(num_client)
        worker_vec[np.random.choice(range(num_client), int(args.cl * num_client), replace=False)] = 1  ### randomly select workers
        selected_clients = np.where(worker_vec == 1)[0]
        for cl in selected_clients:
            localIter = 0

            while (localIter < args.LocalIter):
                trainloader = DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                                     shuffle=True)
                for data in trainloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    sf.zero_grad_ps(net_users[cl])
                    predicts = net_users[cl](inputs)
                    loss = criterions[cl](predicts, labels)
                    loss.backward()
                    sf.update_model(net_users[cl], prev_models[cl], lr=args.lr, momentum=args.Lmomentum, weight_decay=1e-4)
                    localIter +=1
                    if localIter == args.LocalIter:
                        break
        ps_model_flat =sf.get_model_flattened(net_ps, device)
        selected_avg = torch.zeros_like(ps_model_flat).to(device)
        for cl in selected_clients:  ## update model of the selected group of workers
            model_flat = sf.get_model_flattened(net_users[cl], device)
            selected_avg.add_(1 / len(selected_clients), model_flat)
        ps_model_flat = selected_avg
        sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)
        [sf.pull_model(net_users[cl], net_ps) for cl in range(args.num_client)]

        # if run %10 == 5: ##debug
        #     acc = evaluate_accuracy(net_ps, testloader, device)
        #     print('accuracy:', acc * 100,run)


        if run %5==0:
            acc = evaluate_accuracy(net_ps, testloader, device)
            accuracys.append(acc * 100)
            print('accuracy:',acc*100)
    return accuracys

def train_slowmo(args, device):

    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds = dl.get_indices(trainset, args)
    # PS model
    net_ps = get_net(args).to(device)
    net_ps_prev = get_net(args).to(device)
    sf.initialize_zero(net_ps_prev)
    prev_models = [get_net(args).to(device) for u in range(num_client)]
    [sf.initialize_zero(prev_models[u]) for u in range(num_client)]



    net_users = [get_net(args).to(device) for u in range(num_client)]

    criterions = [nn.CrossEntropyLoss() for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)

    # synch all clients models models with PS
    [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]

    net_sizes, net_nelements = sf.get_model_sizes(net_ps)
    ind_pairs = sf.get_indices(net_sizes, net_nelements)
    N_s = (50000 if args.dataset_name == 'cifar10' else 60000)
    accuracys = []

    acc = evaluate_accuracy(net_ps, testloader, device)
    accuracys.append(acc * 100)
    for run in range(args.comm_rounds):
        worker_vec = np.zeros(num_client)
        worker_vec[np.random.choice(range(num_client),int(args.cl * num_client),replace=False)] = 1 ### randomly select workers
        selected_clients = np.where(worker_vec == 1)[0]

        for cl in selected_clients:
            localIter = 0

            while (localIter < args.LocalIter):
                trainloader = DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                                         shuffle=True)
                for data in trainloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    sf.zero_grad_ps(net_users[cl])
                    predicts = net_users[cl](inputs)
                    loss = criterions[cl](predicts, labels)
                    loss.backward()
                    sf.update_model(net_users[cl], prev_models[cl], lr=args.lr, momentum=args.Lmomentum,
                                    weight_decay=1e-4)
                    localIter += 1
                    if localIter == args.LocalIter:
                        break
        ps_model_flat =sf.get_model_flattened(net_ps, device)
        selected_avg = torch.zeros_like(ps_model_flat).to(device)
        for cl in selected_clients:  ## update model of the selected group of workers
            model_flat = sf.get_model_flattened(net_users[cl], device)
            dif_model = model_flat.sub(1, ps_model_flat)
            selected_avg.add_(1 / len(selected_clients), dif_model)
        selected_avg.mul_(1/args.lr) ## pseudo grad
        old_momentum = sf.get_model_flattened(net_ps_prev, device)
        new_momentum = old_momentum.mul(args.beta)
        new_momentum.add_(1,selected_avg)
        ps_model_flat.add_(args.alfa*args.lr,new_momentum)
        sf.make_model_unflattened(net_ps_prev, new_momentum, net_sizes, ind_pairs)
        sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)
        [sf.pull_model(net_users[cl], net_ps) for cl in range(args.num_client)]



        if run % 5 == 0:
            acc = evaluate_accuracy(net_ps, testloader, device)
            accuracys.append(acc * 100)
            print('accuracy:',acc*100)
    return accuracys

def train_2(args, device):

    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds = dl.get_indices(trainset, args)
    # PS model
    net_ps = get_net(args).to(device)
    net_ps_prev = get_net(args).to(device)
    sf.initialize_zero(net_ps_prev)
    prev_models = [get_net(args).to(device) for u in range(num_client)]
    [sf.initialize_zero(prev_models[u]) for u in range(num_client)]



    net_users = [get_net(args).to(device) for u in range(num_client)]

    criterions = [nn.CrossEntropyLoss() for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)

    # synch all clients models models with PS
    [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]

    net_sizes, net_nelements = sf.get_model_sizes(net_ps)
    ind_pairs = sf.get_indices(net_sizes, net_nelements)
    N_s = (50000 if args.dataset_name == 'cifar10' else 60000)
    modelsize = sf.count_parameters(net_ps)
    accuracys = []

    acc = evaluate_accuracy(net_ps, testloader, device)
    accuracys.append(acc * 100)
    worker_block_comm = 0
    group_selection = args.num_client / args.worker_blocks

    for run in range(args.comm_rounds):
        if  run <5:
            worker_vec = np.ones(num_client)
        else:
            worker_vec = np.zeros(num_client)
            selected_group_no = worker_block_comm % group_selection
            worker_block_comm += 1
            stpoint = int(selected_group_no * args.worker_blocks)
            worker_vec[stpoint:stpoint + args.worker_blocks] = 1
        for cl in (np.where(worker_vec == 1)[0]):
            localIter = 0
            trainloader = DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                                     shuffle=True)
            for data in trainloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                sf.zero_grad_ps(net_users[cl])
                predicts = net_users[cl](inputs)
                loss = criterions[cl](predicts, labels)
                loss.backward()
                sf.update_model_2(net_users[cl], prev_models[cl], device, args)
                localIter += 1
                break
        if run <5:
            sf.zero_grad_ps(net_ps)
            [sf.push_grad(net_users[cl], net_ps, num_client) for cl in np.where(worker_vec == 1)[0]]
            sf.update_model(net_ps, net_ps_prev, lr=0.1, momentum=0.9, weight_decay=1e-4)
            [sf.pull_model(prev_models[cl], net_ps_prev) for cl in range(num_client)]
            [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]
        else:
            ps_model_flat =sf.get_model_flattened(net_ps, device)
            selected_avg = torch.zeros_like(ps_model_flat).to(device)
            for cl in (np.where(worker_vec == 1)[0]): ## update model of the selected group of workers
                model_flat = sf.get_model_flattened(net_users[cl], device)
                dif_model = model_flat.sub(1, ps_model_flat)
                selected_avg.add_(1/args.worker_blocks, dif_model)
            ps_model_flat.add_(1, selected_avg)
            sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)
            [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]
            new_momentum = selected_avg.mul(args.gamma/(args.LocalIter* args.lr)) ## update momentum
            new_momentum.add_(1-args.gamma, sf.get_model_flattened(net_ps_prev,device))  ## update momentum
            sf.make_model_unflattened(net_ps_prev, new_momentum, net_sizes, ind_pairs) ## broadcast momentum to workers
            [sf.pull_model(prev_models[cl], net_ps_prev) for cl in range(args.num_client)] ## broadcast model to workers

        # if run %10 == 5: ##debug
        #     acc = evaluate_accuracy(net_ps, testloader, device)
        #     print('accuracy:', acc * 100,run)


        if run % 5 == 0:
            acc = evaluate_accuracy(net_ps, testloader, device)
            accuracys.append(acc * 100)
            print('accuracy:',acc*100)
    return accuracys

def train_nesterov(args, device):

    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds = dl.get_indices(trainset, args)
    # PS model
    net_ps = get_net(args).to(device)
    net_ps_prev = get_net(args).to(device)
    sf.initialize_zero(net_ps_prev)
    prev_models = [get_net(args).to(device) for u in range(num_client)]
    [sf.initialize_zero(prev_models[u]) for u in range(num_client)]



    net_users = [get_net(args).to(device) for u in range(num_client)]

    criterions = [nn.CrossEntropyLoss() for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)

    # synch all clients models models with PS
    [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]

    net_sizes, net_nelements = sf.get_model_sizes(net_ps)
    ind_pairs = sf.get_indices(net_sizes, net_nelements)
    N_s = (50000 if args.dataset_name == 'cifar10' else 60000)
    modelsize = sf.count_parameters(net_ps)
    accuracys = []

    acc = evaluate_accuracy(net_ps, testloader, device)
    accuracys.append(acc * 100)

    for run in range(args.comm_rounds):
        worker_vec = np.zeros(num_client)
        worker_vec[np.random.choice(range(num_client), int(args.cl * num_client), replace=False)] = 1  ### randomly select workers
        selected_clients = np.where(worker_vec == 1)[0]

        for cl in selected_clients:
            localIter =0

            while (localIter < args.LocalIter):
                trainloader = DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                                         shuffle=True)
                for data in trainloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    sf.zero_grad_ps(net_users[cl])
                    predicts = net_users[cl](inputs)
                    loss = criterions[cl](predicts, labels)
                    loss.backward()
                    sf.update_model(net_users[cl], prev_models[cl], lr=args.lr, momentum=args.Lmomentum,
                                    weight_decay=1e-4)
                    localIter += 1
                    if localIter == args.LocalIter:
                        break
        ps_model_flat =sf.get_model_flattened(net_ps, device)
        selected_avg = torch.zeros_like(ps_model_flat).to(device)
        for cl in selected_clients: ## update model of the selected group of workers
            model_flat = sf.get_model_flattened(net_users[cl], device)
            dif_model = model_flat.sub(1, ps_model_flat)
            selected_avg.add_(1/args.worker_blocks, dif_model)
        pseudo_grad =selected_avg.mul(1/args.lr)
        momentum = sf.get_model_flattened(net_ps_prev,device)
        new_momentum = pseudo_grad.sub(1-args.beta, momentum)
        ps_model_flat.add_(args.alfa *args.lr,new_momentum)
        sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)
        [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]
        sf.make_model_unflattened(net_ps_prev, new_momentum, net_sizes, ind_pairs) ## broadcast momentum to workers

        # if run %10 == 5: ##debug
        #     acc = evaluate_accuracy(net_ps, testloader, device)
        #     print('accuracy:', acc * 100,run)


        if run % 5==0:
            acc = evaluate_accuracy(net_ps, testloader, device)
            accuracys.append(acc * 100)
            print('accuracy:',acc*100)
    return accuracys

def train_AFL(args, device):

    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds = dl.get_indices(trainset, args)
    # PS model
    net_ps = get_net(args).to(device)
    net_ps_prev = get_net(args).to(device)
    sf.initialize_zero(net_ps_prev)
    prev_models = [get_net(args).to(device) for u in range(num_client)]
    [sf.initialize_zero(prev_models[u]) for u in range(num_client)]



    net_users = [get_net(args).to(device) for u in range(num_client)]

    criterions = [nn.CrossEntropyLoss() for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)

    # synch all clients models models with PS
    [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]

    net_sizes, net_nelements = sf.get_model_sizes(net_ps)
    ind_pairs = sf.get_indices(net_sizes, net_nelements)
    N_s = (50000 if args.dataset_name == 'cifar10' else 60000)
    modelsize = sf.count_parameters(net_ps)
    accuracys = []

    acc = evaluate_accuracy(net_ps, testloader, device)
    accuracys.append(acc * 100)

    for run in range(args.comm_rounds):
        worker_vec = np.zeros(num_client)
        worker_vec[np.random.choice(range(num_client), int(args.cl * num_client), replace=False)] = 1  ### randomly select workers
        selected_clients = np.where(worker_vec == 1)[0]

        global_M = sf.get_model_flattened(net_ps_prev, device)
        local_M = global_M.mul(args.beta / args.LocalIter) if args.P_M_ver== 1 \
            else global_M.mul(1 / args.LocalIter)

        for cl in selected_clients:
            localIter =0

            while (localIter < args.LocalIter):
                trainloader = DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                                         shuffle=True)
                for data in trainloader:
                    if args.l_update_ver ==1: ## RED
                       w_model = sf.get_model_flattened(net_users[cl],device)
                       w_model.add_(args.lr, local_M)
                       sf.make_model_unflattened(net_users[cl], w_model, net_sizes, ind_pairs)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    sf.zero_grad_ps(net_users[cl])
                    predicts = net_users[cl](inputs)
                    loss = criterions[cl](predicts, labels)
                    loss.backward()
                    sf.update_model(net_users[cl], prev_models[cl], lr=args.lr, momentum=args.Lmomentum,
                                    weight_decay=1e-4)
                    if args.l_update_ver ==2:
                       w_model = sf.get_model_flattened(net_users[cl],device)
                       w_model.add_(args.lr, local_M)
                       sf.make_model_unflattened(net_users[cl], w_model, net_sizes, ind_pairs)
                    localIter += 1
                    if localIter == args.LocalIter:
                        break
        ps_model_flat =sf.get_model_flattened(net_ps, device)
        selected_avg = torch.zeros_like(ps_model_flat).to(device)
        for cl in selected_clients: ## update model of the selected group of workers
            model_flat = sf.get_model_flattened(net_users[cl], device)
            dif_model = model_flat.sub(1, ps_model_flat)
            selected_avg.add_(1/len(selected_clients), dif_model)
        pseudo_grad =selected_avg.mul(1/args.lr)
        momentum = sf.get_model_flattened(net_ps_prev,device)
        if args.P_M_ver == 1:
            new_momentum = pseudo_grad
        elif args.P_M_ver == 2:
            new_momentum = pseudo_grad.sub(1-args.beta, momentum)
        ps_model_flat.add_(args.alfa *args.lr,new_momentum)
        sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)
        [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]
        sf.make_model_unflattened(net_ps_prev, new_momentum, net_sizes, ind_pairs) ## broadcast momentum to workers

        # if run %10 == 5: ##debug
        #     acc = evaluate_accuracy(net_ps, testloader, device)
        #     print('accuracy:', acc * 100,run)


        if run % 5==0:
            acc = evaluate_accuracy(net_ps, testloader, device)
            accuracys.append(acc * 100)
            print('accuracy:',acc*100)
    return accuracys

def train_fedADC(args, device):

    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds = dl.get_indices(trainset, args)
    # PS model
    net_ps = get_net(args).to(device)
    net_ps_prev = get_net(args).to(device)
    sf.initialize_zero(net_ps_prev)
    prev_models = [get_net(args).to(device) for u in range(num_client)]
    [sf.initialize_zero(prev_models[u]) for u in range(num_client)]



    net_users = [get_net(args).to(device) for u in range(num_client)]

    criterions = [nn.CrossEntropyLoss() for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)

    # synch all clients models models with PS
    [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]

    net_sizes, net_nelements = sf.get_model_sizes(net_ps)
    ind_pairs = sf.get_indices(net_sizes, net_nelements)
    N_s = (50000 if args.dataset_name == 'cifar10' else 60000)
    accuracys = []

    acc = evaluate_accuracy(net_ps, testloader, device)
    accuracys.append(acc * 100)
    for run in range(args.comm_rounds):
        worker_vec = np.zeros(num_client)
        worker_vec[np.random.choice(range(num_client),int(args.cl * num_client),replace=False)] = 1 ### randomly select workers
        selected_clients = np.where(worker_vec == 1)[0]

        m_global = (sf.get_model_flattened(net_ps_prev, device)).mul(1 / args.LocalIter)
        for cl in selected_clients:
            localIter = 0
            m_local = None
            while (localIter < args.LocalIter):
                trainloader = DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                                         shuffle=True)
                for data in trainloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    sf.zero_grad_ps(net_users[cl])
                    predicts = net_users[cl](inputs)
                    loss = criterions[cl](predicts, labels)
                    loss.backward()
                    w_grad = sf.get_grad_flattened(net_users[cl],device)
                    w_model = sf.get_model_flattened(net_users[cl],device)
                    if localIter ==0:
                        m_local = w_grad
                    else:
                        m_local.mul_(args.phi)
                        m_local.add_(1, w_grad)
                        m_local.mul_(1/(1+args.phi))
                    w_model.add_(args.lr, m_global.sub(m_local))
                    sf.make_model_unflattened(net_users[cl], w_model, net_sizes, ind_pairs)
                    localIter += 1
                    if localIter == args.LocalIter:
                        break
        ##start comm
        ps_model_flat =sf.get_model_flattened(net_ps, device)
        selected_avg = torch.zeros_like(ps_model_flat).to(device)
        for cl in selected_clients:  ## update model of the selected group of workers
            model_flat = sf.get_model_flattened(net_users[cl], device)
            dif_model = model_flat.sub(1, ps_model_flat)
            selected_avg.add_(1 / len(selected_clients), dif_model)
        selected_avg.mul_(1/args.lr) ## pseudo grad
        # old_momentum = sf.get_model_flattened(net_ps_prev, device)
        # new_momentum = old_momentum.mul(args.beta)
        # new_momentum.add_(1,selected_avg)
        new_momentum = selected_avg
        ps_model_flat.add_(args.alfa*args.lr,new_momentum)

        sf.make_model_unflattened(net_ps_prev, new_momentum, net_sizes, ind_pairs)
        sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)
        [sf.pull_model(net_users[cl], net_ps) for cl in range(args.num_client)]

        # if run %10 == 5: ##debug
        #     acc = evaluate_accuracy(net_ps, testloader, device)
        #     print('accuracy:', acc * 100,run)


        if run % 5 == 0:
            acc = evaluate_accuracy(net_ps, testloader, device)
            accuracys.append(acc * 100)
            print('accuracy:',acc*100)
    return accuracys

def train_fedADCp(args, device):

    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds = dl.get_indices(trainset, args)
    # PS model
    net_ps = get_net(args).to(device)
    net_ps_prev = get_net(args).to(device)
    sf.initialize_zero(net_ps_prev)
    prev_models = [get_net(args).to(device) for u in range(num_client)]
    [sf.initialize_zero(prev_models[u]) for u in range(num_client)]
    modelsize = sf.count_parameters(net_ps)


    net_users = [get_net(args).to(device) for u in range(num_client)]

    criterions = [nn.CrossEntropyLoss() for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)

    # synch all clients models models with PS
    [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]
    optimizers = [torch.optim.SGD(net_users[cl].parameters(), lr=args.lr, weight_decay=1e-4, momentum=0) for cl in
                  range(num_client)]
    net_sizes, net_nelements = sf.get_model_sizes(net_ps)
    ind_pairs = sf.get_indices(net_sizes, net_nelements)
    N_s = (50000 if args.dataset_name == 'cifar10' else 60000)
    accuracys = []

    acc = evaluate_accuracy(net_ps, testloader, device)
    accuracys.append(acc * 100)
    first_model = sf.get_model_flattened(net_ps, device)
    for run in range(args.comm_rounds):
        worker_vec = np.zeros(num_client)
        worker_vec[np.random.choice(range(num_client),int(args.cl * num_client),replace=False)] = 1 ### randomly select workers
        selected_clients = np.where(worker_vec == 1)[0]

        m_local = sf.get_model_flattened(net_ps_prev,device).mul(1/args.LocalIter)
        for cl in selected_clients:
            localIter = 0
            while (localIter < args.LocalIter):
                trainloader = DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                                         shuffle=True)
                for data in trainloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    sf.zero_grad_ps(net_users[cl])
                    predicts = net_users[cl](inputs)
                    loss = criterions[cl](predicts, labels)
                    loss.backward()
                    if run >= args.LocalIter:
                        w_grad = sf.get_grad_flattened(net_users[cl], device)
                        m_local.mul_(args.beta)
                        m_local.add_(1-args.beta,w_grad)
                        sf.make_grad_unflattened(net_users[cl], m_local, net_sizes, ind_pairs)
                    optimizers[cl].step()
                    localIter += 1
                    if localIter == args.LocalIter:
                        break
        ##start comm
        if run < args.LocalIter: ## warm it up
            selected_avg = torch.zeros(modelsize).to(device)
            for cl in selected_clients:  ## update model of the selected group of workers
                model_flat = sf.get_model_flattened(net_users[cl], device)
                selected_avg.add_(1 / len(selected_clients), model_flat)
            sf.make_model_unflattened(net_ps, selected_avg, net_sizes, ind_pairs)
            [sf.pull_model(net_users[cl], net_ps) for cl in range(args.num_client)]
            if args.LocalIter - 1 == run:
                ps_model_flat = sf.get_model_flattened(net_ps, device)
                momentum = first_model.sub(1, ps_model_flat)
                momentum.mul_(1 / args.lr)
                sf.make_model_unflattened(net_ps_prev, momentum, net_sizes, ind_pairs)

        else:
            ps_model_flat = sf.get_model_flattened(net_ps, device)
            selected_avg = torch.zeros_like(ps_model_flat).to(device)
            old_momentum = sf.get_model_flattened(net_ps_prev, device)
            for cl in selected_clients:  ## update model of the selected group of workers
                model_flat = sf.get_model_flattened(net_users[cl], device)
                dif_model = ps_model_flat.sub(1, model_flat)
                dif_model_drift = dif_model.sub(args.beta * args.lr, old_momentum)
                new_dif = dif_model_drift.mul(torch.norm(dif_model) / torch.norm(dif_model_drift))
                selected_avg.add_(1 / len(selected_clients), new_dif)
            selected_avg.mul_(1 / args.lr)  ## pseudo grad
            new_momentum = old_momentum.mul(args.beta)
            new_momentum.add_(1 - args.beta,selected_avg)
            ps_model_flat.sub_(args.alfa * args.lr, new_momentum)

            sf.make_model_unflattened(net_ps_prev, new_momentum, net_sizes, ind_pairs)
            sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)
            [sf.pull_model(net_users[cl], net_ps) for cl in range(args.num_client)]

        if run % 5 == 0:
            acc = evaluate_accuracy(net_ps, testloader, device)
            accuracys.append(acc * 100)
            print('accuracy:',acc*100)
    return accuracys

def train_fedADCn(args, device):

    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds = dl.get_indices(trainset, args)
    # PS model
    net_ps = get_net(args).to(device)
    net_ps_prev = get_net(args).to(device)
    sf.initialize_zero(net_ps_prev)
    prev_models = [get_net(args).to(device) for u in range(num_client)]
    [sf.initialize_zero(prev_models[u]) for u in range(num_client)]



    net_users = [get_net(args).to(device) for u in range(num_client)]

    criterions = [nn.CrossEntropyLoss() for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)
    optimizers = [torch.optim.SGD(net_users[cl].parameters(), lr=args.lr, weight_decay=1e-4, momentum=0) for cl in
                  range(num_client)]
    # synch all clients models models with PS
    [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]

    net_sizes, net_nelements = sf.get_model_sizes(net_ps)
    ind_pairs = sf.get_indices(net_sizes, net_nelements)
    N_s = (50000 if args.dataset_name == 'cifar10' else 60000)
    accuracys = []

    acc = evaluate_accuracy(net_ps, testloader, device)
    accuracys.append(acc * 100)
    for run in range(args.comm_rounds):
        worker_vec = np.zeros(num_client)
        worker_vec[np.random.choice(range(num_client),int(args.cl * num_client),replace=False)] = 1 ### randomly select workers
        selected_clients = np.where(worker_vec == 1)[0]

        m_global = (sf.get_model_flattened(net_ps_prev, device)).mul(1 / args.LocalIter)
        m_local = m_global.mul(args.beta/ args.LocalIter)
        for cl in selected_clients:
            localIter = 0
            while (localIter < args.LocalIter):
                trainloader = DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                                         shuffle=True)
                for data in trainloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    sf.zero_grad_ps(net_users[cl])
                    w_grad = torch.zeros_like(m_local)
                    if run == 0:
                        predicts = net_users[cl](inputs)
                        loss = criterions[cl](predicts, labels)
                        loss.backward()
                        w_grad = sf.get_grad_flattened(net_users[cl], device)
                    elif run<args.LocalIter and run != 0:
                        predicts = net_users[cl](inputs)
                        loss = criterions[cl](predicts, labels)
                        loss.backward()
                        w_grad = sf.get_grad_flattened(net_users[cl], device)
                        psuedo_grad = m_local.mul(args.beta_warm)
                        psuedo_grad.add_(1-args.beta_warm, w_grad)
                        w_grad = psuedo_grad
                    else:
                        if args.l_update_ver ==1:
                            w_model = sf.get_model_flattened(net_users[cl], device)
                            sf.make_model_unflattened(net_users[cl], w_model.add(args.lr*args.beta,m_local), net_sizes, ind_pairs)
                            predicts = net_users[cl](inputs)
                            loss = criterions[cl](predicts, labels)
                            loss.backward()
                            w_grad = sf.get_grad_flattened(net_users[cl], device)
                            w_grad.mul_(1 - args.beta)
                        elif args.l_update_ver ==2:
                            predicts = net_users[cl](inputs)
                            loss = criterions[cl](predicts, labels)
                            loss.backward()
                            w_grad = sf.get_grad_flattened(net_users[cl], device).mul(1-args.beta)
                            w_grad.add_(args.beta, m_local)
                    sf.make_grad_unflattened(net_users[cl], w_grad,net_sizes, ind_pairs)
                    optimizers[cl].step()
                    localIter += 1
                    if localIter == args.LocalIter:
                        break
        ##start comm
        ps_model_flat =sf.get_model_flattened(net_ps, device)
        selected_avg = torch.zeros_like(ps_model_flat).to(device)
        for cl in selected_clients:  ## update model of the selected group of workers
            model_flat = sf.get_model_flattened(net_users[cl], device)
            dif_model = ps_model_flat.sub(1, model_flat)
            selected_avg.add_(1 / len(selected_clients), dif_model)
        selected_avg.mul_(1/args.lr) ## pseudo grad
        new_momentum = selected_avg
        ps_model_flat.sub_(args.alfa*args.lr,new_momentum)

        sf.make_model_unflattened(net_ps_prev, new_momentum, net_sizes, ind_pairs)
        sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)
        [sf.pull_model(net_users[cl], net_ps) for cl in range(args.num_client)]

        # if run %10 == 5: ##debug
        #     acc = evaluate_accuracy(net_ps, testloader, device)
        #     print('accuracy:', acc * 100,run)


        if run % 5 == 0:
            acc = evaluate_accuracy(net_ps, testloader, device)
            accuracys.append(acc * 100)
            print('accuracy:',acc*100)
    return accuracys