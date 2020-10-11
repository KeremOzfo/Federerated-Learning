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
            for cl in (np.where(worker_vec == 1)[0]): ## update the selected group of workers
                model_flat = sf.get_model_flattened(net_users[cl], device)
                dif_model = model_flat.sub(1, ps_model_flat)
                selected_avg.add_(1/args.worker_blocks, dif_model)
            ps_model_flat.add_(1, selected_avg)
            sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)
            [sf.pull_model(net_users[cl], net_ps) for cl in (np.where(worker_vec == 1)[0])]
            for cl in (np.where(worker_vec == 0)[0]): ## update the other workers
                model_flat = sf.get_model_flattened(net_users[cl], device)
                dif_model = model_flat.sub(1, ps_model_flat)
                modified_dif = selected_avg.mul(10)
                modified_dif.add_(1,dif_model)
                model_flat.add_(1/(args.worker_blocks +1), modified_dif)
                sf.make_model_unflattened(net_users[cl], model_flat, net_sizes, ind_pairs)
            new_momentum = selected_avg.mul(args.gamma/(args.LocalIter* args.lr))
            new_momentum.add_(1-args.gamma, sf.get_model_flattened(net_ps_prev,device))
            sf.make_model_unflattened(net_ps_prev, new_momentum, net_sizes, ind_pairs)
            [sf.pull_model(prev_models[cl], net_ps_prev) for cl in range(args.num_client)]

        # if run %10 == 5: ##debug
        #     acc = evaluate_accuracy(net_ps, testloader, device)
        #     print('accuracy:', acc * 100,run)


        if run % args.LocalIter == 0:
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
            [sf.update_model(net_users[cl], prev_models[cl], lr=0.1, momentum=0.9, weight_decay=1e-4) for cl in
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
            for cl in (np.where(worker_vec == 1)[0]):
                model_flat = sf.get_model_flattened(net_users[cl], device)
                dif_model = model_flat.sub(1, ps_model_flat)
                selected_avg.add_(1/args.worker_blocks, dif_model)
            ps_model_flat.add_(1, selected_avg)
            sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)
            [sf.pull_model(net_users[cl], net_ps) for cl in (np.where(worker_vec == 1)[0])]
            for cl in (np.where(worker_vec == 0)[0]):
                model_flat = sf.get_model_flattened(net_users[cl], device)
                dif_model = model_flat.sub(1, ps_model_flat)
                modified_dif = selected_avg.mul(10)
                modified_dif.add_(1,dif_model)
                model_flat.add_(1/(args.worker_blocks +1), modified_dif)
                sf.make_model_unflattened(net_users[cl], model_flat, net_sizes, ind_pairs)
            new_momentum = selected_avg.mul(1/args.LocalIter)
            sf.make_model_unflattened(net_ps_prev, new_momentum, net_sizes, ind_pairs)
            [sf.pull_model(prev_models[cl], net_ps_prev) for cl in range(args.num_client)]

        # if run %10 == 5: ##debug
        #     acc = evaluate_accuracy(net_ps, testloader, device)
        #     print('accuracy:', acc * 100,run)


        if run % args.LocalIter == 0:
            acc = evaluate_accuracy(net_ps, testloader, device)
            accuracys.append(acc * 100)
            print('accuracy:',acc*100)
    return accuracys