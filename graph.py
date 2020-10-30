import numpy as np
import matplotlib.pyplot as plt
from os import *
from parameters import *
import math
from itertools import cycle
import itertools
import argparse

def getCombinations():
    work_load = []
    combinations = []
    args = args_parser_loop()
    for arg in vars(args):
        arg_type = type(getattr(args, arg))
        if arg_type == list and arg != 'lr_change' and arg != 'excluded_gpus':
            work_ = [n for n in getattr(args, arg)]
            work_load.append(work_)
    for t in itertools.product(*work_load):
        combinations.append(t)
    return combinations

def special_adress():
    adress=[]
    labels = []
    args = args_parser_loop()
    combinations = getCombinations()

    for combination in combinations:
        if args.mode == 'AFL':
            newFile = '{}-VER_{}-{}-cls_{}-H_{}-A_{}-B_{}-LR_{}'.format(args.mode, args.l_update_ver,
                                                                           args.P_M_ver, combination[0], args.LocalIter,
                                                                           args.alfa, combination[1], combination[2])
        else:
            newFile = '{}-cls_{}-H_{}-A:{}-B:{}-LR:{}'.format(args.mode, args.numb_cls_usr,
                                                                 args.LocalIter, args.alfa, args.beta, args.lr)

        adress.append('Results/{}'.format(newFile))
        labels.append(newFile)
    print(labels)



    assert len(adress) == len(labels)
    return adress,labels

def compile_results(adress):
    results = []
    f_results = []
    counter = 0
    d_counter = 0
    for i, dir in enumerate(listdir(adress)):
        if dir[0:3] != 'sim':
            vec = np.load(adress + '/'+dir)
            final_result = vec[len(vec)-1]
            if final_result>20:
                f_results.append(final_result)
            else:
                d_counter +=1
            if len(results)==0 and final_result>20:
                results = vec/len(listdir(adress))
            elif final_result>20:
                results += vec/len(listdir(adress))
            counter +=1
    avg = 'DIVERGE' if len(f_results) ==0 else np.average(f_results)
    st_dev ='DIVERGE' if len(f_results) ==0 else np.std(f_results)

    return results, [adress,avg,st_dev, d_counter/counter*100]

def cycle_graph_props(colors,markers,linestyles):
    randoms =[]
    randc = np.random.randint(0,len(colors))
    randm = np.random.randint(0,len(markers))
    randl = np.random.randint(0,len(linestyles))
    m = markers[randm]
    c = colors[randc]
    l = linestyles[randl]
    np.delete(colors,randc)
    np.delete(markers,randm)
    np.delete(linestyles,randl)
    print(colors,markers,linestyles)
    return c,m,l


def avgs(sets):
    avgs =[]
    for set in sets:
        avg = np.zeros_like(set[0])
        avgs.append(avg)
    return avgs

def graph(data, legends,interval):
    marker = ['s', 'v', '+', 'o', '*']
    linestyle =['-', '--', '-.', ':']
    linecycler = cycle(linestyle)
    markercycler = cycle(marker)
    table_data =[]
    for d,legend in zip(data,legends):
        x_axis = []
        table_data.append([legend,d[len(d)-1]])
        l = next(linecycler)
        m = next(markercycler)
        for i in range(0,len(d)):
            x_axis.append(i*interval)
        plt.plot(x_axis,d, marker= m ,linestyle = l ,markersize=2, label=legend)
    plt.axis([0, 1000, 50, 90])
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy')
    plt.title('Federated-Learning')
    plt.legend()
    plt.grid(True)
    plt.show()

def table(data, legends,interval):
    fig = plt.figure(dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    table_data = []
    for d, legend in zip(data, legends):
        table_data.append([legend, d[len(d) - 1]])
    table = ax.table(cellText=table_data, loc='center')
    table.set_fontsize(14)
    table.scale(1, 1)
    ax.axis('off')
    plt.show()


def concateresults(dirsets):
    all_results =[]
    for set in dirsets:
        try:
            listdir(set)
            all_results.append(compile_results(set)[0])
            print(compile_results(set)[1])
        except:
            print('patlamış',set)
            continue
    return all_results


loc = 'Results/'
types = ['benchmark','timeCorrelated','topk']
NNs = ['simplecifar']

locations = []
labels =[]
for tpye in types:
    for nn in NNs:
        locations.append(loc + tpye +'/'+nn)
        labels.append(tpye +'--'+ nn)

intervels = 5
labels = special_adress()[1]
results = concateresults(special_adress()[0])
#results = concateresults(locations)
graph(results,labels,intervels)
#table(results,labels,intervels)
#data,legends = compile_results(loc)
#graph(data,labels,intervels)