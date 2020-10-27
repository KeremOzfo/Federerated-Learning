import numpy as np
import matplotlib.pyplot as plt
from os import *
import math
from itertools import cycle
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

def special_adress():
    adress=[]
    labels = []
    # labels = ['alg1-old','Benchmark-Federated(no M)', 'alg1beta9-sigma9', 'alg2beta9-sigma8' , 'alg2beta9-sigma9'
    #           , 'alg2beta10-sigma8', 'alg2beta10-sigma9', 'alg2beta95-sigma8', 'alg2beta95-sigma9'
    #           , 'alg3beta9-sigma8', 'alg3beta9-sigma9', 'alg3beta10-sigma8', 'alg3beta10-sigma9']
    #
    # labels = ['alg1-beta9-sigma9-gamma10','alg2-beta9-sigma9-gamma9','alg2-beta9-sigma9-gamma10',
    #           'alg2-beta9-sigma95-gamma10','alg2-beta95-sigma9-gamma9','alg2-beta95-sigma9--gamma10'
    #           ,'slowmo-alpha9-beta8', 'slowmo-alpha9-beta9', 'slowmo-alpha10-beta8', 'slowmo-alpha10-beta9']

    # labels = ['slowmo2cls', 'slowmo3cls','slowmo4cls10iter',
    #           'new_nestrov2cls', 'new_nestrov3cls', 'new_nestrov4cls10L',
    #           'fedavg_2cls','fedavg_3cls','fedavg_4cls10L']
    #labels = ['SlowMo-cls_4','SlowMo-cls_3','SlowMo-cls_2']

    folder = 'slowmo'
    for dir in listdir('Results/{}'.format(folder)):
        labels.append(str(dir))


    for l in labels:
        adress.append('Results/{}/{}'.format(folder,l))



    assert len(adress) == len(labels)
    return adress,labels

def compile_results(adress):
    results = []
    f_results = []
    counter = 0
    d_counter = 0
    for i, dir in enumerate(listdir(adress)):
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
    #plt.axis([5, 45,70 ,90])
    #plt.axis([145,155,88,92])
    #plt.axis([290, 300, 87, 95])
    #plt.axis([50, 100, 87, 95])
    plt.axis([0, 1000, 50, 90])
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy')
    plt.title('Federated-Learning')
    plt.legend()
    # ax = plt.table(cellText=table_data, loc='bottom')
    # ax.set_fontsize(3)
    # ax.scale(1, 0.5)
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
        all_results.append(compile_results(set)[0])
        print(compile_results(set)[1])
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