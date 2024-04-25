#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 10:14:16 2019

@author: liaohaiguang
"""

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import collections as col

import AStarSearchSolver as solver

np.random.seed(1)

# Input parameters:
# Default setting: minWidth = 1, minSpacing = 0, 
# tileLength = 10, tileHeight = 10

# 1. Grid Size: for 8-by-8 benchmark, grid size is 8
# 2. vCap and hCap represents vertical capacity for layer2,
# and horizontal capacity for layer 1; unspecified capacity are 
# by default 0
# 3. maxPinNum: maximum number of pins of one net
# 4. capReduce: number of capacity reduction specification, by default = 0

def generator(benchmark_name,gridSize,netNum,vCap,hCap,maxPinNum,savepath):
    
    file = open('%s' % benchmark_name, 'w+')
    
    # Write general information
    file.write('grid {gridSize} {gridSize} 2\n'.format(gridSize=gridSize))
    file.write('vertical capacity 0 {vCap}\n'.format(vCap=vCap))
    file.write('horizontal capacity {hCap} 0\n'.format(hCap=hCap))
    file.write('minimum width 1 1\n')
    file.write('minimum spacing 0 0\n')
    file.write('via spacing 0 0\n')
    file.write('0 0 10 10\n')
    file.write('num net {netNum}\n'.format(netNum=netNum))
    # Write nets information 
    pinNum = np.random.randint(2,maxPinNum+1,netNum) # Generate Pin Number randomly
    for i in range(netNum):
        specificPinNum = pinNum[i]
        file.write('A{netInd} 0{netInd} {pin} 1\n'.format(netInd=i+1,pin=specificPinNum))
        xCoordArray = np.random.randint(1,10*gridSize,specificPinNum)
        yCoordArray = np.random.randint(1,10*gridSize,specificPinNum)
        for j in range(specificPinNum):
            file.write('{x}  {y} 1\n'.format(x=xCoordArray[j],y=yCoordArray[j]))
    # Write capacity information
    file.write('{capReduce}'.format(capReduce=0))
    file.close()
    return

def generator_reducedCapacity(benchmark_name,gridSize,netNum,vCap,hCap,maxPinNum,\
    savepath,reducedCapNum,connection_statistical_array):
    # generate benchmarks with reduced capacity
    file = open('%s' % benchmark_name, 'w+')
    
    # Write general information
    file.write('grid {gridSize} {gridSize} 2\n'.format(gridSize=gridSize))
    file.write('vertical capacity 0 {vCap}\n'.format(vCap=vCap))
    file.write('horizontal capacity {hCap} 0\n'.format(hCap=hCap))
    file.write('minimum width 1 1\n')
    file.write('minimum spacing 0 0\n')
    file.write('via spacing 0 0\n')
    file.write('0 0 10 10\n')
    file.write('num net {netNum}\n'.format(netNum=netNum))
    # Write nets information 
    pinNum = np.random.randint(2,maxPinNum+1,netNum) # Generate Pin Number randomly
    for i in range(netNum):
        specificPinNum = pinNum[i]
        file.write('A{netInd} 0{netInd} {pin} 1\n'.format(netInd=i+1,pin=specificPinNum))
        xCoordArray = np.random.randint(1,10*gridSize,specificPinNum)
        yCoordArray = np.random.randint(1,10*gridSize,specificPinNum)
        for j in range(specificPinNum):
            file.write('{x}  {y} 1\n'.format(x=xCoordArray[j],y=yCoordArray[j]))
    # Write capacity information
    file.write('{capReduce}\n'.format(capReduce=reducedCapNum))
    for i in range(reducedCapNum):
        obstaclei = connection_statistical_array[-(i+1),0:6]
        file.write('{a} {b} {c}   {d} {e} {f}   3\n'.format(a=int(obstaclei[0]),b=int(obstaclei[1]),c=int(obstaclei[2]),\
         d=int(obstaclei[3]),e=int(obstaclei[4]),f=int(obstaclei[5])))
    file.close()
    return
#这个函数 generator_reducedCapacity 用于生成具有减少容量的基准测试。它首先创建一个新文件，写入基准的一般信息，如网格大小、垂直和水平容量等。
# 接着，它随机生成每个网络的引脚数量，并为每个网络的每个引脚随机分配坐标。此外，该函数还会根据 reducedCapNum 参数减少特定数量的连接容量，并在文件中记录这些信息。
# 最后，文件被保存，并且函数结束。这个过程有助于模拟和评估在容量受限的条件下全局布线算法的性能。

#  edge_traffic_stat get statiscal information of edge traffic by solving problems
#  with A* search
def edge_traffic_stat(edge_traffic,gridSize):
    via_capacity  = np.zeros((gridSize,gridSize))
    hoz_capacity = np.zeros((gridSize,gridSize-1)) # Only for Layer 1
    vet_capacity = np.zeros((gridSize-1,gridSize)) # Only for Layer 2
    for i in range(edge_traffic.shape[0]):
        connection = edge_traffic[i,:].astype(int)
#        print(connection)
        diff = (connection[3]-connection[0],\
                connection[4]-connection[1],\
                connection[5]-connection[2])
        if diff[0] == 1:
            hoz_capacity[connection[1],connection[0]] \
            = hoz_capacity[connection[1],connection[0]] + 1
        elif diff[0] == -1:
            hoz_capacity[connection[1],int(connection[0])-1] \
            = hoz_capacity[connection[1],int(connection[0])-1] + 1
        elif diff[1] == 1:
            vet_capacity[connection[1],connection[0]] \
            = vet_capacity[connection[1],connection[0]] + 1
        elif diff[1] == -1:
            vet_capacity[int(connection[1])-1,connection[0]] \
            = vet_capacity[int(connection[1])-1,connection[0]] + 1
        elif abs(diff[2]) == 1:
            via_capacity[connection[0],connection[1]] \
            = via_capacity[connection[0],connection[1]] + 1
        else:
            continue
    return via_capacity, hoz_capacity, vet_capacity
# 这段代码定义了一个函数 edge_traffic_stat，用于统计给定网格大小 gridSize 下的边缘流量 edge_traffic。
# 函数首先创建三个零矩阵来存储通过容量(via_capacity)、水平容量(hoz_capacity)和垂直容量(vet_capacity)。
# 然后，它遍历边缘流量数据，根据连接点的坐标差异来更新相应方向上的容量。如果水平方向上的坐标差为1或-1，则更新水平容量；如果垂直方向上的坐标差为1或-1，则更新垂直容量；

# 如果层之间的坐标差为1，则更新通过容量。这样，函数最终返回三个方向上的容量统计。
def connection_statistical(edge_traffic,gridSize,benchmarkNumber):
    # get connection statistcal in vertical and horizontal direction, 
    # ignoring the via capacity since they tends to be large (set as 10 in simulated env)
    # cleaned edge traffic as input
    connection_cleaned = np.empty((0,6))
    for i in range(edge_traffic.shape[0]):
        connection = edge_traffic[i,:]
        if connection[3]<connection[0] or connection[4]<connection[1] or connection[5]<connection[2]:
            connection_flip = np.asarray([connection[3],connection[4],connection[5],\
                connection[0],connection[1],connection[2]])
            connection_cleaned = np.vstack((connection_cleaned,connection_flip))
        else:
            connection_cleaned = np.vstack((connection_cleaned,connection))

    connection_statistical = np.empty((0,7)) # last position is used for counting 
    connection_list = []

    for i in range(connection_cleaned.shape[0]):
        connectioni = connection_cleaned[i,:]
        # remove via connection before append to list
        if connectioni[2] == connectioni[5]:
            connection_list.append(tuple(connectioni))
    counter = col.Counter(connection_list)
    for key, value in counter.items():
        # print (key,value)
        statisticali = [int(i) for i in key]
        # Normalize the last column by benchmark numbers
        statisticali.append(int(value/benchmarkNumber))
        statisticali = np.asarray(statisticali)
        connection_statistical = np.vstack((connection_statistical,statisticali))
    # Sort connection_statistical
    connection_statistical_list = connection_statistical.tolist()
    connection_statistical_list.sort(key=lambda x: x[6])

    connection_statistical_array = np.asarray(connection_statistical_list)

    return connection_statistical_array
# 这个函数 connection_statistical 的目的是为了在水平和垂直方向上获取连接统计信息，忽略了通过容量（在模拟环境中设置为10）因为它们通常较大。
# 函数首先清理输入的边缘流量数据，将所有反向的连接翻转以统一方向。然后，它移除所有包含通过连接的数据，只保留水平和垂直连接。
# 接下来，使用 collections.Counter 来统计每种连接的出现次数，并将这些次数除以基准数量以归一化。
# 最后，这些统计数据被排序并返回。这个过程有助于理解在给定基准测试中，不同连接类型的分布和频率。

def parse_arguments():
    parser = argparse.ArgumentParser('Benchmark Generator Parser')
    parser.add_argument('--benchNumber',type=int,\
        dest='benchmarkNumber',default=20)
    parser.add_argument('--gridSize',type=int,dest='gridSize',default=16)
    parser.add_argument('--netNum',type=int,dest='netNum',default=5)
    parser.add_argument('--capacity',type=int,dest='cap',default=4)
    parser.add_argument('--maxPinNum',type=int,dest='maxPinNum',default=5)
    parser.add_argument('--reducedCapNum',type=int,dest='reducedCapNum',default=0) # EvenNumber
    return parser.parse_args()
# 这段代码定义了一个函数 parse_arguments()，用于解析命令行参数。它创建了一个 argparse.ArgumentParser 对象，并定义了几个参数：
# benchNumber（基准数量，默认为20），gridSize（网格大小，默认为16），netNum（网络数量，默认为5），capacity（容量，默认为4），
# maxPinNum（最大引脚数，默认为5），以及reducedCapNum（减少的容量数，默认为0）。这些参数允许用户自定义生成基准测试时的不同设置。最后，函数返回解析后的参数对象。

if __name__ == "__main__":
    args = parse_arguments()
    benchmarkNumber = args.benchmarkNumber

    os.chdir('..')
    os.system('rm -r benchmark')
    os.system('rm -r capacityplot_A*')
    os.system('rm -r solutionsA*')
    os.system('rm -r benchmark_reduced')
    os.system('rm -r capacityplot_A*_reduced')
    os.system('rm -r solutionsA*_reduced')

    os.chdir('BenchmarkGenerator')

    # benchmarkNumber = 20
    savepath = '../benchmark/'  # specify path to save benchmarks
    os.makedirs(savepath)
    os.chdir(savepath)
     
    gridSize = args.gridSize; netNum = args.netNum
    vCap = args.cap; hCap = args.cap; maxPinNum = args.maxPinNum
    reducedCapNum = args.reducedCapNum

    for i in range(benchmarkNumber):
        benchmark_name = "test_benchmark_{num}.gr".format(num=i+1)
        generator(benchmark_name,gridSize,netNum,vCap,hCap,maxPinNum,savepath)
    
    
    # Get statistical information about edge traffic by solving benchmarks
    # with A*Star Search
        # initialize edge traffic basket 
        #(data structure: startTile x,y,z and endTile x,y,z )
    
    edge_traffic =np.empty(shape=(0,6))
       # solving problems with A* search
    os.chdir("../benchmark/")
#    benchmarkfile = "test_benchmark_5.gr"
    solution_savepath = '../solutionsA*/' 
    if not os.path.exists('../capacityPlot_A*/'):
        os.mkdir('../capacityPlot_A*/')

    os.mkdir(solution_savepath)
    benEnum = 0
    for benchmarkfile in os.listdir('.'):
        benEnum = benEnum + 1
        edge_traffic_individual =np.empty(shape=(0,6))
        routeListMerged = solver.solve(benchmarkfile,solution_savepath)
        print('routeListMerged',routeListMerged)
        for netCount in range(len(routeListMerged)):
            for pinCount in range(len(routeListMerged[netCount])-1):
                pinNow = routeListMerged[netCount][pinCount]
                pinNext = routeListMerged[netCount][pinCount+1]
                connection = [int(pinNow[3]),int(pinNow[4]),int(pinNow[2]),\
                              int(pinNext[3]),int(pinNext[4]),int(pinNext[2])]
                edge_traffic_individual = np.vstack((edge_traffic_individual,connection))
                edge_traffic = np.vstack((edge_traffic,connection))
# 这段代码的目的是使用A*搜索算法解决布线问题，并收集处理过程中每一对连接点的数据。首先，它初始化一个空的edge_traffic数组用于存储连接数据。
        # 然后，它更改工作目录到"benchmark/"，并为解决方案和容量图创建保存路径。接下来，代码遍历所有基准文件，
        # 对每个文件使用求解器solver.solve来找到布线解决方案，同时收集并保存每一条路径上的连接点数据到edge_traffic_individual和edge_traffic中。
        # 这样，最终edge_traffic包含了所有基准测试中的连接数据，用于后续的分析和处理。

        # calculate capacity utilization for individual benchmark 
        totalEdgeUtilization = edge_traffic_individual.shape[0]
        connection_statistical_array = connection_statistical(edge_traffic_individual,gridSize,1)
        totalEdgeUtilized = connection_statistical_array.shape[0]
        edge_utilize_plot = vCap - connection_statistical_array[:,-1] # Assumption: vCap = hCap
        plt.figure()
        plt.plot(edge_utilize_plot,'r',label="Capacity after A* route")
        plt.plot(vCap*np.ones_like(edge_utilize_plot),'b',label="Capacity before A* route")
        plt.ylabel('Remaining capacity')
        plt.legend()
        plt.savefig('../capacityPlot_A*/edgePlotwithCapacity{cap}number{ben}.png'.format(cap=vCap,ben=benEnum))

        # Draw heatmap of individual problems
        via_capacity_individual,hoz_capacity_individual,vet_capacity_individual =\
         edge_traffic_stat(edge_traffic_individual,gridSize)
        plt.figure()
        plt.imshow(via_capacity_individual,cmap='hot', interpolation='nearest')
        plt.title('Via Capacity Heatmap')
        if not os.path.exists('../capacityPlot_A*/'):
            os.mkdir('../capacityPlot_A*/')
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.savefig('../capacityPlot_A*/viaCapacity_{pro}.png'.format(pro=benchmarkfile))
        
        plt.figure()
        plt.imshow(vet_capacity_individual,cmap='hot', interpolation='nearest')
        plt.title('Vertical Capacity Heatmap (Layer2)')
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.savefig('../capacityPlot_A*/vetCapacity_{pro}.png'.format(pro=benchmarkfile))
        
        plt.figure()
        plt.imshow(hoz_capacity_individual,cmap='hot', interpolation='nearest')
        plt.title('Horizontal Capacity Heatmap (Layer1)')
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.savefig('../capacityPlot_A*/hozCapacity_{pro}.png'.format(pro=benchmarkfile))

    # calculate capacity utilization
    # print("Total num of edge uitilization: ",edge_traffic.shape[0]) # print total num of edge uitilization
    totalEdgeUtilization = edge_traffic.shape[0]
    via_capacity,hoz_capacity,vet_capacity = edge_traffic_stat(edge_traffic,gridSize)
    connection_statistical_array = connection_statistical(edge_traffic,gridSize,benchmarkNumber)
    # print('connection_statistical_array',connection_statistical_array[-10:-1,:])
    totalEdgeUtilized = connection_statistical_array.shape[0]
    # print('totalEdgeUtilized',totalEdgeUtilized)
#    print(via_capacity)
#    for i in range(via_capacity.shape[0]):
#        for j in range(via_capacity.shape[1]):
#            print(via_capacity[i,j])
        
     #draw a heat map of capacity utilization
    edgeNum = gridSize*gridSize + 2*gridSize*(gridSize-1)
    utilization_frequency = np.empty((0,2)) # 0 column: utilization times, 1 column: num of edges
    unsedEdge = [int(0),int(np.abs(edgeNum-totalEdgeUtilized))]
    utilization_frequency = np.vstack((utilization_frequency,unsedEdge))
    frequency_basket = []
    for i in range(connection_statistical_array.shape[0]):
        frequency_basket.append(int(connection_statistical_array[i,6]))
    counter_frequency = col.Counter(frequency_basket)
    for key, value in counter_frequency.items():
        frequencyi = [key,value] 
        utilization_frequency = np.vstack((utilization_frequency,frequencyi))
        # for key, value in counter.items():
        # # print (key,value)
        # statisticali = [int(i) for i in key]
        # statisticali.append(value)
        # statisticali = np.asarray(statisticali)
        # connection_statistical = np.vstack((connection_statistical,statisticali))
    print(utilization_frequency)


    plt.figure()
    plt.imshow(via_capacity,cmap='hot', interpolation='nearest')
    plt.title('Via Capacity Heatmap')
    if not os.path.exists('../capacityPlot_A*/'):
    	os.mkdir('../capacityPlot_A*/')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('../capacityPlot_A*/viaCapacity.png')
    
    plt.figure()
    plt.imshow(vet_capacity,cmap='hot', interpolation='nearest')
    plt.title('Vertical Capacity Heatmap (Layer2)')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('../capacityPlot_A*/vetCapacity.png')
    
    plt.figure()
    plt.imshow(hoz_capacity,cmap='hot', interpolation='nearest')
    plt.title('Horizontal Capacity Heatmap (Layer1)')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('../capacityPlot_A*/hozCapacity.png')

    plt.figure()
    # plt.plot(utilization_frequency[:,0],utilization_frequency[:,1])
    plt.bar(utilization_frequency[:,0],utilization_frequency[:,1])
    plt.plot(utilization_frequency[:,0],utilization_frequency[:,1],'r-')

    plt.title('Edge Utilization Histogram')
    plt.savefig('../capacityPlot_A*/edgeHist.png')
    


    # # For plotting edge utilization, only those visited edges are plotted; via not taken into account
    # edge_utilize_plot = vCap - connection_statistical_array[:,-1] # Assumption: vCap = hCap
    # plt.figure()
    # plt.plot(edge_utilize_plot,'r',label="Capacity after A* route")
    # plt.plot(vCap*np.ones_like(edge_utilize_plot),'b',label="Capacity before A* route")
    # plt.ylabel('Remaining capacity')
    # plt.legend()
    # plt.savefig('../capacityPlot_A*/edgePlotwithCapacity{cap}.png'.format(cap=vCap))



    #  Deal with benchmark with reduced capacity
    os.chdir('../BenchmarkGenerator/') # Go back to BenchmarkGenerator folder
    
    savepath = '../benchmark_reduced/'  # specify path to save benchmarks
    os.makedirs(savepath)
    os.chdir(savepath)

    if not os.path.exists('../capacityPlot_A*_reduced/'):
        os.mkdir('../capacityPlot_A*_reduced/')

    for i in range(benchmarkNumber):
        benchmark_name = "test_benchmark_{num}.gr".format(num=i+1)
        generator_reducedCapacity(benchmark_name,gridSize,netNum,vCap,hCap,maxPinNum,savepath,\
            reducedCapNum,connection_statistical_array)

  # Get statistical information about edge traffic by solving benchmarks
    # with A*Star Search
        # initialize edge traffic basket 
        #(data structure: startTile x,y,z and endTile x,y,z )
    edge_traffic =np.empty(shape=(0,6))
    
       # solving problems with A* search
    os.chdir("../benchmark_reduced/")
#    benchmarkfile = "test_benchmark_5.gr"
    solution_savepath = '../solutionsA*_reduced/' 
    os.mkdir(solution_savepath)
    benEnum = 0
    for benchmarkfile in os.listdir('.'):
        benEnum = benEnum + 1
        edge_traffic_individual =np.empty(shape=(0,6))
        routeListMerged = solver.solve(benchmarkfile,solution_savepath)
        for netCount in range(len(routeListMerged)):
            for pinCount in range(len(routeListMerged[netCount])-1):
                pinNow = routeListMerged[netCount][pinCount]
                pinNext = routeListMerged[netCount][pinCount+1]
                connection = [int(pinNow[3]),int(pinNow[4]),int(pinNow[2]),\
                              int(pinNext[3]),int(pinNext[4]),int(pinNext[2])]
                edge_traffic_individual = np.vstack((edge_traffic_individual,connection))
                edge_traffic = np.vstack((edge_traffic,connection))
                # calculate capacity utilization for individual benchmark 
        

        totalEdgeUtilization = edge_traffic_individual.shape[0]
        connection_statistical_array = connection_statistical(edge_traffic_individual,gridSize,1)
        totalEdgeUtilized = connection_statistical_array.shape[0]
        edge_utilize_plot = vCap - connection_statistical_array[:,-1] # Assumption: vCap = hCap
        plt.figure()
        plt.plot(edge_utilize_plot,'r',label="Capacity after A* route")
        plt.plot(vCap*np.ones_like(edge_utilize_plot),'b',label="Capacity before A* route")
        plt.ylabel('Remaining capacity')
        plt.legend()
        plt.savefig('../capacityPlot_A*_reduced/edgePlotwithCapacity{cap}number{ben}.png'.format(cap=vCap,ben=benEnum))

        # Draw heatmap of individual problem
        via_capacity_individual,hoz_capacity_individual,vet_capacity_individual =\
         edge_traffic_stat(edge_traffic_individual,gridSize)

        plt.figure()
        plt.imshow(via_capacity_individual,cmap='hot')
        plt.title('Via Capacity Heatmap')
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.savefig('../capacityPlot_A*_reduced/viaCapacity_{pro}.png'.format(pro=benchmarkfile))
        
        plt.figure()
        plt.imshow(vet_capacity_individual,cmap='hot', interpolation='nearest')
        plt.title('Vertical Capacity Heatmap (Layer2)')
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.savefig('../capacityPlot_A*_reduced/vetCapacity_{pro}.png'.format(pro=benchmarkfile))
        
        plt.figure()
        plt.imshow(hoz_capacity_individual,cmap='hot', interpolation='nearest')
        plt.title('Horizontal Capacity Heatmap (Layer1)')
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.savefig('../capacityPlot_A*_reduced/hozCapacity_{pro}.png'.format(pro=benchmarkfile))


    # calculate capacity utilization
    # print("Total num of edge uitilization: ",edge_traffic.shape[0]) # print total num of edge uitilization
    totalEdgeUtilization = edge_traffic.shape[0]
    via_capacity,hoz_capacity,vet_capacity = edge_traffic_stat(edge_traffic,gridSize)
    connection_statistical_array = connection_statistical(edge_traffic,gridSize,benchmarkNumber)
    # print(connection_statistical_array[-30:-1,:])
    totalEdgeUtilized = connection_statistical_array.shape[0]
    # print('totalEdgeUtilized',totalEdgeUtilized)
#    print(via_capacity)
#    for i in range(via_capacity.shape[0]):
#        for j in range(via_capacity.shape[1]):
#            print(via_capacity[i,j])
        
     #draw a heat map of capacity utilization
    edgeNum = gridSize*gridSize + 2*gridSize*(gridSize-1)
    utilization_frequency = np.empty((0,2)) # 0 column: utilization times, 1 column: num of edges
    unsedEdge = [int(0),int(np.abs(edgeNum-totalEdgeUtilized))]
    utilization_frequency = np.vstack((utilization_frequency,unsedEdge))
    frequency_basket = []
    for i in range(connection_statistical_array.shape[0]):
        frequency_basket.append(int(connection_statistical_array[i,6]))
    counter_frequency = col.Counter(frequency_basket)
    for key, value in counter_frequency.items():
        frequencyi = [key,value] 
        utilization_frequency = np.vstack((utilization_frequency,frequencyi))
        # for key, value in counter.items():
        # # print (key,value)
        # statisticali = [int(i) for i in key]
        # statisticali.append(value)
        # statisticali = np.asarray(statisticali)
        # connection_statistical = np.vstack((connection_statistical,statisticali))
    # print(utilization_frequency)

    plt.figure()
    plt.imshow(via_capacity,cmap='hot', interpolation='nearest')
    plt.title('Via Capacity Heatmap')
    plt.colorbar()
    plt.gca().invert_yaxis()
    os.mkdir('../capacityPlot_A*_reduced')
    plt.savefig('../capacityPlot_A*_reduced/viaCapacity.png')
    
    plt.figure()
    plt.imshow(vet_capacity,cmap='hot', interpolation='nearest')
    plt.title('Vertical Capacity Heatmap (Layer2)')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('../capacityPlot_A*_reduced/vetCapacity.png')
    
    plt.figure()
    plt.imshow(hoz_capacity,cmap='hot', interpolation='nearest')
    plt.title('Horizontal Capacity Heatmap (Layer1)')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('../capacityPlot_A*_reduced/hozCapacity.png')

    plt.figure()
    # plt.plot(utilization_frequency[:,0],utilization_frequency[:,1])
    plt.bar(utilization_frequency[:,0],utilization_frequency[:,1])
    plt.plot(utilization_frequency[:,0],utilization_frequency[:,1],'r-')
    plt.title('Edge Utilization Histogram')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('../capacityPlot_A*_reduced/edgeHist.png')
    







