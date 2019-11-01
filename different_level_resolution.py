
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)

from functools import partial
import pandas
import json

import random
from networkx.readwrite import json_graph
import json
import numpy as np
import requests
import geopandas
import maup



import os
import random
import json
import geopandas as gpd
import functools
import datetime
import matplotlib
# matplotlib.use('Agg')


import matplotlib.pyplot as plt

import csv
from networkx.readwrite import json_graph

from functools import partial
import networkx as nx
import numpy as np

from gerrychain import Graph
from gerrychain import MarkovChain
from gerrychain.constraints import (Validator, single_flip_contiguous,
                                    within_percent_of_ideal_population, UpperBound)
from gerrychain.proposals import propose_random_flip, propose_chunk_flip
from gerrychain.accept import always_accept
from gerrychain.updaters import Election, Tally, cut_edges
from gerrychain import GeographicPartition
from gerrychain.partition import Partition
from gerrychain.proposals import recom
from gerrychain.metrics import mean_median, efficiency_gap
from gerrychain.tree import recursive_tree_part, bipartition_tree_random, PopulatedGraph, contract_leaves_until_balanced_or_none, find_balanced_edge_cuts


def get_spanning_tree_u_w(G):
    node_set=set(G.nodes())
    x0=random.choice(tuple(node_set))
    x1=x0
    while x1==x0:
        x1=random.choice(tuple(node_set))
    node_set.remove(x1)
    tnodes ={x1}
    tedges=[]
    current=x0
    current_path=[x0]
    current_edges=[]
    while node_set != set():
        next=random.choice(list(G.neighbors(current)))
        current_edges.append((current,next))
        current = next
        current_path.append(next)

        if next in tnodes:
            for x in current_path[:-1]:
                node_set.remove(x)
                tnodes.add(x)
            for ed in current_edges:
                tedges.append(ed)
            current_edges = []
            if node_set != set():
                current=random.choice(tuple(node_set))
            current_path=[current]


        if next in current_path[:-1]:
            current_path.pop()
            current_edges.pop()
            for i in range(len(current_path)):
                if current_edges !=[]:
                    current_edges.pop()
                if current_path.pop() == next:
                    break
            if len(current_path)>0:
                current=current_path[-1]
            else:
                current=random.choice(tuple(node_set))
                current_path=[current]

    #tgraph = Graph()
    #tgraph.add_edges_from(tedges)
    return G.edge_subgraph(tedges)

def my_uu_bipartition_tree_random(
    graph,
    pop_col,
    pop_target,
    epsilon,
    node_repeats=1,
    # spanning_tree=None,
    choice=random.choice):
    populations = {node: graph.nodes[node][pop_col] for node in graph}


    #     populations = {node: graph.nodes[node]["population"] for node in graph}
    # choice = random.choice
    # pop_target = ideal_population
    # epsilon = .2
    k = 10
    pop_target = np.sum([graph.nodes[node]["population"] for node in graph])/ k
    possible_cuts = []
    #
    # if spanning_tree is None:
    #     spanning_tree = get_spanning_tree_u_w(graph)

    while len(possible_cuts) == 0:
        spanning_tree = get_spanning_tree_u_w(graph)
        h = PopulatedGraph(spanning_tree, populations, pop_target, epsilon)
        possible_cuts = find_balanced_edge_cuts(h, choice=choice)

    # TODO: pulling off the third of the graph
    #This seems to be pulling off a third of the graph ...
    #So the way to continue -- is pull off that third into one block
    #Then repeat this,
    return choice(possible_cuts).subset


def get_spanning_tree_mst(graph):
    for edge in graph.edges:
        graph.edges[edge]["weight"] = random.random()

    spanning_tree = nx.tree.maximum_spanning_tree(
        graph, algorithm="kruskal", weight="weight"
    )
    return spanning_tree

def my_mst_bipartition_tree_random(
    graph,
    pop_col,
    pop_target,
    epsilon,
    node_repeats=1,
    spanning_tree=None,
    choice=random.choice):
    populations = {node: graph.nodes[node][pop_col] for node in graph}

    possible_cuts = []
    if spanning_tree is None:
        spanning_tree = get_spanning_tree_mst(graph)

    while len(possible_cuts) == 0:
        spanning_tree = get_spanning_tree_mst(graph)
        h = PopulatedGraph(spanning_tree, populations, pop_target, epsilon)
        possible_cuts = find_balanced_edge_cuts(h, choice=choice)

    return choice(possible_cuts).subset


def fixed_endpoints(partition):
    return partition.assignment[(19, 0)] != partition.assignment[(20, 0)] and partition.assignment[(19, 39)] != \
           partition.assignment[(20, 39)]


def boundary_condition(partition):
    blist = partition["boundary"]
    o_part = partition.assignment[blist[0]]

    for x in blist:
        if partition.assignment[x] != o_part:
            return True

    return False



def annealing_cut_accept_backwards(partition):
    boundaries1 = {x[0] for x in partition["cut_edges"]}.union({x[1] for x in partition["cut_edges"]})
    boundaries2 = {x[0] for x in partition.parent["cut_edges"]}.union({x[1] for x in partition.parent["cut_edges"]})

    t = partition["step_num"]

    # if t <100000:
    #    beta = 0
    # elif t<400000:
    #    beta = (t-100000)/100000 #was 50000)/50000
    # else:
    #    beta = 3
    base = .1
    beta = 5

    bound = 1
    if partition.parent is not None:
        bound = (base ** (beta * (-len(partition["cut_edges"]) + len(partition.parent["cut_edges"])))) * (
                    len(boundaries1) / len(boundaries2))

        if not popbound(partition):
            bound = 0
        if not single_flip_contiguous(partition):
            bound = 0
            # bound = min(1, (how_many_seats_value(partition, col1="G17RATG",
        # col2="G17DATG")/how_many_seats_value(partition.parent, col1="G17RATG",
        # col2="G17DATG"))**2  ) #for some states/elections probably want to add 1 to denominator so you don't divide by zero

    return random.random() < bound


def go_nowhere(partition):
    return partition.flip(dict())


def slow_reversible_propose(partition):
    """Proposes a random boundary flip from the partition in a reversible fasion
    by selecting uniformly from the (node, flip) pairs.
    Temporary version until we make an updater for this set.
    :param partition: The current partition to propose a flip from.
    :return: a proposed next `~gerrychain.Partition`
    """

    # b_nodes = {(x[0], partition.assignment[x[1]]) for x in partition["cut_edges"]
    #           }.union({(x[1], partition.assignment[x[0]]) for x in partition["cut_edges"]})

    flip = random.choice(list(partition["b_nodes"]))

    return partition.flip({flip[0]: flip[1]})


def slow_reversible_propose_bi(partition):
    """Proposes a random boundary flip from the partition in a reversible fasion
    by selecting uniformly from the (node, flip) pairs.
    Temporary version until we make an updater for this set.
    :param partition: The current partition to propose a flip from.
    :return: a proposed next `~gerrychain.Partition`
    """

    # b_nodes = {(x[0], partition.assignment[x[1]]) for x in partition["cut_edges"]
    #           }.union({(x[1], partition.assignment[x[0]]) for x in partition["cut_edges"]})

    fnode = random.choice(list(partition["b_nodes"]))

    return partition.flip({fnode: -1 * partition.assignment[fnode]})


def geom_wait(partition):
    return int(np.random.geometric(
        len(list(partition["b_nodes"])) / (len(partition.graph.nodes) ** (len(partition.parts)) - 1), 1)) - 1


def b_nodes(partition):
    return {(x[0], partition.assignment[x[1]]) for x in partition["cut_edges"]
            }.union({(x[1], partition.assignment[x[0]]) for x in partition["cut_edges"]})


def b_nodes_bi(partition):
    return {x[0] for x in partition["cut_edges"]}.union({x[1] for x in partition["cut_edges"]})


def uniform_accept(partition):
    bound = 0
    if popbound(partition) and single_flip_contiguous(partition) and boundary_condition(partition):
        bound = 1

    return random.random() < bound


def cut_accept(partition):
    bound = 1
    if partition.parent is not None:
        bound = (partition["base"] ** (-len(partition["cut_edges"]) + len(
            partition.parent["cut_edges"])))  # *(len(boundaries1)/len(boundaries2))

    return random.random() < bound
############


# link = input()
# r = requests.get(url='https://people.csail.mit.edu/ddeford//COUNTY/COUNTY_55.json') # Wisconsin county
# r = requests.get(url='https://people.csail.mit.edu/ddeford//COUSUB/COUSUB_05.json') # Wisconsin
# r = requests.get(url='https://people.csail.mit.edu/ddeford//COUSUB/COUSUB_17.json')
# r = requests.get(url='https://people.csail.mit.edu/ddeford//COUNTY/COUNTY_06.json') # WV county
# print(json.loads(r.content))
link = input("Put graph link: ")
r = requests.get(url=link)
data = json.loads(r.content)
g = json_graph.adjacency_graph(data)

graph = Graph(g)
graph.issue_warnings()

pos = {}
for node in graph.nodes():
    pos[node] = [graph.node[node]['C_X'], graph.node[node]['C_Y'] ]

plt.show()


steps = 1000
ns = 1
m = 10
#widths = [0,.5,1,1.5,2,2.5,3]
#widths = [1,2,3]
#widths = [1.5]
widths = [0]
# chaintype = "uniform_tree"
chaintype = "tree"
p = .6
proportion = p*6

for p in [.6]:
    proportion = p * 6
    plt.figure()
    nx.draw(graph, pos, node_size = 1, width = 1, cmap=plt.get_cmap('jet'))

    y_coordinates = [graph.node[node]['C_Y'] for node in graph.nodes()]
    mean_y = np.mean(y_coordinates)

    horizontal = []
    for x in graph.nodes():
        if graph.node[x]['C_Y'] < mean_y:
            horizontal.append(x)
    vertical = []
    for x in graph.nodes():
        if graph.node[x]['C_Y'] >= mean_y:
            vertical.append(x)


    cddict = {}  # {x: 1-2*int(x[0]/gn)  for x in graph.nodes()}

    start_plans = [horizontal]
    alignment = 0
    for n in graph.nodes():
        if n in start_plans[alignment]:
            cddict[n] = 1
        else:
            cddict[n] = -1

    for edge in graph.edges():
        graph[edge[0]][edge[1]]['cut_times'] = 0

    for node in graph.nodes():
        graph.node[node]["population"] = graph.node[node]["TOTPOP"]
        graph.node[node]["part_sum"] = cddict[node]
        graph.node[node]["last_flipped"] = 0
        graph.node[node]["num_flips"] = 0


    ####CONFIGURE UPDATERS

    def new_base(partition):
        return base


    def step_num(partition):
        parent = partition.parent

        if not parent:
            return 0

        return parent["step_num"] + 1


    bnodes = [x for x in graph.nodes() if graph.node[x]["boundary_node"] == 1]


    def bnodes_p(partition):
        return [x for x in graph.nodes() if graph.node[x]["boundary_node"] == 1]


    updaters = {'population': Tally('population'),
                "boundary": bnodes_p,
                'cut_edges': cut_edges,
                'step_num': step_num,
                'b_nodes': b_nodes_bi,
                'base': new_base,
                'geom': geom_wait,
                # "Pink-Purple": Election("Pink-Purple", {"Pink":"pink","Purple":"purple"})
                }

    #########BUILD PARTITION

    grid_partition = Partition(graph, assignment=cddict, updaters=updaters)
    pop1 = .05

    base = 1
    # ADD CONSTRAINTS
    popbound = within_percent_of_ideal_population(grid_partition, pop1)
    '''
    plt.figure()
    nx.draw(graph, pos={x: x for x in graph.nodes()}, node_size=ns,
            node_shape='s', cmap='tab20')
    plt.savefig("./plots/Attractor/" + str(alignment) + "SAMPLES:" + str(steps) + "Size:" + str(m) + "WIDTH:" + str(width) + "chaintype:" +str(chaintype) +    "B" + str(int(100 * base)) + "P" + str(
        int(100 * pop1)) + "start.eps", format='eps')
    plt.close()'''

    #########Setup Proposal
    ideal_population = sum(grid_partition["population"].values()) / len(grid_partition)
    #
    # # tree_proposal = partial(recom,
    #                         pop_col="population",
    #                         pop_target=ideal_population,
    #                         epsilon=1,
    #                         node_repeats=1
    #                         )

    #######BUILD MARKOV CHAINS

    if chaintype == "tree":
        tree_proposal = partial(recom, pop_col="population", pop_target=ideal_population, epsilon=0.05,
                                node_repeats=1, method=my_mst_bipartition_tree_random)

        exp_chain = MarkovChain(tree_proposal,
                                Validator([#popbound  # ,boundary_condition
                                           ]), accept=cut_accept, initial_state=grid_partition,
                                total_steps=steps)

    if chaintype == "uniform_tree":
        #tree_proposal = partial(uniform_tree_propose)
        tree_proposal = partial(recom, pop_col="population", pop_target=ideal_population, epsilon=0.05,
                                node_repeats=1, method=my_uu_bipartition_tree_random)


        #This tree proposal is returning partitions that are balanced to within epsilon
        #But its not good enough for this validator.
        exp_chain = MarkovChain(tree_proposal,
                                Validator([#popbound  # ,boundary_condition
                                           ]), accept=cut_accept, initial_state=grid_partition,
                                total_steps=steps)

    #########Run MARKOV CHAINS

    rsw = []
    rmm = []
    reg = []
    rce = []
    rbn = []
    waits = []

    import time

    st = time.time()

    t = 0
    seats = [[],[]]
    vote_counts = [[],[]]
    old = 0
    #skip = next(exp_chain)
    #skip the first partition
    k = 0
    num_cuts_list = []
    for part in exp_chain:
        if k > 0:
            #if part.assignment == old:
            #    print("didn't change")
            print(part["population"])
            rce.append(len(part["cut_edges"])) #get rid of this, its counted in numcuts below
            waits.append(part["geom"]) #You can ignore this
            rbn.append(len(list(part["b_nodes"]))) #also ignore this


            num_cuts = len(part["cut_edges"])
            num_cuts_list.append(num_cuts)
            for edge in part["cut_edges"]:
                graph[edge[0]][edge[1]]["cut_times"] += 1
                # print(graph[edge[0]][edge[1]]["cut_times"])

            if part.flips is not None: #Probably you can ignore this.
                f = list(part.flips.keys())[0]
                graph.node[f]["part_sum"] = graph.node[f]["part_sum"] - dict(part.assignment)[f] * (
                    abs(t - graph.node[f]["last_flipped"]))
                graph.node[f]["last_flipped"] = t
                graph.node[f]["num_flips"] = graph.node[f]["num_flips"] + 1
            for i in [0, 1]:
                top = []
                bottom = []
                rural_top = 0
                urban_top = 0
                rural_bottom = 0
                urban_bottom = 0
                for n in graph.nodes():
                    if part.assignment[n] == 1:
                        #This iterates through all of the nodes of graph that are assigned to block 1
                        #You'll want to have them "vote" in some way.
                        #
                        rural_top += graph.node[n]["RVAP"]
                        urban_top += graph.node[n]["UVAP"]
                        #top.append( int(n[i] < proportion*m))
                        #Voting data here.
                    if part.assignment[n] == -1:
                        #bottom.append( int( n[i] < proportion*m))
                        rural_bottom += graph.node[n]["RVAP"]
                        urban_bottom += graph.node[n]["UVAP"]
                        #Same thing here for rural_bot and urban_bot
                        # bottom.append(0)


                top_seat = int(rural_top > urban_top)
                bottom_seat = int(rural_bottom > urban_bottom)
                #might have to worry about there being either so many urban or rural that these go always
                #one direction

                #just a hack for now ...

                #could replace with top_seat = int(rural_top  / 10 > urban)
                total_seats = top_seat + bottom_seat
                seats[i].append(total_seats)
            #old = part.assignment
        t += 1
        k += 1

    width = 0
    diagonal_bias = 0
    print("average cut size:", np.mean(num_cuts_list))
    # f = open("./plots/Attractor/" + str(alignment) + "SAMPLES:" + str(steps) + "Size:" + str(m) + "chaintype:" + str(chaintype) + "Bias:" + str(diagonal_bias) + "P" + str(
    #      int(100 * pop1)) + "proportion:" + str(p) + "edges.txt", 'a')

    means = np.mean(seats,1)
    stds = np.std(seats,1)

     # f.write( str( means[0] ) + "(" + str(stds[0]) + ")," + str( means[1] ) + "(" + str(stds[1]) + ")" + "at width:" + str(width) + '\n')

    # f.write("mean:" +  str(np.mean(seats,1)) + "var:" + str(np.var(seats,1)) + "stdev:" + str(np.std(seats,1)) +  "at width:" + str(width) + '\n' )

    # f.close()
    print(str( means[0] ) + "(" + str(stds[0]) + ")" + str( means[1] ) + "(" + str(stds[1]) + ")" )
    print(seats)

    plt.figure()
    nx.draw(graph, pos, node_color=[0 for x in graph.nodes()], node_size=1,
            edge_color=[graph[edge[0]][edge[1]]["cut_times"] for edge in graph.edges()], node_shape='s',
            cmap='magma', width=3)
    # plt.savefig("./plots/Attractor/" + str(alignment) + "SAMPLES:" + str(steps) + "Size:" + str(m) + "WIDTH:" + str(width) + "chaintype:" +str(chaintype) +  "Bias:" + str(diagonal_bias) +  "P" + str(
    #      int(100 * pop1)) + "edges.eps", format='eps')
    plt.show()
    plt.close()

    # A2 = np.zeros([6 * m, 6 * m])
    # for n in graph.nodes():
    #     #print(n[0], n[1] - 1, dict(part.assignment)[n])
    #     A2[n[0], n[1]] = dict(part.assignment)[n]
    #
    # plt.figure()
    # plt.imshow(A2, cmap = 'jet')
    # plt.axis('off')
    # # plt.savefig("./plots/Attractor/" + "Size:" + str(m) + "WIDTH:" + str(width) + "chaintype:" +str(chaintype) + "Bias:" + str(diagonal_bias) + "P" + str(
    # #     int(100 * pop1)) + "sample_partition.eps", format='eps')
    # plt.close()
    #
    # plt.figure()
    # plt.hist(seats)