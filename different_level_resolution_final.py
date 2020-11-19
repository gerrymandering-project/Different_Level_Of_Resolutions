import requests
import random
import json
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph
from functools import partial
import networkx as nx
import numpy as np
from gerrychain import Graph
from gerrychain import MarkovChain
from gerrychain.constraints import (Validator, single_flip_contiguous,
                                    within_percent_of_ideal_population)
from gerrychain.updaters import Tally, cut_edges
from gerrychain.partition import Partition
from gerrychain.proposals import recom
from gerrychain.tree import PopulatedGraph, find_balanced_edge_cuts
import time


def get_spanning_tree_u_w(G):
    node_set = set(G.nodes())
    x0 = random.choice(tuple(node_set))
    x1 = x0
    while x1 == x0:
        x1 = random.choice(tuple(node_set))
    node_set.remove(x1)
    tnodes = {x1}
    tedges = []
    current = x0
    current_path = [x0]
    current_edges = []
    while node_set != set():
        next = random.choice(list(G.neighbors(current)))
        current_edges.append((current, next))
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
                current = random.choice(tuple(node_set))
            current_path = [current]

        if next in current_path[:-1]:
            current_path.pop()
            current_edges.pop()
            for i in range(len(current_path)):
                if current_edges != []:
                    current_edges.pop()
                if current_path.pop() == next:
                    break
            if len(current_path) > 0:
                current = current_path[-1]
            else:
                current = random.choice(tuple(node_set))
                current_path = [current]
    return G.edge_subgraph(tedges)


def my_uu_bipartition_tree_random(
    graph,
    pop_col,
    pop_target,
    epsilon,
    node_repeats=1,
    spanning_tree=None,
    choice = random.choice):
    populations = {node: graph.nodes[node][pop_col] for node in graph}

    # pop_target = ideal_population
    k = 2
    pop_target = np.sum([graph.nodes[node]["population"] for node in graph])/ k
    possible_cuts = []

    while len(possible_cuts) == 0:
        spanning_tree = get_spanning_tree_u_w(graph)
        h = PopulatedGraph(spanning_tree, populations, pop_target, epsilon)
        possible_cuts = find_balanced_edge_cuts(h, choice=choice)

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


# helper method used for find k_partition_tree and return graph that remove edges
def recursive_my_mst_k_partition_tree_random(
        test_graph,
        pop_col,
        pop_target,
        epsilon,
        num_blocks,
        our_blocks,
        spanning_tree=None,
        choice=random.choice,):

    populations = {node: test_graph.nodes[node][pop_col] for node in test_graph}
    # pop_target = np.sum([test_graph.nodes[node]["population"] for node in test_graph]) / num_blocks
    possible_cuts = []

    while len(possible_cuts) == 0:
        spanning_tree = get_spanning_tree_mst(test_graph)
        h = PopulatedGraph(spanning_tree, populations, pop_target, epsilon)
        possible_cuts = find_balanced_edge_cuts(h, choice=choice)

    # remove possible cuts to get new partition
    spanning_tree.remove_edge(possible_cuts[0].edge[0], possible_cuts[0].edge[1])
    unfrozen_tree = nx.Graph(spanning_tree)
    comps = nx.connected_components(unfrozen_tree)

    list_comps = list(comps)
    population_block = 0
    list_blocks = []

    # iterate through all new blocks to find out which block's population close to ideal population, add to block list
    for x in list_comps:
        for y in x:
            population_block += populations[y]
        list_blocks.append(population_block)
        population_block = 0

    index = min(range(len(list_blocks)), key=lambda x: abs(list_blocks[x]-pop_target))
    our_blocks.append(list_comps[index])
    sub_graph = test_graph.copy()

    for x in list_comps[index]:
        sub_graph.remove_node(x)
    return sub_graph


#The purpose of this is to create a starting partition -- this will return a list of blocks, and
#you'll need to turn that into a Gerrychain assignment object, as a place to start hte recomb chain.
def my_mst_kpartition_tree_random(
        graph,
        pop_col,
        pop_target,
        epsilon,
        num_blocks,
        node_repeats=1,
        spanning_tree=None,
        choice=random.choice,):

    our_blocks = []
    repeat_time = num_blocks - 1
    sub_graph = graph.copy()

    pop_target = np.sum([graph.nodes[node]["population"] for node in graph]) / num_blocks

    # repeated call for cut spanning tree into blocks with ideal population we want
    for x in range(repeat_time):
        sub_graph = recursive_my_mst_k_partition_tree_random(sub_graph, pop_col, pop_target, epsilon, num_blocks,
                                                             our_blocks, spanning_tree=None, choice=random.choice,)
    last_spanning_tree = get_spanning_tree_mst(sub_graph)
    comps = nx.connected_components(last_spanning_tree)
    list_comps = list(comps)
    our_blocks.append(list_comps[0])

    return our_blocks


def boundary_condition(partition):
    blist = partition["boundary"]
    o_part = partition.assignment[blist[0]]

    for x in blist:
        if partition.assignment[x] != o_part:
            return True

    return False


# def geom_wait(partition):
#     return int(np.random.geometric(
#         len(list(partition["b_nodes"])) / (len(partition.graph.nodes) ** (len(partition.parts)) - 1), 1)) - 1
#
#
# def b_nodes(partition):
#     return {(x[0], partition.assignment[x[1]]) for x in partition["cut_edges"]
#             }.union({(x[1], partition.assignment[x[0]]) for x in partition["cut_edges"]})
#
#
# def b_nodes_bi(partition):
#     return {x[0] for x in partition["cut_edges"]}.union({x[1] for x in partition["cut_edges"]})


def cut_accept(partition):
    bound = 1
    if partition.parent is not None:
        bound = (partition["base"] ** (-len(partition["cut_edges"]) + len(
            partition.parent["cut_edges"])))  # *(len(boundaries1)/len(boundaries2))

    return random.random() < bound

# get the graph from online json file
def graph_from_url():
    # link = input("Put graph link: ")
    link = "https://people.csail.mit.edu/ddeford//COUNTY/COUNTY_37.json"
    r = requests.get(link)
    data = json.loads(r.content)
    g = json_graph.adjacency_graph(data)
    graph = Graph(g)
    graph.issue_warnings()

    return graph


def visualize_partition(graph, pos, partition_dict):
    nx.draw(graph, pos, node_color=[partition_dict[x] for x in graph.nodes()])


def set_up_graph():
    graph = graph_from_url()
    for node in graph.nodes():
        graph.nodes[node]['pos'] = (graph.node[node]['C_X'], graph.node[node]['C_Y'])
        # pos[node] = [graph.node[node]['C_X'], graph.node[node]['C_Y']]

    # original graph with no partition divide
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size=1, width=1, cmap=plt.get_cmap('jet'))
    plt.show()

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
    return graph, horizontal


def set_up_markov_chain(graph, horizontal):
    steps = 1000
    ns = 1
    m = 10
    widths = [0]
    chaintype = "tree"
    p = .6
    proportion = p*6
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
    return steps, chaintype

####CONFIGURE UPDATERS

def new_base(partition):
    return base


def step_num(partition):
    parent = partition.parent

    if not parent:
        return 0

    return parent["step_num"] + 1


# bnodes = [x for x in graph.nodes() if graph.node[x]["boundary_node"] == 1]

#
# def bnodes_p(partition):
#     return [x for x in graph.nodes() if graph.node[x]["boundary_node"] == 1]


#########BUILD PARTITION
# building partition dicitionary (assignment)
# [ 1,2,3], [4,5,6] ... the dictionary will : {1 : 0, 2 : 0, ..., 5 : 1, 6 : 1 }$...
# It is the partition dictionary for k-partition
pop1 = .05
base = 1

# building partition dictionary with k number of blocks
def build_partition(graph):

    partition_dict = {}
    partition_block = my_mst_kpartition_tree_random(graph, pop_col="population", pop_target=0, epsilon=0.5,
                                                num_blocks=8, node_repeats=1, spanning_tree=None,
                                                choice=random.choice)
    for n in graph.nodes:
        for x in range(len(partition_block)):
            if n in partition_block[x]:
                partition_dict[n] = x

    updaters = {'population': Tally('population'),
            'cut_edges': cut_edges,
            'step_num': step_num,
            'base': new_base,
            }
    grid_partition = Partition(graph, assignment=partition_dict, updaters=updaters)
    return grid_partition, partition_dict


#######BUILD MARKOV CHAINS
def build_markov_chain(steps, chaintype, ideal_population, grid_partition):
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
    return exp_chain


#########Run MARKOV CHAINS
def run_markov_chain(exp_chain, partition_dict, graph):
    rsw = []
    rmm = []
    reg = []
    rce = []
    rbn = []
    waits = []
    st = time.time()

    num_blocks = len(set(partition_dict.values()))

    seats = [[] for y in range(num_blocks)]
    t = 0
    k = 0
    old = 0
    num_cuts_list = []
    seats_won_table = []

# '''
# seats is organized as follows:
#
# seats[k][i] is the outcome on the ith map of district k
#
#
# '''
    for part in exp_chain:
        seats_won = 0
        if k > 0:
            print(part["population"])

            num_cuts = len(part["cut_edges"])
            num_cuts_list.append(num_cuts)
            for edge in part["cut_edges"]:
                graph[edge[0]][edge[1]]["cut_times"] += 1

            if part.flips is not None: # Probably you can ignore this.
                f = list(part.flips.keys())[0]
                graph.node[f]["part_sum"] = graph.node[f]["part_sum"] - dict(part.assignment)[f] * (
                    abs(t - graph.node[f]["last_flipped"]))
                graph.node[f]["last_flipped"] = t
                graph.node[f]["num_flips"] = graph.node[f]["num_flips"] + 1

            # calculate the voting data for rural and urban seats
            for i in range(num_blocks):
                rural_pop = 0
                urban_pop = 0
                for n in graph.nodes:
                    if part.assignment[n] == i:
                        rural_pop += graph.node[n]["RVAP"]
                        urban_pop += graph.node[n]["UVAP"]
                total_seats = int(rural_pop > urban_pop)
                seats_won += total_seats
                seats_won_table.append(seats_won)
                seats[i].append(total_seats)

        t += 1
        k += 1
    return seats, seats_won_table


# for collecting voting data of rural and urban seats
def collect_voting_data(seats):
    rural_seats = 0
    urban_seats = 0
    rural_seats_list = []
    urban_seats_list = []
    for map_index in range(len(seats[0])):
        for n in range(len(seats)):
            rural_seats += seats[n][map_index]
            urban_seats += 1 - seats[n][map_index]
        rural_seats_list.append(rural_seats)
        urban_seats_list.append(urban_seats)
        urban_seats = 0
        rural_seats = 0


def main():
    graph, horizontal = set_up_graph()
    steps, chaintype = set_up_markov_chain(graph, horizontal)
    grid_partition, partition_dict = build_partition(graph)
    ideal_population = sum(grid_partition["population"].values()) / len(grid_partition)
    exp_chain = build_markov_chain(steps, chaintype, ideal_population, grid_partition)
    seats, seats_won_table = run_markov_chain(exp_chain, partition_dict, graph)
    collect_voting_data(seats)
    # plt.hist(rural_seats_list)
    plt.hist(seats_won_table)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_color=[0 for x in graph.nodes()], node_size=0,
            edge_color=[graph[edge[0]][edge[1]]["cut_times"] for edge in graph.edges()], node_shape='s',
            cmap='magma', width=0.5)
    plt.savefig("./plots/" + "NC_COUNTY.svg", format='svg')
    # plt.savefig("./plots/Attractor/" + str(alignment) + "SAMPLES:" + str(steps) + "Size:" + str(m) + "WIDTH:" + str(width) + "chaintype:" +str(chaintype) +  "Bias:" + str(diagonal_bias) +  "P" + str(
    #      int(100 * pop1)) + "edges.eps", format='eps')
    plt.show()


main()
# make a histogram to record rural_seats
# plt.hist(rural_seats_list)
# plt.show()
# width = 0
# diagonal_bias = 0
# print("average cut size:", np.mean(num_cuts_list))
# # f = open("./plots/Attractor/" + str(alignment) + "SAMPLES:" + str(steps) + "Size:" + str(m) + "chaintype:" + str(chaintype) + "Bias:" + str(diagonal_bias) + "P" + str(
# #      int(100 * pop1)) + "proportion:" + str(p) + "edges.txt", 'a')
#
# means = np.mean(seats,1)
# stds = np.std(seats,1)
#
#  # f.write( str( means[0] ) + "(" + str(stds[0]) + ")," + str( means[1] ) + "(" + str(stds[1]) + ")" + "at width:" + str(width) + '\n')
#
# # f.write("mean:" +  str(np.mean(seats,1)) + "var:" + str(np.var(seats,1)) + "stdev:" + str(np.std(seats,1)) +  "at width:" + str(width) + '\n' )
#
# # f.close()
# print(str( means[0] ) + "(" + str(stds[0]) + ")" + str( means[1] ) + "(" + str(stds[1]) + ")" )
# print(seats)
#
# plt.figure()
# nx.draw(graph, pos, node_color=[0 for x in graph.nodes()], node_size=.5,
#         edge_color=[graph[edge[0]][edge[1]]["cut_times"] for edge in graph.edges()], node_shape='s',
#         cmap='magma', width=2)
# # plt.savefig("./plots/Attractor/" + str(alignment) + "SAMPLES:" + str(steps) + "Size:" + str(m) + "WIDTH:" + str(width) + "chaintype:" +str(chaintype) +  "Bias:" + str(diagonal_bias) +  "P" + str(
# #      int(100 * pop1)) + "edges.eps", format='eps')
# plt.show()
# plt.close()
#
# # A2 = np.zeros([6 * m, 6 * m])
# # for n in graph.nodes():
# #     #print(n[0], n[1] - 1, dict(part.assignment)[n])
# #     A2[n[0], n[1]] = dict(part.assignment)[n]
# #
# # plt.figure()
# # plt.imshow(A2, cmap = 'jet')
# # plt.axis('off')
# # # plt.savefig("./plots/Attractor/" + "Size:" + str(m) + "WIDTH:" + str(width) + "chaintype:" +str(chaintype) + "Bias:" + str(diagonal_bias) + "P" + str(
# # #     int(100 * pop1)) + "sample_partition.eps", format='eps')
# # plt.close()
# #
# # plt.figure()
# # plt.hist(seats)


