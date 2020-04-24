from Gerrymandering import facefinder as ff
import requests
import json
from gerrychain import Graph, Partition
from networkx.readwrite import json_graph
import math
import matplotlib.pyplot as plt
import networkx as nx
from gerrychain.updaters import Tally, cut_edges
import numpy as np
import pickle


def bnodes_p(graph, partition):
    return [x for x in graph.nodes() if graph.node[x]["boundary_node"] == 1]


def new_base(partition):
    base = 1
    return base


def step_num(partition):
    parent = partition.parent

    if not parent:
        return 0

    return parent["step_num"] + 1


def bnodes_p(graph, partition):
    return [x for x in graph.nodes() if graph.node[x]["boundary_node"] == 1]


def b_nodes_bi(partition):
    return {x[0] for x in partition["cut_edges"]}.union({x[1] for x in partition["cut_edges"]})


def geom_wait(partition):
    return int(np.random.geometric(
        len(list(partition["b_nodes"])) / (len(partition.graph.nodes) ** (len(partition.parts)) - 1), 1)) - 1


def buildPartition(graph, mean):
    """
    The function build 2 partition based on x coordinate value
    Parameters:
        graph (Graph): The given graph represent state information
        mean (int): the value to determine partition
    Returns:
        Partition: the partitions of the graph based on mean value
    """
    assignment = {}

    # assign node into different partition based on x coordinate
    for x in graph.node():
        if graph.node[x]['C_X'] < mean:
            assignment[x] = -1
        else:
            assignment[x] = 1

    updaters = {'population': Tally('population'),
                "boundary": bnodes_p,
                'cut_edges': cut_edges,
                'step_num': step_num,
                'b_nodes': b_nodes_bi,
                'base': new_base,
                'geom': geom_wait,
                }

    grid_partition = Partition(graph, assignment=assignment, updaters=updaters)
    return grid_partition


def compute_cross_edge(graph, partition):
    """
    The function finds the edges that cross from one partition to another
    partition
    Parameters:
        graph (Graph): The given graph represent state information
        partition (Partition): the partition of the given graph
    Returns:
        list: the list of edges that cross two partitions
    """
    cross_list = []
    for n in graph.edges:
        if Partition.crosses_parts(partition, n):
            cross_list.append(n)
    return cross_list  # cut edges of partition


def viz(graph, edge_set, partition):
    """
    The function visualize the distance of each edge in the graph
    Parameters:
        graph (Graph): The given graph represent state information
        edge_set (list): the list of edges need to be highlight in graph
        partition (Partition): the partition of the given graph
    """
    values = [1 - int(x in edge_set) for x in graph.edges()]
    # node_labels = {x : x for x in graph.nodes()}
    distance_dictionary = {}
    for x in graph.nodes():
        distance_dictionary[x] = graph.node[x]['distance']

    color_dictionary = {}
    for x in graph.nodes():
        if x in partition.parts[-1]:
            color_dictionary[x] = 1
        else:
            color_dictionary[x] = 2

    node_values = [color_dictionary[x] for x in graph.nodes()]
    plt.figure()
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_color=node_values,
            edge_color=values, labels=distance_dictionary, width=4, node_size=0,
            font_size=7)
    plt.show()


def distance_from_partition(graph, boundary_edges):
    """
    The function calculates the each node's shortest distance from boundary edges
    partition
    Parameters:
        graph (Graph): The given graph represent state information
        boundary_edges(list): the list of boundary edges
    Returns:
        Graph: graph with the label of distance for each node
    """
    for node in graph.nodes():
        dist = math.inf
        for bound in boundary_edges:
            # check if the node is in the boundary of edges
            if node == bound[0] or node == bound[1]:
                dist = 0
            else:
                min_dist = min(dist, len(nx.shortest_path(graph, source=node,
                                                       target=bound[0])) - 1)
                min_dist = min(min_dist, len(nx.shortest_path(graph, source=node,
                                                       target=bound[1])) - 1)

                if min_dist < dist:
                    dist = min_dist
        graph.node[node]["distance"] = dist
    return graph


def set_up_graph(link):
    """
    The function used to convert online json file to adjacency graph
    Parameters:
        link (string): the link for online json file
    Returns:
        Graph: graph with the information
        mean: the mean value used for partition
    """
    link = "https://people.csail.mit.edu/ddeford//COUNTY/COUNTY_13.json"
    r = requests.get(link)
    data = json.loads(r.content)
    g = json_graph.adjacency_graph(data)
    graph = Graph(g)
    graph.issue_warnings()

    horizontal = []
    node_degree = []

    # find the node with degree 1 or 2 and remove it
    for node in graph.nodes():
        graph.nodes[node]["pos"] = [graph.node[node]['C_X'], graph.node[node]['C_Y']]
        horizontal.append(graph.node[node]['C_X'])
        if graph.degree(node) == 1 or graph.degree(node) == 2:
            node_degree.append(node)

    # remove node with degree 1 or 2 since it will impact the outcome of graph
    for i in node_degree:
        graph.remove_node(i)

    # calculate mean value for partition
    mean = sum(horizontal) / len(horizontal)
    return graph, mean


def metamandering(graph, mean, partition):
    """
    The function used do metamandering for given graph based on partition
    Parameters:
        graph (Graph): The given graph represent state information
        mean (int): the value to determine partition
        partition (Partition): the partition of the given graph
    """
    # the following line can use if you don't have adjacency graph yet
    # graph, mean = set_up_graph(link)
    ff.draw_with_location(graph)

    dual = ff.planar_dual(graph, False)  # used for getting dual graph
    ff.draw_with_location(dual)

    # the following line can use if you don't have partition yet
    # partition = buildPartition(graph, mean)
    cross_edges_in_graph = compute_cross_edge(graph, partition)

    dual_list = []  # dual edges cross primal edges

    for edge in dual.edges:
        if dual.edges[edge]["original_name"] in cross_edges_in_graph:
            dual_list.append(edge)

    distance_from_partition(dual, dual_list)
    viz(dual, dual_list, partition)

    special_faces = ff.special_faces(dual, 1)
    graph = ff.face_refines(graph, special_faces)

    plt.figure()
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size=0,
            width=0.5, cmap=plt.get_cmap('jet'))
    plt.show()


# unable to import as json file because graph after metamandering has set in label
# e.g. 'rotation': {76: 95, 95: 65, 65: 86, 86: 106, 106: 76} and
# 'pos': array([-83.32783883,  32.43444145])
#  {'id': frozenset({106, 4, 86})}, {'id': frozenset({106, 4, 76})}, {'id': frozenset({65, 123, 4, 86}
def save_json_file(graph):
    data = json_graph.adjacency_data(graph)
    data['graph'] = []
    bad_node = []
    for node in data['nodes']:
        # node['pos'] = list(node['pos'])
        if 'pos' in node:
            del node['pos']
        if 'rotation' in node:
            # node['rotation'] = list(node['rotation'])
            del node['rotation']

        # I guess here the frozenset and pos is the information for face, should we keep this information?
        for i in range(len(node)):
            if isinstance(list(node.values())[i], set):
                key = list(node.keys())[i]
                del node[key]
                if len(node) == 0:
                    bad_node.append(node)
            elif type(list(node.values())[i]) is frozenset:
                key = list(node.keys())[i]
                del node[key]
                if len(node) == 0:
                    bad_node.append(node)

    for i in bad_node:
        data['nodes'].remove(i)

    bad_node = []
    break_bool = False
    for adj_node in data['adjacency']:
        bad_key = []
        for node in adj_node:
            for i in range(len(node)):
                if isinstance(list(node.values())[i], set):
                    bad_key.append(node)
                elif type(list(node.values())[i]) is frozenset:
                    bad_key.append(node)
        for i in bad_key:
            adj_node.remove(i)
        for node in adj_node:
            for i in range(len(node)):
                if i == 0 and list(node.keys())[0] == 'id':
                    bad_node.append(adj_node)
                    break_bool = True
                    break
            if break_bool:
                break_bool = False
                break
    for i in bad_node:
        data['adjacency'].remove(i)

    remove_geometries(data)
    with open("some.json", "w") as f:
        json.dump(data, f)
    # Graph.to_json(graph, "graph.json")


def remove_geometries(data):
    for node in data["nodes"]:
        bad_keys = []
        for key in node:
            if hasattr(node[key], "__geo_interface__"):
                bad_keys.append(key)
        for key in bad_keys:
            del node[key]


# unable to convert list to graph
def load_json_file(file):
    g = Graph.from_json("some.json")
    for node in g.nodes():
        g.nodes[node]["pos"] = [g.node[node]['C_X'], g.node[node]['C_Y']]

    g_sample = Graph.from_json("another.json")


def save_partition(partition):
    assignment = partition.parts
    for i in assignment:
        assignment[i] = tuple(assignment[i])
    with open("partition.p", 'wb') as fp:
        pickle.dump(assignment, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_partition(file):
    with open('partition.p', 'rb') as fp:
        loaded_partition = pickle.load(fp)

# todo: take the graph, and a collection of partitions inside

# Todo: create a json file and save graph to json file, able to read partition from json file
