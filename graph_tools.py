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
import matplotlib.pyplot as plt

#link = input("Put graph link: ")

def graph_from_url(link):
    r = requests.get(url=link)
    data = json.loads(r.content)
    g = json_graph.adjacency_graph(data)
    graph = Graph(g)
    graph.issue_warnings()
    pos = {}
    for node in graph.nodes():
        pos[node] = [graph.node[node]['C_X'], graph.node[node]['C_Y'] ]
    return graph


