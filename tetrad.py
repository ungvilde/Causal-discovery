# to Run: 
# $ source ~/.bash_profile

import sys
import torch
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import jpype.imports

BASE_DIR = ".."
sys.path.append(BASE_DIR)
jpype.startJVM(classpath=[f"{BASE_DIR}/pytetrad/resources/tetrad-current.jar"])

from tetrad_helpers import *
from torch_geometric.utils import to_networkx

seed=123
np.random.seed(seed)

# model parameters
n_neurons=6
n_hidden=2
n_timelags=2
prob_of_edge = 0.1

# set up summary and full time graph
G_summary = nx.erdos_renyi_graph(n=n_neurons, p=prob_of_edge, directed=True, seed=123)
latent_nodes = np.random.choice(n_neurons, size=n_hidden, replace=False)

node_color = ['grey' if node in latent_nodes else 'red' for node in G_summary.nodes()]

plt.figure()
nx.draw_circular(G_summary, with_labels=True, node_color=node_color)
plt.savefig('/Users/vildeung/Documents/Masteroppgave/code/causal_discovery/causal_discovery/figures/summary_graph.pdf')
plt.close()

dag, _, _ = create_fulltime_graph_tetrad(G_summary, n_timelags=n_timelags, latent_nodes=latent_nodes)
print(dag)

# learn pag with oracle 
fci = ts.Fci(ts.test.MsepTest(dag))
kn = timeseries_knowledge(n_neurons, n_timelags=n_timelags)
fci.setKnowledge(kn)
pag = fci.search()

print(pag)

summary_edges = {(i,j) : set() for i in range(n_neurons) for j in range(n_neurons) if j != i}

nodes_obs = pag.getNodes()
nodes_str = pag.getNodeNames()
for i, node1 in enumerate(nodes_obs):
    for j, node2 in enumerate(nodes_obs):
        if pag.isAdjacentTo(node1, node2):
            edge_str = str(pag.getEdge(node1, node2))
            neuron_id = nodes_str[i][1:str(nodes_str[i]).find(',')]
            target_id = nodes_str[j][1:str(nodes_str[j]).find(',')]
            if neuron_id != target_id and edge_str.find(f'x{neuron_id}') < edge_str.find(f'x{target_id}'):
                #edge_str = str(pag.getEdge(node1, node2))
                edge_str = edge_str[(edge_str.find(' ')+1):edge_str.rfind(' ')]
                summary_edges[(int(neuron_id), int(target_id))].add(edge_str)

summary_edges = { edge: edge_type for edge, edge_type in summary_edges.items() if len(edge_type) > 0}
print(summary_edges)

# max_key = max(d, key= lambda x: len(d[x]))


'''
# now do interventions
intervened_node = 6
print('INTERVENE ON NODE ', intervened_node)

manipulated_edges = list(G_summary.in_edges(intervened_node))
G_manipulated = G_summary.copy()
G_manipulated.remove_edges_from(manipulated_edges)

plt.figure()
nx.draw_circular(G_manipulated, with_labels=True, node_color=node_color)
plt.savefig('/Users/vildeung/Documents/Masteroppgave/code/causal_discovery/causal_discovery/figures/manipulated_graph.pdf')
plt.close()

nx.draw_circular(G_manipulated, with_labels=True, node_color=node_color)
dag, _, _ = create_fulltime_graph_tetrad(G_manipulated, n_timelags=n_timelags, latent_nodes=latent_nodes)

fci = ts.Fci(ts.test.MsepTest(dag))
kn_manipulated = interventional_knowledge(kn, intervened_node, n_neurons, n_timelags)
fci.setKnowledge(kn)

pag = fci.search()

print(pag)

# algorithm

# learn PMG from obs. data and bk
# intervene on node A where A has the most edges of type *-o
# if A has the same dependence on B in interventional setting, then A *-> B
# else A *-- B
'''