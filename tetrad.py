# to Run: 
# git clone https://github.com/cmu-phil/py-tetrad/
# cd py-tetrad/pytetrad
# source ~/.bash_profile

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
n_neurons=8
n_hidden=4
n_timelags=2
prob_of_edge = 0.1

# set up summary and full time graph
G_summary = nx.erdos_renyi_graph(n=n_neurons, p=prob_of_edge, directed=True, seed=seed)
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

summary_edges = get_hypersummary(pag, n_neurons)
intervened_node = get_intervened_node(summary_edges, n_neurons)

intervened_node = 6

print(summary_edges)

'''
# now do interventions
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

summary_edges = get_hypersummary(pag, n_neurons)
print(summary_edges)

# algorithm

# learn PMG from obs. data and bk
# intervene on node A where A has the most edges of type *-o
# if A has the same dependence on B in interventional setting, then A *-> B
# else A *-- B

# note: need to include unconnected nodes in summary result
# should 

'''