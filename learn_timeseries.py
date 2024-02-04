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
from functions import *
#from torch_geometric.utils import to_networkx

seed=123
np.random.seed(seed)

n_neurons=5
n_timelags=2
refractory_effect = 2

# set up summary and full time graph
summary_graph = nx.DiGraph()
summary_graph.add_nodes_from(np.arange(n_neurons))
summary_graph.add_edges_from([(0,1), (0,2), (1,3), (4, 3)])

latent_nodes = [0]
observed_nodes = np.sort([i for i in summary_graph.nodes() if i not in latent_nodes])

n_hidden = len(latent_nodes)
n_obs = len(observed_nodes)

fulltime_graph, pos, time_label = create_fulltime_graph(summary_graph, n_timelags=n_timelags)
    
node_color = ['grey' if node in latent_nodes else 'red' for node in summary_graph.nodes()]

# get the observed nodes in fulltime graph
observed_nodes_fulltime = []
for node in observed_nodes:
    for idx in time_label:
        if time_label[idx].find(f'X{node}') != -1:
            observed_nodes_fulltime.append(idx)

summary_observed = nx.subgraph(summary_graph, observed_nodes)
fulltime_graph_obs = nx.subgraph(fulltime_graph, observed_nodes_fulltime)
fulltime_node_color = ['grey' if node not in observed_nodes_fulltime else 'red' for node in fulltime_graph.nodes()]

# visualisation of one of the networks generated
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
nx.draw_networkx(summary_graph, arrows=True, 
                 ax=ax[0], with_labels=True, 
                 node_size=400, alpha=1, node_color=node_color,
                pos=nx.circular_layout(summary_graph))
ax[0].set_title("Summary graph with latents")
nx.draw_networkx(fulltime_graph, pos=pos, labels=time_label, node_size=400,ax=ax[1], node_color=fulltime_node_color,alpha=0.5)
ax[1].set_title("Full time graph with latents")
plt.tight_layout()
plt.savefig(
    f'/Users/vildeung/Documents/Masteroppgave/code/causal_discovery/causal_discovery/figures/timeseries_graph_nobs{n_obs}_nhid{n_hidden}.pdf'
    )

# get the true MAG for the observed variables
mag = get_mag_from_dag(fulltime_graph, observed_nodes_fulltime)
fig, ax = plt.subplots(1, 2, figsize=(5, 3))
plt.figure(figsize=(5,4))
nx.draw_networkx(mag, 
                 pos={k: pos[k] for k in observed_nodes_fulltime},
                 labels={k: time_label[k] for k in observed_nodes_fulltime},
                 node_size=400, 
                 node_color='red',
                )
plt.title('Maximal ancestral graph')
plt.tight_layout()
plt.savefig(
    f'/Users/vildeung/Documents/Masteroppgave/code/causal_discovery/causal_discovery/figures/MAG_nobs{n_obs}_nhid{n_hidden}.pdf'
    )

# learn pag with oracle 
dag, _, _ = create_fulltime_graph_tetrad(summary_graph, n_timelags=n_timelags, latent_nodes=latent_nodes, refractory_effect=refractory_effect)

fci = ts.Fci(ts.test.MsepTest(dag))
kn = timeseries_knowledge(n_neurons, n_timelags=n_timelags, refractory_effect=refractory_effect)
fci.setKnowledge(kn)
pag = fci.search()

print(pag)
summary_edges = get_hypersummary(pag, n_neurons)
print(summary_edges)
