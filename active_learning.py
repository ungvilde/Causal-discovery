# To run:
## be in pytetrad directory
## make sure spikeenv is active
## make sure you have the right PATH
# source ~/.bash_profile
# echo $JAVA_HOME

import sys
import os
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
from MH_functions import *

seed = 900
np.random.seed(seed)

p = 0.1 # prob. of edge in summary graph
n_timelags = 1
refractory_effect = n_timelags
n_obs = 5
n_hidden = 2
n_neurons = n_obs+n_hidden
observed_nodes = np.arange(n_obs)
latent_nodes = np.arange(n_obs, n_obs+n_hidden)

observed_graph = nx.erdos_renyi_graph(n=n_obs, p=p, directed=True, seed=seed)
summary_graph = nx.DiGraph()
summary_graph.add_nodes_from(observed_nodes)
summary_graph.add_nodes_from(latent_nodes)
summary_graph.add_edges_from(observed_graph.edges())

for L in latent_nodes:
    a, b = np.random.choice(observed_nodes, size=2, replace=False)
    summary_graph.add_edges_from([(L,a),(L,b)])

#############
# do plotting
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
fig, ax = plt.subplots(1, 2, figsize=(n_neurons, n_neurons // 2))
nx.draw_networkx(summary_graph, arrows=True, 
                 ax=ax[0], with_labels=True, 
                 node_size=400, alpha=1, node_color=node_color,
                pos=nx.circular_layout(summary_graph))
ax[0].set_title("Summary graph with latents")
nx.draw_networkx(fulltime_graph, pos=pos, labels=time_label, node_size=400, ax=ax[1], node_color=fulltime_node_color,alpha=1)
ax[1].set_title("Full time graph with latents")
plt.tight_layout()
plt.savefig('/Users/vildeung/Documents/Masteroppgave/code/causal_discovery/causal_discovery/figures/timeseries_graph.pdf')
plt.close()
##########

pmg, num_interventions = active_learner(
    summary_graph=summary_graph, 
    observed_neurons=observed_nodes, 
    latent_neurons=latent_nodes, 
    n_timelags=n_timelags,
    method='entropy',
    burnin=50,
    n_samples=100,
    max_iter=len(observed_nodes))

#print(pmg)
print('Model identified after', num_interventions, 'interventions.')
