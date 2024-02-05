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

seed=132
np.random.seed(seed)

n_neurons = 10
n_timelags = 2
refractory_effect = 2
n_obs = 7

# set up summary and full time graph
summary_graph = nx.erdos_renyi_graph(n=n_neurons, p=0.25, directed=True, seed=196)

observed_nodes = np.sort(np.random.choice(n_neurons, size=n_obs, replace=False))
latent_nodes = [i for i in summary_graph.nodes() if i not in observed_nodes]

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
fig, ax = plt.subplots(1, 2, figsize=(n_neurons, n_neurons // 2))
nx.draw_networkx(summary_graph, arrows=True, 
                 ax=ax[0], with_labels=True, 
                 node_size=400, alpha=1, node_color=node_color,
                pos=nx.circular_layout(summary_graph))
ax[0].set_title("Summary graph with latents")
nx.draw_networkx(fulltime_graph, pos=pos, labels=time_label, node_size=400,ax=ax[1], node_color=fulltime_node_color,alpha=1)
ax[1].set_title("Full time graph with latents")
plt.tight_layout()
plt.savefig(
    f'/Users/vildeung/Documents/Masteroppgave/code/causal_discovery/causal_discovery/figures/timeseries_graph.pdf'
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
    f'/Users/vildeung/Documents/Masteroppgave/code/causal_discovery/causal_discovery/figures/MAG.pdf'
    )

# learn pag with oracle 
dag, _, _ = create_fulltime_graph_tetrad(summary_graph, n_timelags=n_timelags, latent_nodes=latent_nodes, refractory_effect=refractory_effect)

fci = ts.Fci(ts.test.MsepTest(dag))
fci.setVerbose(False)
kn = timeseries_knowledge(n_neurons, n_timelags=n_timelags, refractory_effect=refractory_effect)
fci.setKnowledge(kn)
pag = fci.search()

summary_edges = get_hypersummary(pag, n_neurons)
print(summary_edges)

intervention_nodes, targets = get_interventions(summary_edges, n_neurons) # neurons with undecided marks and their resp. targets
count_interventions = 0
for i in range(len(intervention_nodes)):
    intervention_node = intervention_nodes[i]
    target_nodes = targets[intervention_node]

    # convert nodes to fulltime equivalents
    intervention_nodes_fulltime = np.arange(intervention_node*(n_timelags+1), (intervention_node+1)*(n_timelags+1))
    target_nodes_fulltime = []
    for node in target_nodes:
        target_nodes_fulltime.extend(list(np.arange(node*(n_timelags+1), (node+1)*(n_timelags+1))))

    # get the manipulated graph
    manipulated_graph = summary_graph.copy()
    manipulated_graph.remove_edges_from(summary_graph.in_edges(intervention_node))

    # now get fulltime graph under manipulation
    manipulated_fulltime_graph, _, _ = create_fulltime_graph(manipulated_graph, n_timelags=n_timelags)

    # make plot
    fulltime_node_color = ['yellow' if node in intervention_nodes_fulltime \
                            else 'red' if node in observed_nodes_fulltime \
                            else 'grey' \
                            for node in manipulated_fulltime_graph.nodes()]
    fig, ax = plt.subplots(1, 2, figsize=(n_neurons, n_neurons // 2))
    nx.draw_networkx(manipulated_graph, arrows=True, 
                    ax=ax[0], with_labels=True, 
                    node_size=400, alpha=1, node_color=node_color,
                    pos=nx.circular_layout(manipulated_graph))
    ax[0].set_title("Manipulated graph with latents")
    nx.draw_networkx(manipulated_fulltime_graph, pos=pos, labels=time_label, node_size=400,ax=ax[1], node_color=fulltime_node_color)
    ax[1].set_title("Manipulated full time graph with latents")
    plt.tight_layout()
    plt.savefig(
        f'/Users/vildeung/Documents/Masteroppgave/code/causal_discovery/causal_discovery/figures/manipulated_timeseries_graph.pdf'
        )

    for node1 in intervention_nodes_fulltime:
        for node2 in target_nodes_fulltime:
            cause_time = node1 % (n_timelags+1)
            effect_time = node2 % (n_timelags+1)
            cause_neuron = node1 // (n_timelags+1)
            effect_neuron = node2 // (n_timelags+1)

            if (node1, node2) in manipulated_fulltime_graph.edges() and effect_time > cause_time:
                print('Require', f'x{cause_neuron},{cause_time}', '-->',f'x{effect_neuron},{effect_time}')
                kn.setRequired(f'x{cause_neuron},{cause_time}', f'x{effect_neuron},{effect_time}')
            
            if (node1, node2) not in manipulated_fulltime_graph.edges() and effect_time > cause_time:
                print('Forbid', f'x{cause_neuron},{cause_time}', '-->',f'x{effect_neuron},{effect_time}')
                kn.setForbidden(f'x{cause_neuron},{cause_time}', f'x{effect_neuron},{effect_time}')

    fci.setKnowledge(kn)
    pag = fci.search()
    #print(pag)
    summary_edges = get_hypersummary(pag, n_neurons)
    print(summary_edges)

    causal_discovery_complete = True
    for edge in summary_edges:
        if summary_edges[edge].find('o') != -1:
            causal_discovery_complete = False
    count_interventions+=1
    if causal_discovery_complete:
        print(f'All edges were discovered after {count_interventions} interventions (out of {len(intervention_nodes)}).')
        break
