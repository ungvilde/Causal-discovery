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

seed = 777
np.random.seed(seed)

n_timelags = 2
refractory_effect = n_timelags
n_obs = 4
n_hidden = 2
n_neurons = n_obs+n_hidden
# set up summary and full time graph
#hidden_graph = nx.DiGraph(n=n_hidden)

observed_nodes = np.arange(n_obs)
latent_nodes = np.arange(n_obs, n_obs+n_hidden)

observed_graph = nx.erdos_renyi_graph(n=n_obs, p=0.5, directed=True, seed=seed)
summary_graph = nx.DiGraph()
summary_graph.add_nodes_from(observed_nodes)
summary_graph.add_nodes_from(latent_nodes)
summary_graph.add_edges_from(observed_graph.edges())

for L in latent_nodes:
    a, b = np.random.choice(observed_nodes, size=2, replace=False)
    summary_graph.add_edges_from([(L,a),(L,b)])

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

fig, ax = plt.subplots(1, 2, figsize=(n_neurons, n_neurons // 2))
nx.draw_networkx(summary_graph, arrows=True, 
                 ax=ax[0], with_labels=True, 
                 node_size=400, alpha=1, node_color=node_color,
                pos=nx.circular_layout(summary_graph))
ax[0].set_title("Summary graph with latents")
nx.draw_networkx(fulltime_graph, pos=pos, labels=time_label, node_size=400, ax=ax[1], node_color=fulltime_node_color,alpha=1)
ax[1].set_title("Full time graph with latents")
plt.tight_layout()
plt.show()

# learn pag with oracle 
dag, _, _ = create_fulltime_graph_tetrad(
    summary_graph, 
    n_timelags=n_timelags, latent_nodes=latent_nodes, refractory_effect=refractory_effect)

print('Run FCI:')
fci = ts.Fci(ts.test.MsepTest(dag))
kn = td.Knowledge()
kn = timeseries_knowledge(n_neurons, n_timelags=n_timelags, refractory_effect=refractory_effect)
fci.setKnowledge(kn)
pag = fci.search()

summary_edges = get_hypersummary(pag, n_neurons)

# print('Edge | FCI ')
# for edge in summary_edges:
#     print(edge, summary_edges[edge])

A = get_adjacency_matrix_from_tetrad(pag, n_timelags)

# np.savetxt('/Users/vildeung/Documents/Masteroppgave/code/causal_discovery/causal_discovery/data/adj_mat.txt', A, fmt='%d')
# intervention_node = select_intervention_node(A, 1000, 2000)
# print('node = ',intervention_node, ', neuron = ', intervention_node // (n_timelags+1))

'''
#intervention_nodes, targets = get_interventions(summary_edges, n_neurons) # neurons with undecided marks and their resp. targets
count_interventions = 0
do_interventions=True
print('Do interventions in following sequence:', intervention_nodes)
while do_interventions:
    count_interventions+=1
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

    # evaluate if there is a causal effect between intervention node and target nodes
    for node1 in intervention_nodes_fulltime:
        for node2 in target_nodes_fulltime:
            cause_time = node1 % (n_timelags+1)
            effect_time = node2 % (n_timelags+1)
            
            cause_neuron = node1 // (n_timelags+1)
            effect_neuron = node2 // (n_timelags+1)

            cond_set = [i for i in manipulated_fulltime_graph.nodes() if i not in [node1, node2] and i in observed_nodes_fulltime]
            are_adjacent = nx.d_separated(manipulated_fulltime_graph, {node1}, {node2}, cond_set) == False
            is_ancestor = node1 in nx.ancestors(manipulated_fulltime_graph, node2)
            if are_adjacent and is_ancestor:
                # require causal effect because information will flow from node1 to node 2
                print('Require', f'x{cause_neuron},{cause_time}', '-->',f'x{effect_neuron},{effect_time}')
                kn.setRequired(f'x{cause_neuron},{cause_time}', f'x{effect_neuron},{effect_time}')
            else:
                print('Forbid', f'x{cause_neuron},{cause_time}', '-->',f'x{effect_neuron},{effect_time}')
                kn.setForbidden(f'x{cause_neuron},{cause_time}', f'x{effect_neuron},{effect_time}')

    # add intervention knowledge 
    fci.setKnowledge(kn)
    pag = fci.search()
    summary_edges = get_hypersummary(pag, n_neurons)

    # print results
    print('After ', count_interventions, f'interventions (out of {len(intervention_nodes)} suggested):')
    print('Edge | FCI | Truth')
    for edge in summary_edges:
        try:
            if summary_edges[edge] == true_summary_edges[edge]:
                print(edge, summary_edges[edge], true_summary_edges[edge])
            if summary_edges[edge] != true_summary_edges[edge]:
                print(edge, summary_edges[edge], true_summary_edges[edge])
        except KeyError:
            print(edge, summary_edges[edge])
            continue

    causal_discovery_complete = True
    for edge in summary_edges:
        if summary_edges[edge].find('o') != -1:
            causal_discovery_complete = False
'''