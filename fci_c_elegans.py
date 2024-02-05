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
from torch_geometric.utils import to_networkx

seed=123
np.random.seed(seed)

n_timelags=2
refractory_effect = 2

# import c elegans network
c_elegans_network = torch.load('/Users/vildeung/Documents/Masteroppgave/code/causal_discovery/causal_discovery/data/c_elegans_data.pt')
c_elegans_graph = to_networkx(c_elegans_network, node_attrs = ['position'])

position_dict = nx.get_node_attributes(c_elegans_graph, 'position')
sample_space = []
lower, upper = 0., 0.25 # nodes in c. elegans head 
for neuron in position_dict:
    if position_dict[neuron] > lower and position_dict[neuron] < upper and c_elegans_graph.in_degree(neuron) > 15 and c_elegans_graph.out_degree(neuron) > 15:
        sample_space.append(neuron)

print(len(sample_space))

n_neurons=8
n_obs = 5

# sample some nodes from the head to represent full graph
model_nodes = np.sort(np.random.choice(sample_space, size = n_neurons, replace = False))
summary_graph = nx.subgraph(c_elegans_graph, model_nodes)
summary_graph = nx.convert_node_labels_to_integers(summary_graph, first_label=0)

# sample a subset of full model as observable variables
observed_nodes = np.sort(np.random.choice(summary_graph.nodes(), size = n_obs, replace = False))

# visualise the c elegans model
#nx.draw_circular(summary_graph, with_labels=True)
#plt.show()

# specify the hidden variables
latent_nodes = [node for node in summary_graph.nodes() if node not in observed_nodes]
n_hidden = len(latent_nodes)

# get the full time graph of the underlying model
fulltime_graph, pos, time_label = create_fulltime_graph(summary_graph, n_timelags=n_timelags)
    
# get the indeces of the observed nodes in fulltime graph
observed_nodes_fulltime = []
for node in observed_nodes:
    for idx in time_label:
        if time_label[idx].find(f'X{node}') != -1:
            observed_nodes_fulltime.append(idx)

# visualisation of one of the networks generated
node_color = ['grey' if node in latent_nodes else 'red' for node in summary_graph.nodes()]
fulltime_node_color = ['grey' if node not in observed_nodes_fulltime else 'red' for node in fulltime_graph.nodes()]
fig, ax = plt.subplots(1, 2, figsize=(n_neurons, n_neurons//2))
nx.draw_networkx(summary_graph, arrows=True, 
                 ax=ax[0], with_labels=True, 
                 node_size=400, alpha=1, node_color=node_color,
                pos=nx.circular_layout(summary_graph))
ax[0].set_title("Summary graph with latents")
nx.draw_networkx(fulltime_graph, pos=pos, labels=time_label, node_size=400,ax=ax[1], node_color=fulltime_node_color,alpha=1)
ax[1].set_title("Full time graph with latents")
plt.tight_layout()
plt.savefig(
    f'/Users/vildeung/Documents/Masteroppgave/code/causal_discovery/causal_discovery/figures/timeseries_graph_nobs{n_obs}_nhid{n_hidden}.pdf'
    )

# get the true MAG for the observed variables
mag = get_mag_from_dag(fulltime_graph, observed_nodes_fulltime)
plt.figure(figsize=(n_neurons//1.5, n_neurons//2))
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
