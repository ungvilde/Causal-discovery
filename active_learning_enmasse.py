# To run:
## be in pytetrad directory
## make sure spikeenv is active
## make sure you have the right PATH
# source ~/.bash_profile
# echo $JAVA_HOME
# conda activate spikeenv

import sys
import os
import torch
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import jpype.imports
import seaborn as sns

BASE_DIR = ".."
sys.path.append(BASE_DIR)
jpype.startJVM(classpath=[f"{BASE_DIR}/pytetrad/resources/tetrad-current.jar"])

from tetrad_helpers import *
from functions import *
from MH_functions import *

# seed = 4
# np.random.seed(seed)

p = 0.1 # prob. of edge in summary graph
n_timelags = 2
refractory_effect = n_timelags
n_obs = 6
n_hidden = 3
n_neurons = n_obs+n_hidden
observed_nodes = np.arange(n_obs)
latent_nodes = np.arange(n_obs, n_obs+n_hidden)
n_networks = 10



for network in range(n_networks):
    for method in ['random', 'entropy-byneuron']:
        observed_graph = nx.erdos_renyi_graph(n=n_obs, p=p, directed=True)
        summary_graph = nx.DiGraph()
        summary_graph.add_nodes_from(observed_nodes)
        summary_graph.add_nodes_from(latent_nodes)
        summary_graph.add_edges_from(observed_graph.edges())

        for L in latent_nodes:
            s = np.random.choice([2])
            targets = np.random.choice(observed_nodes, size=s, replace=False)
            summary_graph.add_edges_from([(L,i) for i in targets])

        #############
        # do plotting
        #############

        node_color = ['grey' if node in latent_nodes else 'red' for node in summary_graph.nodes()]
        summary_observed = nx.subgraph(summary_graph, observed_nodes)

        # visualisation of one of the networks generated
        fig, ax = plt.subplots(1, 1)

        nx.draw_networkx(summary_graph, arrows=True, 
                        ax=ax, with_labels=True, 
                        node_size=400, alpha=1, node_color=node_color,
                        pos=nx.circular_layout(summary_graph),
                        connectionstyle="arc3,rad=0.1",arrowstyle='->'
                        )

        ax.set_title("Summary graph with latents")
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()
        ##########

        pmg, num_interventions = active_learner(
            summary_graph=summary_graph, 
            observed_neurons=observed_nodes, 
            latent_neurons=latent_nodes, 
            n_timelags=n_timelags,
            method='random',
            burnin=0,
            n_samples=3_000,
            max_iter=10)

        get_pag_arrows(pmg)
        print('Model identified after', num_interventions, 'interventions.')
