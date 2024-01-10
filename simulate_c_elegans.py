import pickle
import torch
import numpy as np 
import networkx as nx

from spikeometric.models import BernoulliGLM
from spikeometric.datasets import NormalGenerator, ConnectivityDataset
from spikeometric.stimulus import RegularStimulus

from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_networkx, from_networkx

from tqdm import tqdm

# load C. elegans network daya
network_data = torch.load('data/c_elegans_data.pt')

# set up the neuron model
neuron_model = BernoulliGLM(
    theta=3.,
    dt=1.,
    coupling_window=1,
    abs_ref_scale=3,
    abs_ref_strength=-100,
    rel_ref_scale=0,
    rel_ref_strength=-30,
    alpha=0.5,
    beta=0.2,
    r = 1
)

# sample observable nodes
G = to_networkx(network_data, node_attrs = ['position'])
n_neurons = network_data.num_nodes

spike_data = []
spike_data_dict = dict()
n_timesteps = 10**4

# single neuron stimulation protocol
stimulation_protocol = [[i] for i in range(n_neurons)]
stimulation_protocol_str = [str(i) for i in range(n_neurons)] + ['null']

#print(stimulation_protocol_str)

for i, intervention in tqdm(enumerate(stimulation_protocol_str), total = len(stimulation_protocol_str)):
    stimulus_mask = torch.zeros(n_neurons, dtype=torch.bool)
    
    if intervention != 'null':
        intervention_set = stimulation_protocol[i]
        stimulus_mask[intervention_set] = True
        
    neuron_model.add_stimulus(lambda t: 2*stimulus_mask)
    spikes = neuron_model.simulate(network_data, n_steps=n_timesteps, verbose=False)
    
    spike_data_dict[intervention] = spikes

with open(f'data/c_elegans_spike_data_single_node_stimuli.pickle','wb') as f:
        pickle.dump(spike_data_dict,f)  