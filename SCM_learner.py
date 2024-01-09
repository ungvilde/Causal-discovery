import torch
import numpy as np 
import statsmodels.api as sm
import networkx as nx

from torch_geometric.data import Data

from tqdm import tqdm


def SCM_learner(
    spike_data, 
    node_list, 
    stimulation_protocol, 
    alpha=0.05):
    
    """
    spike_data: dictionary of observed neuron data sets, where the key is the intervention used. 'null' is no intervetion. 
    '1' means node 1 was intervened on, '123' means node 1, 2 and 3 were intervened on, etc.
    
    node_list: a list contining node names of the observed variables
    
    stimulation_protocol: list of indeces for intervened neurons at each intervention.
    
    alpha = significance level, default is alpha=0.05
    """
    
    SCM_learned = nx.DiGraph()
    SCM_learned.add_nodes_from(np.sort(node_list))

    # observational data
    spikes = spike_data['null']
    
    #stimulation_protocol_str = [''.join(str(i) for i in intervention_set) for intervention_set in stimulation_protocol]
        
    # loop through nodes in network, to test what other notes can explain spiking using a linear model
    for idx, neuron_id in enumerate(node_list):
        
        target_spikes = spikes[idx].numpy()
        
        source_spikes = np.delete(torch.roll(spikes, 1), idx, axis=0) # effect from 1 time step before
        #source_spikes2 = np.delete(torch.roll(spikes, 2), idx, axis=0) # effect from 2 time step before
        autoregressive_feature1 = torch.roll(spikes[idx], 1).numpy() # 1 time step history effects
        autoregressive_feature2 = torch.roll(spikes[idx], 2).numpy() # 2 time step history effects
        autoregressive_feature3 = torch.roll(spikes[idx], 3).numpy() # 3 time step history effects

        # create design matrix
        X = np.vstack((source_spikes, autoregressive_feature1, autoregressive_feature2))

        # use a linear regression model to assess if there is a significantly non-zero association between source and target neuron
        linear_model = sm.OLS(target_spikes.T, sm.add_constant(X.T))
        res = linear_model.fit()

        source_nodes = np.delete(node_list, idx) # remove target neuron
        p_values = res.pvalues[1:len(node_list)] # p-values of t-test of effect from source to targat neuron (first time step coefficient)
        
        #print(p_values)
        
        for k, p in enumerate(p_values):
            if p < alpha:
                SCM_learned.add_edge(source_nodes[k], neuron_id)
    
    # Adjacency matrix based on observational data
    A_learned = nx.adjacency_matrix(SCM_learned).todense() 
    SCM_observational = SCM_learned.copy()

    # loop through each intervention set to remove confounding effects
    for intervention_set in stimulation_protocol:
        intervention_set_str = ''.join(str(x) for x in intervention_set)        
        spikes = spike_data[intervention_set_str] # get data set where the given intervention occurred

        for intervened_neuron in intervention_set:            
            intervened_neuron_idx = node_list.index(intervened_neuron)

            # get indeces of neurons that intervened node is correlated with
            target_idx = np.where(A_learned[intervened_neuron_idx, :] == 1) 
            #target_neurons = node_list[target_idx[0]]
            target_neurons = [v for _, v in SCM_observational.out_edges(intervened_neuron)]

            if len(target_neurons) == 0:
                continue

            else:
                # loop through the neurons that the intervened node is observationally correlated with
                for target_neuron in target_neurons:
                    target_neuron_idx = node_list.index(target_neuron)

                    sources_idx = np.where(A_learned[:, target_neuron_idx] == 1) # possible explanations for target neuron
                    #source_neurons = index_obs[sources_idx[0]]
                    source_neurons = [u for u, _ in SCM_observational.in_edges(target_neuron)]

                    intervened_idx = np.where(source_neurons == intervened_neuron)[0]
                    target_spikes = spikes[target_neuron_idx].numpy()

                    source_spikes = torch.roll(spikes[sources_idx[0]], 1).numpy() # effect from 1 time step before
                    #source_spikes2 = torch.roll(spikes[sources_idx[0]], 2).numpy() # effect from 2 time step before
                    autoregressive_feature1 = torch.roll(spikes[target_neuron_idx], 1).numpy() # 1 time step history effects
                    autoregressive_feature2 = torch.roll(spikes[target_neuron_idx], 2).numpy() # 2 time step history effects

                    # create design matrix
                    X = np.vstack((source_spikes, autoregressive_feature1, autoregressive_feature2))

                    # use a linear regression model to assess if there is a significantly non-zero association between source and target neuron
                    linear_model = sm.OLS(target_spikes.T, sm.add_constant(X.T))
                    res = linear_model.fit()
                    #print(res.pvalues)
                    p_values = res.pvalues[intervened_idx+1] # p-values of t-test of effect from source to targat neuron (first time step coefficient)
                    #
                    #print('p_values = ',p_values)
                    #print(res.pvalues)
                    if np.any(p_values > alpha):
                        #print(f'removed edge ({intervened_neuron}, {target_neuron})')
                        SCM_learned.remove_edge(intervened_neuron, target_neuron)

    return SCM_learned