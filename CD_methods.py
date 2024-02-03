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
    node_list = list(node_list)
    SCM_learned = nx.DiGraph()
    SCM_learned.add_nodes_from(np.sort(node_list))

    # observational data
    spikes = spike_data['null']
            
    for idx, neuron_id in enumerate(node_list):
        
        target_spikes = spikes[idx].numpy()
        
        source_spikes = np.delete(torch.roll(spikes, 1), idx, axis=0) # effect from 1 time step before
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
                
        for k, p in enumerate(p_values):
            if p < alpha:
                SCM_learned.add_edge(source_nodes[k], neuron_id)
    
    SCM_observational = SCM_learned.copy()

    # loop through each intervention set to remove confounding effects
    for intervention_set in tqdm(stimulation_protocol, total=len(stimulation_protocol)):
        intervention_set_str = ''.join(str(x) for x in intervention_set)        
        spikes = spike_data[intervention_set_str] # get data set where the given intervention occurred

        for intervened_neuron in intervention_set:            

            # get indeces of neurons that intervened node is correlated with
            target_neurons = [v for _, v in SCM_observational.out_edges(intervened_neuron)]
            # TODO: the target neurons that are also in the intervention set need not be analysed further
            # because we know their activity will be independent of other nodes
            
            if len(target_neurons) == 0:
                continue

            else:

                # loop through the neurons that the intervened node is observationally correlated with
                for target_neuron in target_neurons:
                    target_neuron_idx = node_list.index(target_neuron)

                    # include all possible explanations for target neuron in the model
                    source_neurons = [u for u, _ in SCM_observational.in_edges(target_neuron)]
                    source_neurons_idx = [node_list.index(i) for i in source_neurons]                    
                    
                    target_spikes = spikes[target_neuron_idx].numpy()

                    source_spikes = torch.roll(spikes[source_neurons_idx], 1).numpy() # effect from 1 time step before
                    autoregressive_feature1 = torch.roll(spikes[target_neuron_idx], 1).numpy() # 1 time step history effects
                    autoregressive_feature2 = torch.roll(spikes[target_neuron_idx], 2).numpy() # 2 time step history effects

                    # create design matrix
                    X = np.vstack((source_spikes, autoregressive_feature1, autoregressive_feature2))

                    # use a linear regression model to assess if there is a significantly non-zero association between source and target neuron
                    linear_model = sm.OLS(target_spikes.T, sm.add_constant(X.T))
                    res = linear_model.fit()

                    intervened_neuron_idx = source_neurons.index(intervened_neuron)
                    p_values = res.pvalues[intervened_neuron_idx+1] # p-values of t-test of effect from source to targat neuron (first time step coefficient)
                    
                    if np.any(p_values > alpha):
                        SCM_learned.remove_edge(intervened_neuron, target_neuron)

    return SCM_learned

def observational_learner(
    spikes, 
    node_list, 
    alpha=0.05):
    
    """
    spikes: observational spike data
    
    node_list: a list contining node names of the observed variables
        
    alpha = significance level, default is alpha=0.05
    """
    node_list = list(node_list)
    SCM_observational = nx.DiGraph()
    SCM_observational.add_nodes_from(np.sort(node_list))
            
    for idx, neuron_id in enumerate(node_list):
        
        target_spikes = spikes[idx].numpy()
        
        source_spikes = np.delete(torch.roll(spikes, 1), idx, axis=0) # effect from 1 time step before
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
                
        for k, p in enumerate(p_values):
            if p < alpha:
                SCM_observational.add_edge(source_nodes[k], neuron_id)
    
    return SCM_observational

def interventional_learner(
    spike_data, 
    node_list, 
    stimulation_protocol,
    SCM_learned,
    alpha=0.05,
    verbose=True):
    
    """
    spike_data: dictionary of observed neuron data sets, where the key is the intervention used. 'null' is no intervetion. 
    '1' means node 1 was intervened on, '1_2_3' means node 1, 2 and 3 were intervened on, etc.
    
    node_list: a list contining node names of the observed variables
    
    stimulation_protocol: list of indeces for intervened neurons at each intervention.
    
    SCM_learned: learned model based on observational data and other interventions

    alpha = significance level, default is alpha=0.05
    """
    node_list = list(node_list) 
    SCM_previous = SCM_learned.copy()
    disable = verbose is False
    # loop through each intervention set to remove confounding effects
    for intervention_set in tqdm(stimulation_protocol, total=len(stimulation_protocol), disable=disable):
        intervention_set_str = '_'.join(str(x) for x in intervention_set)        
        spikes = spike_data[intervention_set_str] # get data set where the given intervention occurred

        for intervened_neuron in intervention_set:            

            # get indeces of neurons that intervened node is correlated with
            target_neurons = [v for _, v in SCM_previous.out_edges(intervened_neuron)]
            # TODO: the target neurons that are also in the intervention set need not be analysed further
            # because we know their activity will be independent of other nodes
            
            if len(target_neurons) == 0:
                continue
            
            else:

                # loop through the neurons that the intervened node is observationally correlated with
                for target_neuron in target_neurons:

                    if target_neuron in intervention_set:
                        continue
                    
                    else:
                        target_neuron_idx = node_list.index(target_neuron)

                        # include all possible explanations for target neuron in the model
                        source_neurons = [u for u, _ in SCM_previous.in_edges(target_neuron)]
                        source_neurons_idx = [node_list.index(i) for i in source_neurons]                    
                        
                        target_spikes = spikes[target_neuron_idx].numpy()

                        source_spikes = torch.roll(spikes[source_neurons_idx], 1).numpy() # effect from 1 time step before
                        autoregressive_feature1 = torch.roll(spikes[target_neuron_idx], 1).numpy() # 1 time step history effects
                        autoregressive_feature2 = torch.roll(spikes[target_neuron_idx], 2).numpy() # 2 time step history effects

                        # create design matrix
                        X = np.vstack((source_spikes, autoregressive_feature1, autoregressive_feature2))

                        # use a linear regression model to assess if there is a significantly non-zero association between source and target neuron
                        linear_model = sm.OLS(target_spikes.T, sm.add_constant(X.T))
                        res = linear_model.fit()

                        intervened_neuron_idx = source_neurons.index(intervened_neuron)
                        p_values = res.pvalues[intervened_neuron_idx+1] # p-values of t-test of effect from source to targat neuron (first time step coefficient)
                        #print(p_values, (intervened_neuron, target_neuron))
                        if p_values > alpha:
                            SCM_learned.remove_edge(intervened_neuron, target_neuron)

    return SCM_learned