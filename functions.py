import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

def compute_SHD(G_true, G_learned):
    SHD = 0
    for edge in G_learned.edges():
        if edge not in G_true.edges():
            SHD+=1
    for edge in G_true.edges():
        if edge not in G_learned.edges():
            SHD+=1
    return SHD

def count_true_positive(G_true, G_learned, nodelist):
    A_true = nx.adjacency_matrix(G_true, nodelist=nodelist).todense() 
    A_learned = nx.adjacency_matrix(G_learned, nodelist=nodelist).todense() 
    TP = np.sum((A_true == 1)*(A_learned==1))
    return TP

def count_true_negative(G_true, G_learned, nodelist):
    A_true = nx.adjacency_matrix(G_true, nodelist=nodelist).todense() 
    A_learned = nx.adjacency_matrix(G_learned, nodelist=nodelist).todense() 
    TN = np.sum((A_true == 0)*(A_learned==0)) 
    return TN

def count_false_positive(G_true, G_learned, nodelist):
    A_true = nx.adjacency_matrix(G_true, nodelist=nodelist).todense() 
    A_learned = nx.adjacency_matrix(G_learned, nodelist=nodelist).todense() 
    FP = np.sum((A_true == 0)*(A_learned==1))
    return FP

def count_false_negative(G_true, G_learned, nodelist):
    A_true = nx.adjacency_matrix(G_true, nodelist=nodelist).todense() 
    A_learned = nx.adjacency_matrix(G_learned, nodelist=nodelist).todense() 
    FN = np.sum((A_true == 1)*(A_learned==0)) 
    return FN

def compute_sensitivity(G_true, G_learned, nodelist):
    TP = count_true_positive(G_true, G_learned, nodelist)
    FP = count_false_positive(G_true, G_learned, nodelist)
    return TP / (TP + FP)

def compute_specificity(G_true, G_learned, nodelist):
    TN = count_true_negative(G_true, G_learned, nodelist)
    FN = count_false_negative(G_true, G_learned, nodelist)
    return TN / (TN + FN)

def make_rasterplot(spikedata, xlim=[0, 1000], node_labels=None):
    num_neurons = spikedata.shape[0]

    for i in range(num_neurons):
        plt.scatter(np.where(spikedata[i,:] == 1)[0], i*np.ones_like(np.where(spikedata[i,:] == 1)[0]), marker='|', s=100)

    if node_labels is None:
        node_labels = np.arange(num_neurons)

    plt.yticks(ticks=np.arange(num_neurons), labels=node_labels)
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron index")
    plt.xlim(xlim)
    plt.ylim([-0.5, num_neurons - 0.5])
    plt.title("Observed spiking activity")
    plt.tight_layout()
    plt.show()

def create_fulltime_graph(G_summary, n_timelags):
    G_fulltime = nx.DiGraph()
    num_nodes = G_summary.number_of_nodes()

    pos = dict()
    time_label = dict()
    nodename2idx = dict()

    for idx, node_name in enumerate(G_summary.nodes()):
        nodename2idx[node_name] = idx


    for i, node_name in enumerate(G_summary.nodes()):
        node_idx = i*(n_timelags+1)
        pos[node_idx] = np.array([0, i])
        time_label[node_idx] = f'X{i}{0}'

        for current_time in range(n_timelags):
            G_fulltime.add_edge(node_idx + current_time, node_idx + current_time + 1) # this is the autoregressive effect
            
            pos[node_idx + current_time + 1] = np.array([current_time + 1, i])
            time_label[node_idx + current_time + 1] = f'X{i}{current_time + 1}'
            
            for _, v in G_summary.out_edges(node_name):
                j = nodename2idx[v]
                target_idx = j*(n_timelags+1)
                
                for effect_time in range(current_time+1, n_timelags+1):
                    G_fulltime.add_edge(node_idx + current_time, target_idx + effect_time) # these are causal interactions  
    
    return G_fulltime, pos, time_label

def background_knowledge_timeseries(
    nodelist, 
    n_timelags, 
    require_history_effects=True, 
    forbid_contemporaneous_effects=True,
    forbid_effect_preceeding_cause=True
    ):

    bk = BackgroundKnowledge()
    num_nodes = len(nodelist) // (n_timelags+1)

    for i in range(num_nodes):
        node_idx = i*(n_timelags+1)

        for current_time in range(n_timelags+1):

            # History effects are required
            if require_history_effects and current_time < n_timelags:
                bk.add_required_by_node(nodelist[node_idx + current_time], nodelist[node_idx + current_time + 1])
    
            for j in range(num_nodes):
                target_idx = j*(n_timelags+1)

                # no contemporaneous effects
                if forbid_contemporaneous_effects and target_idx != node_idx:
                    bk.add_forbidden_by_node(nodelist[node_idx + current_time], nodelist[target_idx + current_time])  

                for effect_time in range(n_timelags+1):
                    # Not allowed going back in time
                    if forbid_effect_preceeding_cause and effect_time < current_time:
                        bk.add_forbidden_by_node(nodelist[node_idx+current_time], nodelist[target_idx + effect_time])
                    
    return bk
