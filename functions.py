import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from tqdm import tqdm

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1,len(s)))

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

def get_MAG_summary(mag, n_timelags):
    summary_edges = {(u // (n_timelags+1), v // (n_timelags+1)) : None for u, v in mag if u // (n_timelags+1) != v // (n_timelags+1)}

    for edge in mag:
        u, v = edge
        neuron1, neuron2 = u // (n_timelags+1), v // (n_timelags+1)

        if neuron1 == neuron2:
            # if u and v belong to the same neuron
            continue

        elif mag[edge] == '-->':
            # when we have causal effect
            summary_edges[(neuron1, neuron2)] = '-->'

        elif mag[edge] == '<->' and summary_edges[(neuron1, neuron2)] != '-->':
            summary_edges[(neuron1, neuron2)] = '<->'
    
    return summary_edges

def get_mag_from_dag(full_dag, observed_nodes):

    latent_nodes = [node for node in full_dag.nodes() if node not in observed_nodes]
    is_adjacent = [] #stores pairs of nodes that are adjacent in the MAG
    pairs = list(itertools.combinations(observed_nodes, 2))
    skeleton = full_dag.to_undirected()

    for node1, node2 in pairs:
        #print(node1,node2)
        conditioning_set = [node for node in observed_nodes if node != node1 and node != node2]
        
        if (node1, node2) in full_dag.edges():
            is_adjacent.append((node1, node2)) # trivially inducing path
        else:
            colliders = sorted(nx.compute_v_structures(full_dag))
            # look for inducing paths
            all_simple_paths = sorted(nx.all_simple_paths(skeleton, source = node1, target = node2))
            for path in all_simple_paths:
                is_inducing_path = True
                L = len(path)

                for k in range(1, L-1): # loop through nonendpoint vertices in the path
                    parent1, child, parent2 = path[k-1], path[k], path[k+1] 

                    child_not_latent = child not in latent_nodes
                    not_collider =  (parent1, child, parent2) not in colliders
                    not_ancestor = child not in nx.ancestors(full_dag, node1) or child not in nx.ancestors(full_dag, node2)
                                        
                    if child_not_latent and (not_collider or not_ancestor):
                        is_inducing_path = False
                        # we now know it is not an inducing path and can break
                        break
                            
                if is_inducing_path:
                    is_adjacent.append((node1, node2))
                    # the pair is adjacent in mag, go to next pair
                    break

    mag = {}    
    for node1, node2 in is_adjacent:   
        if node1 in nx.ancestors(full_dag, node2): # ancestors have direct edge
            mag[(node1, node2)] = '-->'
        if node2 in nx.ancestors(full_dag, node1):
            mag[(node2, node1)] = '-->'
        if node1 not in nx.ancestors(full_dag, node2) and node2 not in nx.ancestors(full_dag, node1):
            mag[(node1, node2)] = '<->'
            mag[(node2, node1)] = '<->'
            
    return mag

def get_PAG_adjacency_matrix(pag, n_timelags):
    #0: No edge
    #1: Circle
    #2: Arrowhead
    #3: Tail

    nodes_obs = pag.getNodes()
    nodes_str = pag.getNodeNames()
    n_nodes = len(nodes_obs)
    A = np.zeros((n_nodes, n_nodes))

    for i, node1 in enumerate(nodes_obs):
        for j, node2 in enumerate(nodes_obs):
            if pag.isAdjacentTo(node1, node2):
                edge_str = str(pag.getEdge(node1, node2))                
                edge_type = edge_str[(edge_str.find(' ')+1):edge_str.rfind(' ')]

                if edge_type == '-->': # i --> j
                    A[i, j] = 2
                    A[j, i] = 3
                if edge_type == '<->': # i <-> j
                    A[i, j] = 2
                    A[j, i] = 2
                if edge_type == 'o->': # i o-> j
                    A[i, j] = 2
                    A[j, i] = 1
                if edge_type == 'o-o': # i o-o j
                    A[i, j] = 1
                    A[j, i] = 1
    return A