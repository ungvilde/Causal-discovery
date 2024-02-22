import networkx as nx
import numpy as np
import itertools

from tetrad_helpers import *
from pag2mag_helpers import *
from functions import *

def powerset(lst):
    all_combs = []
    for i in range(0, len(lst)+1):
        combination = [list(x) for x in itertools.combinations(lst, i)]
        all_combs.extend(combination)
    return all_combs

def select_intervention_node(pmg, burnin, L):
    #nodes = np.arange(pmg.shape[0])
    vertices_with_undecided_marks = np.unique(np.where(pmg == 1)[1])

    if len(vertices_with_undecided_marks) == 1:
        return vertices_with_undecided_marks[0]
    
    sampled_graphs = MCMC_sampler(pmg, burnin=burnin, L=L)
    consistent_graphs = []
    
    for g in sampled_graphs:
        if check_if_consistent(pmg, g):
            consistent_graphs.append(g)
    

    n_graph = len(consistent_graphs)
    entropy = []
    
    print('Collected', n_graph, 'samples consistent with the PAG.')

    for v in vertices_with_undecided_marks:
        #ind = np.where(pmg[:, v] == 1)[0] # where c *-o v
        #listC = nodes[ind] # get set of nodes c
        listC = np.where(pmg[:, v] == 1)[0] 
        all_local_structure = powerset(listC) # enumerate all possible local structures
        count_local_structure = np.zeros(len(all_local_structure)) # for counting occurence of each local structure in sampled graphs
        
        for g in consistent_graphs:
            #has_arrowhead = np.where(g[listC, v] == 2)[0] 
            #local_structure = list(listC[has_arrowhead]) # the c nodes with arrowhead in sampled consistent graph
            local_structure = list(np.where(g[listC, v] == 2)[0])
            count_local_structure += 1*np.array([struct == local_structure for struct in all_local_structure])
        
        count_local_structure = count_local_structure[count_local_structure != 0]
        #prob = count_local_structure / np.sum(count_local_structure)
        prob = count_local_structure / n_graph
        entropy.append(-1*sum(prob * np.log2(prob)))
    
    print('Verteces:')
    print(vertices_with_undecided_marks)
    print('Entropy:')
    print(entropy)
    return vertices_with_undecided_marks[np.argmax(entropy)] # intervention node

def MCMC_sampler(pmg, burnin, L):
    n_nodes = pmg.shape[0]
    
    # get an initial MAG and a set of consistent MAGS
    mag_init = pag2mag(pmg) # NOTE: probably wont work when there are o-o type edges.

    MAG_set_init = get_consistent_MAGs(mag_init)

    # sample a random MAG in the set
    rand_idx = np.random.choice(len(MAG_set_init))
    mag_prev = MAG_set_init[rand_idx]

    graph_list = [] # for storing sampled MAGs
    MAG_set_prev = get_consistent_MAGs(mag_prev)
    N_prev = len(MAG_set_prev)

    for _ in range(burnin + L):
        
        # sample next candidate mag from transformation set of previous mag
        rand_idx = np.random.choice(N_prev)
        mag_next = MAG_set_prev[rand_idx]
        MAG_set_next = get_consistent_MAGs(mag_next)
        N_next = len(MAG_set_next)

        # transition probability
        prob = min(1, N_prev / N_next)  
        if np.random.rand() <= prob: # if we accept
            #print('accepted, prob =', prob)
            graph_list.append(mag_next)
            mag_prev = mag_next
            MAG_set_prev = MAG_set_next
            N_prev = N_next

        else: # if we reject
            #print('rejected, prob =', prob)
            graph_list.append(mag_prev)

    return graph_list

def check_if_consistent(pmg, sampled_pag):
    # determine whether the arrowheads in pmg are also in s_pag
    ind = np.where(pmg == 2)
    arrowhead_same = np.all(sampled_pag[ind] == 2)

    ind = np.where(pmg == 3)
    tail_same = np.all(sampled_pag[ind] == 3)

    return arrowhead_same * tail_same
  
def transition_conditions(mag, a, b):
    is_satisfied = True

    # check condition 1
    mag1 = mag.copy()
    mag1[a,b] = 0
    mag1[b, a] = 0
    if a in get_possible_ancestors(mag1, b):
        is_satisfied = False
        return is_satisfied
    
    # check condition 2
    pa_a = set(np.where((mag[:,a] == 2) * (mag[a,:]== 3))[0])
    pa_b = set(np.where((mag[:,b] == 2) * (mag[b,:]== 3))[0])
    sp_a = set(np.where((mag[:,a] == 2) * (mag[a,:]== 2))[0])
    sp_b = set(np.where((mag[:,b] == 2) * (mag[b,:]== 2))[0])
    set1 = pa_a.difference(pa_b)
    set2 = sp_a.difference(pa_b.union(sp_b))
    if len(set1) > 0 or len(set2) > 0:
        is_satisfied = False
        return is_satisfied

    # check condition 3
    mag1 = mag.copy()
    list1 = np.where((mag1[:, a] != 0) * (mag1[a, :] == 2) * (mag1[:, b] == 2) * (mag1[b, :] == 3))[0]
    while(len(list1) > 0):
        c = list1[0]
        list1 = list1[1:]
        done = False

        while(not done and mag1[c, a] != 0 and mag1[c, b] != 0 and mag1[a, b] != 0):
            min_disc_path = get_discriminating_path(mag1, c, a, b)
            if len(min_disc_path) == 0:
                done = True
            else:
                is_satisfied = False
                return is_satisfied
   
    return is_satisfied

def update_paths(new_path, indeces, old_path):
    tmp = [new_path + [i] for i in indeces]
    
    if old_path is None or old_path == []: # add new path to indeces and return
        res = tmp
        
    elif isinstance(old_path[0], int):
        res = [old_path]
        for lst in tmp:
            res.append(lst)
    else:
        res = old_path + tmp
    return res

def get_discriminating_path(mag, c, a, b):
    # path (d, .., a, b, c)
    # a <-* b o-* c, with a -> c
    # the algorithm searches for a discriminating
    # path p = <d, . . . , a,b,c> for b of minimal length

    p = mag.shape[0]
    visited = np.zeros(p, dtype= np.bool_)
    visited[[a, b, c]] = True
    
    # find all neighbours of a not visited yet, d *-> a
    listD = np.where((mag[a, :] != 0) * (mag[:, a] == 2) * ~visited)[0] 
    if len(listD) > 0:
        path_list = update_paths([a], listD, None)
        while(len(path_list) > 0):
            # next element in the queue
            mpath = path_list[0]
            d = mpath[-1]

            if mag[c, d] == 0 and mag[d, c] == 0:
                min_disc_path = list(reversed(mpath)) + [b, c]
                return min_disc_path
            
            else:
                pred = mpath[-2]
                path_list.remove(mpath)

                if mag[d, c] == 2 and mag[c, d] == 3 and mag[pred, d] == 2:
                    visited[d] = True
                    listR = np.where((mag[d, :] != 0) * (mag[:, d] == 2) * ~visited)[0] # r *-> d
                    if len(listR) > 0:
                        path_list = update_paths(mpath[1:], listR, path_list)

    return [] 

def get_consistent_MAGs(mag):
    MAGs = []
    
    # judge whether a directed edge could be reversed
    nodes = np.where((mag == 2)*(mag.T == 3)) # a --> b
    #print(nodes)
    for i in range(nodes[0].shape[0]):
        a = nodes[0][i] 
        b = nodes[1][i]
        #print('check', a, '-->', b)
        
        pa_a = set(np.where((mag[:,a] == 2) * (mag[a,:]== 3))[0])
        pa_b = set(np.where((mag[:,b] == 2) * (mag[b,:]== 3))[0])
        sp_a = set(np.where((mag[:,a] == 2) * (mag[a,:]== 2))[0])
        sp_b = set(np.where((mag[:,b] == 2) * (mag[b,:]== 2))[0])
             
        if pa_b == {a}.union(pa_a) and sp_b == sp_a:
            #print(f'reverse edge {a} --> {b}:')
            new_mag = mag.copy()
            new_mag[a, b] = 3
            new_mag[b, a] = 2
            MAGs.append(new_mag)

    # judge whether a directed could be bi-directed
    nodes = np.where((mag == 2)*(mag.T == 3)) # a --> b
    for i in range(nodes[0].shape[0]):
        a = nodes[0][i]
        b = nodes[1][i]
        if transition_conditions(mag, a, b):
            #print('make bidirected')
            new_mag = mag.copy()
            new_mag[a, b] = 2
            new_mag[b, a] = 2
            MAGs.append(new_mag)           

    # judge whether one bi-directed edge can be directed
    nodes = np.where((mag == 2)*(mag.T == 2))
    for i in range(nodes[0].shape[0]):
        a = nodes[0][i]
        b = nodes[1][i]

        mag_potential = mag.copy()
        mag_potential[b, a] = 3
        if transition_conditions(mag_potential, a, b):
            #print('make directed')
            new_mag = mag_potential
            MAGs.append(new_mag)  
    
    return MAGs

def get_possible_ancestors(pag, b):
    n_nodes = pag.shape[0]
    nodes = np.arange(n_nodes)
    A = np.zeros_like(pag)
    possAn_b = set()

    for i in nodes:
        for j in nodes:
            if pag[i, j] == 2 and pag[j, i] != 2: # if i *-> j then 
                A[i, j] = 1
            if pag[i, j] == 1 and pag[j, i] == 1:
                A[i, j] = 1
                A[j, i] = 1
    
    G = nx.from_numpy_array(A, create_using=nx.MultiDiGraph())
    paths = nx.shortest_path(G, target = b)

    for endpoint in paths:
        for a in paths[endpoint]:
            possAn_b.add(a)
    
    return possAn_b

def pag_adjacency_matrix(pag, n_nodes):

    A = np.zeros((n_nodes, n_nodes))
    for i, j in pag:
        if pag[(i, j)] == '-->':
            A[i, j] = 2
            A[j, i] = 3
        if pag[(i, j)] == '<->':
            A[i, j] = 2
            A[j, i] = 2
        if pag[(i, j)] == 'o->':
            A[i, j] = 2
            A[j, i] = 1
        if pag[(i, j)] == 'o-o':
            A[i, j] = 1
            A[j, i] = 1
    
    return A

def active_learner(
    summary_graph, 
    observed_neurons, 
    latent_neurons, 
    n_timelags, 
    method='entropy', 
    burnin=500, 
    n_samples=500,
    max_iter=10):
    """
    summary_graph: graph to learn
    n_neurons: number of neurons in data
    n_timelags: number fo time lags in model
    kn: background knowledge of data
    method: random or entropy for selecting interventions
    """
    n_obs = len(observed_neurons)
    n_hidden = len(latent_neurons)
    n_neurons = n_obs + n_hidden

    fulltime_dag,_,_ = create_fulltime_graph_tetrad(
        summary_graph, 
        n_timelags=n_timelags, latent_nodes=latent_neurons, refractory_effect=n_timelags
        )
    
    # first learn from observational data and BK
    fci = ts.Fci(ts.test.MsepTest(fulltime_dag)) # learn using CI oracle
    kn = td.Knowledge() 
    kn = timeseries_knowledge(n_neurons, n_timelags=n_timelags, refractory_effect=n_timelags)
    fci.setKnowledge(kn) # add BK
    pmg_null = fci.search() 

    pmg = get_adjacency_matrix_from_tetrad(pmg_null, n_timelags = n_timelags) # adjacency matrix for PAG
    intervention_count = 0

    summary_edges = get_hypersummary(pmg_null, n_neurons)
    print('observational summary:')
    print(summary_edges)

    while not is_identified(pmg):

        if method == 'entropy':
            intervention_node = select_intervention_node(pmg, burnin, n_samples)
        else:
            raise NotImplementedError('random method not implemented yet')

        intervention_count+=1
        intervention_neuron = intervention_node // (n_timelags+1) # from full time node index to neuron index
        print(f'Doing intervention no. {intervention_count} on neuron {intervention_neuron}.')

        # intervene on max. entropy neuron, corresponding to these nodes in full time graph
        intervention_nodes_fulltime = np.arange(intervention_neuron*(n_timelags+1), (intervention_neuron+1)*(n_timelags+1))

        # get manipulated graph
        manipulated_graph = summary_graph.copy()
        manipulated_graph.remove_edges_from(summary_graph.in_edges(intervention_neuron))
        
        # plt.figure()
        # nx.draw_networkx(manipulated_graph,pos=nx.circular_layout(manipulated_graph),with_labels=True)
        # plt.show()
        
        # get full time graph under manipulation
        manipulated_fulltime_graph, _, _ = create_fulltime_graph(manipulated_graph, n_timelags=n_timelags)

        # TODO: identify adjacent nodes to intervention nodes
        # check if intervention node is ancestor in manipulated graph
        print('Updating local BK based on intervention.')
        for x in intervention_nodes_fulltime:
            adjacent_nodes = np.where(pmg[:,x] != 0)[0]
            #print('neurons adj. to ', x // (n_timelags+1), 'are', adjacent_nodes // (n_timelags+1))
            #print('nodes adj. to ', x, 'are', adjacent_nodes)

            for nb_x in adjacent_nodes:
                if nb_x // (n_timelags+1) == x // (n_timelags+1):
                    continue
                elif x in nx.ancestors(manipulated_fulltime_graph, nb_x): # x causes nb_x
                    print(f'Require x{intervention_neuron},{x % (n_timelags+1)} --> x{nb_x//(n_timelags+1)},{nb_x%(n_timelags+1)}')
                    kn.setRequired(f'x{intervention_neuron},{x % (n_timelags+1)}', f'x{nb_x//(n_timelags+1)},{nb_x%(n_timelags+1)}')
                else: # x and nb_x are confounded
                    print(f'Forbid x{intervention_neuron},{x % (n_timelags+1)} --> x{nb_x//(n_timelags+1)},{nb_x%(n_timelags+1)}')
                    kn.setForbidden(f'x{intervention_neuron},{x % (n_timelags+1)}', f'x{nb_x//(n_timelags+1)},{nb_x%(n_timelags+1)}')

        # add interventional knowledge 
        fci.setKnowledge(kn)
        pmg_interventional = fci.search()
        pmg = get_adjacency_matrix_from_tetrad(pmg_interventional, n_timelags = n_timelags) # adjacency matrix for PAG
        
        summary_edges = get_hypersummary(pmg_interventional, n_neurons)
        print('post-interventional summary:')
        print(summary_edges)

        if intervention_count>max_iter:
            break

    return pmg, intervention_count