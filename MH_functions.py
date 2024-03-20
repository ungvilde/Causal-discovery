import networkx as nx
import numpy as np
import numba as nb
import itertools

from tetrad_helpers import *
from pag2mag_helpers import *
from functions import *
#from update_graph import *

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
    
    # print('Verteces:')
    # print(vertices_with_undecided_marks)
    # print('Entropy:')
    # print(entropy)
    return vertices_with_undecided_marks[np.argmax(entropy)] # intervention node

def MCMC_sampler(pmg, burnin, L):
    #np.random.seed(221355)
    n_nodes = pmg.shape[0]
    
    # get an initial MAG and a set of consistent MAGS
    mag_init = pag2mag(pmg) 

    ###graph_list = collect(mag_init, burnin, L)
    MAG_set_init = get_consistent_MAGs(mag_init)

    # sample a random MAG in the set
    rand_idx = np.random.choice(len(MAG_set_init))
    mag_prev = MAG_set_init[rand_idx]

    graph_list = [] # for storing sampled MAGs
    MAG_set_prev = get_consistent_MAGs(mag_prev)
    N_prev = len(MAG_set_prev)

    for _ in tqdm(range(burnin + L)):
        
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

def collect(mag_init, burnin, L):
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

def get_discriminating_path(mag, a, b, c):
    # path (d, .., a, b, c)
    # a <-* b o-* c, with a -> c
    # the algorithm searches for a discriminating
    # path p = <d, . . . , a,b,c> for b of minimal length

    p = mag.shape[0]
    visited = np.zeros(p, dtype= np.bool_)
    visited[[a, b, c]] = True
    
    # find all neighbours of a not visited yet, d *-> a
    listD = list(np.where((mag[a, :] != 0) * (mag[:, a] == 2) * ~visited)[0])
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
                    listR = list(np.where((mag[d, :] != 0) * (mag[:, d] == 2) * ~visited)[0]) # r *-> d
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
    method='entropy-byneuron', 
    burnin=500, 
    n_samples=500,
    max_iter=10):
    """
    summary_graph: graph to learn
    n_neurons: number of neurons in data
    n_timelags: number fo time lags in model
    method: random or entropy for selecting interventions
    """
    n_obs = len(observed_neurons)
    n_hidden = len(latent_neurons)
    n_neurons = n_obs + n_hidden

    # Get the full time graph representation
    fulltime_dag,_,_ = create_fulltime_graph_tetrad(
        summary_graph, 
        n_timelags=n_timelags, latent_nodes=latent_neurons, refractory_effect=n_timelags
        )
    
    # first learn from observational data and time ordering
    fci = ts.Fci(ts.test.MsepTest(fulltime_dag))
    kn = td.Knowledge() 
    kn = timeseries_knowledge(n_neurons, n_timelags=n_timelags, refractory_effect=n_timelags)
    fci.setKnowledge(kn) # add BK
    pmg_null = fci.search() 
    #print('result from observational data:')
    #print(pmg_null)

    pmg = get_adjacency_matrix_from_tetrad(pmg_null, n_timelags = n_timelags) # adjacency matrix for PAG
    
    p = pmg.shape[0]
    for i in range(p): # entner rule
        for j in range(p):
            t_diff = i % (n_timelags + 1) - j % (n_timelags + 1)
            if t_diff == 1 and i // (n_timelags+1) != j // (n_timelags+1) and pmg[i,j] == 3 and pmg[j,i] == 2:
                #print(j,'->',i)
                if pmg[i-1, j-1] == 1 and pmg[j-1, i-1] == 2:
                    pmg[i-1, j-1] = 3 # make circle into tail

    #print('after entner update')
    #get_pag_arrows(pmg)
    
    intervention_count = 0
    # print('result from observational data:')
    # get_pag_arrows(pmg)
    fulltime_graph, _, _ = create_fulltime_graph(summary_graph, n_timelags=n_timelags)
    
    while not is_identified(pmg):
        if method == 'entropy-singlenode':
            print(f'Collect {n_samples+burnin} MH samples and select intervention')
            intervention_node = select_intervention_node(pmg, burnin, n_samples)
            intervention_neuron = intervention_node // (n_timelags+1) # from full time node index to neuron index

        elif method == 'entropy-byneuron':
            print(f'Collect {n_samples+burnin} MH samples and select intervention')
            intervention_neuron = select_intervention_neuron(pmg, burnin, n_samples, n_neurons=n_neurons, n_timelags=n_timelags)
        elif method == 'random':
            vertices_with_undecided_marks = np.unique(np.where(pmg == 1)[1])
            neurons_with_undecided_marks = np.unique(vertices_with_undecided_marks // (n_timelags + 1))
            intervention_neuron = np.random.choice(neurons_with_undecided_marks)
            #intervention_neuron = intervention_node // (n_timelags+1)
        else:
            raise NotImplementedError('selection method not implemented')
        intervention_count+=1
        print(f'Doing intervention no. {intervention_count} on neuron {intervention_neuron}.')
        # intervene on max. entropy neuron, corresponding to these nodes in full time graph
        intervention_nodes_fulltime = np.arange(intervention_neuron*(n_timelags+1), (intervention_neuron+1)*(n_timelags+1))

        #print('Updating graph based on intervention.')
        for x in intervention_nodes_fulltime:
            adjacent_nodes = np.where(pmg[:, x] != 0)[0]
            for nb_x in reversed(adjacent_nodes):
                if x in nx.ancestors(fulltime_graph, nb_x): # intervention_node causes nb_x
                    #print(f'We know {x} --> {nb_x}')
                    pmg[nb_x, x] = 3
                    pmg[x, nb_x] = 2
                else: # intervention node does not cause n
                    #print(f'We know {x} <-* {nb_x}')
                    pmg[nb_x, x] = 2
            # update graph
            pmg = update_graph(pmg)

        if intervention_count>max_iter:
            break

    return pmg, intervention_count

def update_graph(pag):
    if np.any(pag != 0):
        p = pag.shape[0]
        old_pag = np.zeros((p,p))
        while np.any(old_pag != pag):
            old_pag = pag.copy() # continue until no further updates
            #print('Applying R1')
            pag = apply_R1(pag)
            #print('Applying R2')
            pag = apply_R2(pag)
            #print('Applying R4')
            pag = apply_R4_new(pag) # Wang rule
            #print('Applying R8')
            pag = apply_R8(pag)
            #print('Applying R10')
            pag = apply_R10(pag)
    return pag

def apply_R4_new(pag):
    ind = np.column_stack(np.where((pag != 0) * (pag.T == 1)))
    while len(ind) > 0:
        b = ind[0, 0]
        c = ind[0, 1]
        ind = ind[1:]
        indA = list(np.where( (pag[b,:]==2)*(pag[:,b]!=0)*(pag[c,:]==3)*(pag[:,c]==2))[0])
        while(len(indA) > 0 and pag[c, b] == 1):
            a = indA[0]
            indA = indA[1:]
            done = False
            while(not done and pag[a, b] != 0 and pag[a, c] != 0 and pag[b, c] != 0):
                md_path = get_discriminating_path(pag, a, b, c)
                N_md = len(md_path)
                if N_md == 0:
                    done = True
                else:
                    print("R4: There is a discriminating path between",md_path[0],"and",
                    c,"for", b,", and",b,"is in Sepset of", c,"and",md_path[0],
                    ". Orient:",b,"->",c)
                    pag[b, c] = 2
                    pag[c, b] = 3
                    done = True
    return pag

def apply_R1(pag):
    ind = np.column_stack(np.where((pag == 2) * (pag.T != 0)))
    for i in range(len(ind)):
        a = ind[i,0]
        b = ind[i,1]
        indC = set(np.where( (pag[b,:] != 0)*(pag[:,b] == 1)*(pag[a,:] == 0)*(pag[:,a] == 0))[0])
        indC = indC.difference([a])
        if len(indC) > 0:
            print("R1:","Orient:",a,"*->",b,"o-*",indC, "as:",b,"->",indC)
            pag[b, list(indC)] = 2
            pag[list(indC), b] = 3
    return pag

def apply_R2(pag):
    ind = np.column_stack(np.where((pag == 1) * (pag.T != 0)))
    for i in range(len(ind)):
        a = ind[i, 0]
        c = ind[i, 1]        
        indB = list(np.where(np.logical_or(
                                    (pag[a, :] == 2) * (pag[:, a] == 3) * (pag[c,:] != 0) * (pag[:,c] == 2), 
                                    (pag[a, :] == 2) * (pag[:, a] != 0) * (pag[c,:] == 3) * (pag[:, c] == 2) 
                                    ))[0])
        if len(indB)>0:
            print("R2: Orient:",a,"->", indB, "*->",c,"or",a,"*->",indB, "->",c,"with",a, "*-o", c,"as:",a, "*->",c)
            pag[a, c] = 2
    return pag

def apply_R8(pag):
    ind = np.column_stack(np.where((pag == 2) * (pag.T == 1)))
    for i in range(len(ind)):
        a = ind[i,0]
        c = ind[i,1]  
        indB = list(np.where( (pag[:,a] == 3)*np.logical_or(pag[a,:] == 1, pag[a,:] == 2)*(pag[c,:] == 3)*(pag[:,c] == 2))[0])
        if len(indB) > 0:
            print(f'R8: Orient {c} *-- {a}.')
            pag[c,a] = 3
    return pag

def apply_R10(pag):
    # simplified version
    ind = np.column_stack(np.where((pag == 2) * (pag.T == 1)))
    while len(ind) > 0:
        a = ind[0, 0]
        c = ind[0, 1]
        ind = ind[1:]
        indB = list(np.where((pag[c,:]==3)*(pag[:,c]==2))[0])
        if len(indB) >= 2:
            counterB = 0
            while counterB < len(indB) and pag[c, a] == 1:
                b = indB[counterB]
                counterB += 1

                indD = list(set(indB).difference([b]))
                counterD = 0
                while counterD < len(indD) and pag[c,a]==1:
                    d = indD[counterD]
                    counterD += 1

                    if ((pag[a, b]==1 or pag[a, b] == 2) and 
                        (pag[b, a]==1 or pag[b, a] == 3) and 
                        (pag[a, d]==1 or pag[a, d] == 2) and
                        (pag[d, a]==1 or pag[d, a] == 3) and (pag[d, b] == 0) and (pag[b, d] == 0) 
                        ):
                        print("R10: Orient:", a, "->", c)
                        pag[c, a] = 3
    return pag

def select_intervention_neuron(pmg, burnin, L, n_timelags, n_neurons):
    
    vertices_with_undecided_marks = np.unique(np.where(pmg == 1)[1])
    neurons_with_undecided_marks = np.unique([x // (n_timelags+1) for x in vertices_with_undecided_marks])

    if len(vertices_with_undecided_marks) == 1:
        return vertices_with_undecided_marks[0] // (n_timelags+1) 
    sampled_graphs = MCMC_sampler(pmg, burnin=burnin, L=L)
    consistent_graphs = []  
    for g in sampled_graphs:
        if check_if_consistent(pmg, g):
            consistent_graphs.append(g)
    n_graph = len(consistent_graphs)
    entropy = []
    print('Collected', n_graph, 'samples consistent with the PAG.')
    for v in vertices_with_undecided_marks:
        listC = np.where(pmg[:, v] == 1)[0]  # where c *-o v
        all_local_structure = powerset(listC) # enumerate all possible local structures
        count_local_structure = np.zeros(len(all_local_structure)) # for counting occurence of each local structure in sampled graphs
        for g in consistent_graphs:
            local_structure = list(np.where(g[listC, v] == 2)[0])
            count_local_structure += 1*np.array([struct == local_structure for struct in all_local_structure])        
        count_local_structure = count_local_structure[count_local_structure != 0]
        prob = count_local_structure / n_graph
        entropy.append(-1*sum(prob * np.log2(prob)))
    
    # print('Verteces:')
    # print(vertices_with_undecided_marks)
    # print('Entropy:')
    # print(entropy)

    entropy_by_neuron = {neuron : 0 for neuron in neurons_with_undecided_marks}
    for i, v in enumerate(vertices_with_undecided_marks):
        entropy_by_neuron[v // (n_timelags+1)] += entropy[i]
    
    print(neurons_with_undecided_marks)
    print(entropy_by_neuron)
    
    return max(entropy_by_neuron, key=entropy_by_neuron.get) # intervention node