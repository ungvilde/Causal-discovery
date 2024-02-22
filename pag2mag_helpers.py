import networkx as nx
import numpy as np
import networkx as nx

def pag2mag(pag, max_chord=10):
    mag = pag.copy() # copy partial ancestral graph adjacency matrix
    nodes = np.where((pag == 2)*(pag.T==1)) # a o-> b
    for i in range(nodes[0].shape[0]):
        a = nodes[0][i] 
        b = nodes[1][i]
        mag[b,a] = 3 # transform o-> to -->

    A_undir = mag.copy()
    A_dir = mag.copy()
    A_t = mag + mag.T    

    A_undir[A_t != 2] = 0 # Here, we are interested only in the o-o edges, delete all the other edges
    A_dir[A_t == 2] = 0 # Here, we have the edges that are determined

    # Find connected components of o-o edges
    G_undir = nx.from_numpy_array(A_undir, create_using=nx.Graph())
    conn_comp = [list(c) for c in nx.connected_components(G_undir)]
    
    # Here we transform circle component to a valid DAG
    A_valid_dag = A_undir.copy()

    for cc in conn_comp:
        if len(cc) > 1: # not singleton 
            if len(cc) > max_chord: # to big to handle
                return None
            A_cc = A_undir[cc,:][:,cc] # adj. matrix for con. comp.
            G_cc = nx.from_numpy_array(A_cc, create_using=nx.Graph()) 
            if not nx.is_chordal(G_cc): # Need to be chordal to be valid
                return None
            _A = get_special_dag(A_undir, A_cc, cc) # DAG with orientations of circle component.
            A_valid_dag *= _A 
    
    mag = A_valid_dag + A_dir # get MAG with valid transformations
    return mag

def get_special_dag(gm, a, cc_nodes):
    tmp = a.copy() # for selecting subsets of a
    cc_names = list(cc_nodes.copy()) # to get correct node label for nodes in connected component

    while(np.sum(a) != 0):
        sink_nodes = find_sink_nodes(a) # get sink nodes
        for x in sink_nodes:
            if check_adjacent(a, x): 
                inc_to_x = (a[:, x] == 1) * (a[x, :] == 1) # nodes that point to x
                if np.any(inc_to_x):
                    real_inc_to_x = [cc_names[i] for i in range(len(cc_names)) if inc_to_x[i]] # get node label to transform
                    real_x = cc_names[x] # true target
                    gm[real_x, real_inc_to_x] = 3
                    gm[real_inc_to_x, real_x] = 2
                cc_names.pop(x) # remove x when done with transformation
                select = [idx for idx in range(a.shape[0]) if idx != x] 
                a = tmp[select, :][:, select] # subset of a to continue 
                break

    return gm

def find_sink_nodes(A):
    AA = A.copy()
    AA[(AA == AA.T)*(AA==1)] = 0
    return np.where(np.sum(AA, axis=0) == 0)[0]

def check_adjacent(A, x): # check if adj. y to x also is adj. to eachother, to ensure shielded collider
    A1 = (A == 1)
    r = A1[x, :]
    c = A1[:, x]
    nb_x = np.where(np.logical_or(r, c))[0] # neighbours of x
    nb_x = set([xx for xx in nb_x])
    undir_n = np.where(r * c)[0]
    for y in undir_n:
        adj_x = nb_x.difference(set([y]))
        ny = np.where(np.logical_or(A1[y,:], A1[:,y]))[0]
        ny = set([yy for yy in ny])
        adj_y = ny.difference(set([x]))
        if not np.all([x in adj_y for x in adj_x ]):
            return False
    return True