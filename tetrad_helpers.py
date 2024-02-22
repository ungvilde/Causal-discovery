
import edu.cmu.tetrad.graph as tg
import edu.cmu.tetrad.search as ts
import edu.cmu.tetrad.data as td
import java.util as jutil

def timeseries_knowledge(
    n_neurons,
    n_timelags, 
    refractory_effect,
    require_refractory_effects=True, 
    forbid_contemporaneous_effects=True
    ):

    #refractory_effect=n_timelags #refractory effects occur at every relevant time lag

    kn = td.Knowledge()
    for i in range(n_neurons):
        for t in range(n_timelags+1):
            kn.addToTier(t, f'x{i},{t}')
    
    if require_refractory_effects:
        for i in range(n_neurons):
            for current_time in range(n_timelags):
                for t_ref in range(1, refractory_effect+1):
                    if current_time + t_ref <= n_timelags:
                        kn.setRequired(f'x{i},{current_time}', f'x{i},{current_time+t_ref}')
    
    #for i in range(n_neurons): # forbid neurons from causal influende 
    #    for current_time in range(n_timelags+1):
    #        for j in range(n_neurons):
    #            if i != j:
    #                kn.setForbidden(f'x{i},{current_time}', f'x{j},{current_time+1}')

    #for t in range(n_timelags+1):
    #    kn.setTierForbiddenWithin(t, True)
    
    return kn

def create_fulltime_graph_tetrad(G_summary, n_timelags, refractory_effect,latent_nodes):
    """
    G_summary: a networkx.DiGraph object that represents what neurons causally affect each other
    n_timelags: time window in which causal interactions occur
    refractory_effect: time window in which refractory effects occur (must be less than or equal to n_timelags)
    """

    n_neurons = G_summary.number_of_nodes()   
    nodes = jutil.ArrayList()

    nodename2idx_fulltime = dict()
    nodename2idx_summary = dict()
    
    #refractory_effect = 2

    idx=0
    for i, node_name in enumerate(G_summary.nodes()):
        for t in range(n_timelags+1):
            nodes.add(tg.GraphNode('x' + str(i) + ',' + str(t)))
            
            nodename2idx_fulltime[f'x{i},{t}'] = idx # relate node name to index in fulltime graph
            nodename2idx_summary[node_name] = i # relate node name to node index in summary graph
            
            idx+=1
        
    fulltime_dag = tg.EdgeListGraph(nodes)

    for i, node_name in enumerate(G_summary.nodes()):
        for current_time in range(n_timelags):
            
            node_current_idx = nodename2idx_fulltime[f'x{i},{current_time}']

            for t_ref in range(1, refractory_effect+1):
                refractory_effect_time = current_time + t_ref
                
                if refractory_effect_time <= n_timelags:

                    # this is the refractory effect
                    node_refractory_idx = nodename2idx_fulltime[f'x{i},{refractory_effect_time}']
                    fulltime_dag.addDirectedEdge(nodes.get(node_current_idx), nodes.get(node_refractory_idx))
            
            for _, target_name in G_summary.out_edges(node_name):
                j = nodename2idx_summary[target_name]
                
                for effect_time in range(current_time+1, n_timelags+1): 
                    target_idx = nodename2idx_fulltime[f'x{j},{effect_time}']

                    # these are causal interactions
                    fulltime_dag.addDirectedEdge(nodes.get(node_current_idx), nodes.get(target_idx))   
    
    if len(latent_nodes) > 0:
        for node_name in latent_nodes:
            i = nodename2idx_summary[node_name]
            for t in range(n_timelags+1):
                node_idx = nodename2idx_fulltime[f'x{i},{t}']
                nodes.get(node_idx).setNodeType(tg.NodeType.LATENT)

    return fulltime_dag, nodename2idx_fulltime, nodename2idx_summary

def interventional_knowledge(
    kn,
    intervened_node,
    n_neurons,
    n_timelags, 
    ):

    for i in range(n_neurons):
        for current_time in range(n_timelags):
            for effect_time in range(current_time+1, n_timelags+1):
                if i != intervened_node:
                    kn.setForbidden(f'x{i},{current_time}', f'x{intervened_node},{effect_time}')

    return kn

def get_hypersummary(pag, n_neurons):
    summary_edges = {(i, j) : set() for i in range(n_neurons) for j in range(n_neurons) if j != i}

    nodes_obs = pag.getNodes()
    nodes_str = pag.getNodeNames()

    for i, node1 in enumerate(nodes_obs):
        for j, node2 in enumerate(nodes_obs):
            if pag.isAdjacentTo(node1, node2):
                edge_str = str(pag.getEdge(node1, node2))
                neuron_id = nodes_str[i][1:str(nodes_str[i]).find(',')]
                target_id = nodes_str[j][1:str(nodes_str[j]).find(',')]
                
                if neuron_id != target_id and edge_str.find(f'x{neuron_id}') < edge_str.find(f'x{target_id}'):
                    edge_str = edge_str[(edge_str.find(' ')+1):edge_str.rfind(' ')]
                    summary_edges[(int(neuron_id), int(target_id))].add(edge_str)

    summary_edges = { edge: edge_type for edge, edge_type in summary_edges.items() if len(edge_type) > 0}

    for edge in summary_edges:
        if '-->' in summary_edges[edge]:
            summary_edges[edge] = '-->'
        if 'o->' in summary_edges[edge] and not '-->' in summary_edges[edge]:
            summary_edges[edge] = 'o->'
        if 'o-o' in summary_edges[edge] and not '-->' in summary_edges[edge] and not 'o->' in summary_edges[edge]:
            summary_edges[edge] = 'o-o'
        if '<->' in summary_edges[edge] and not '-->' in summary_edges[edge] and not 'o->' in summary_edges[edge] and not 'o-o' in summary_edges[edge]:
            summary_edges[edge] = '<->'

    return summary_edges

def get_interventions(hypersummary, n_neurons):
    unresolved_count = {}
    unresolved_targets = {}

    for u, v in hypersummary:
        edge_str= hypersummary[(u, v)]
        if edge_str.find('o-') != -1:
            unresolved_count[u] = 0
            unresolved_targets[u] = []

    for u, v in hypersummary:
        edge_str= hypersummary[(u, v)]
        if edge_str.find('o-') != -1:
            unresolved_count[u] += 1
            unresolved_targets[u].append(v)

    return sorted(unresolved_count, key=unresolved_count.get, reverse=True), unresolved_targets

