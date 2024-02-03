import sys
import torch

import jpype.imports

BASE_DIR = ".."
sys.path.append(BASE_DIR)
jpype.startJVM(classpath=[f"{BASE_DIR}/pytetrad/resources/tetrad-current.jar"])

import edu.cmu.tetrad.graph as tg
import edu.cmu.tetrad.search as ts
import edu.cmu.tetrad.data as td
import java.util as jutil
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import d_separation
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

nodes = jutil.ArrayList()

for i in range(1, 7):
    nodes.add(tg.GraphNode('x' + str(i)))

dag = tg.EdgeListGraph(nodes)

dag.addDirectedEdge(nodes.get(0), nodes.get(1))
dag.addDirectedEdge(nodes.get(2), nodes.get(3))
dag.addDirectedEdge(nodes.get(4), nodes.get(5))
dag.addDirectedEdge(nodes.get(2), nodes.get(1))
dag.addDirectedEdge(nodes.get(2), nodes.get(5))

print('----------')
print('TETRAD INPUT DAG')
print('----------')
print(dag)

kn = td.Knowledge()

kn.addToTier(0, 'x1')
kn.addToTier(0, 'x3')
kn.addToTier(0, 'x5')

kn.addToTier(1, 'x2')
kn.addToTier(1, 'x4')
kn.addToTier(1, 'x6')

kn.setRequired('x1','x2')
kn.setRequired('x3','x4')
kn.setRequired('x5','x6')

kn.setTierForbiddenWithin(0, False)
kn.setTierForbiddenWithin(1, False)

#nodes.get(2).setNodeType(tg.NodeType.LATENT)
#nodes.get(3).setNodeType(tg.NodeType.LATENT)

print(kn)

FCI = ts.Fci(ts.test.MsepTest(dag))
FCI.setKnowledge(kn)
#FCI.setPossibleMsepSearchDone(True)
pag = FCI.search()
print(pag)
'''
kn = td.Knowledge()

# autoregressive effects
kn.setRequired('x1','x2')
kn.setRequired('x3','x4')
kn.setRequired('x5','x6')

# true causal effects
#kn.setRequired('x3','x2')
#kn.setRequired('x3','x6')


# cause preceeding effect
kn.setForbidden('x2','x1')
kn.setForbidden('x2','x3')
kn.setForbidden('x2','x5')

kn.setForbidden('x4','x1')
kn.setForbidden('x4','x3')
kn.setForbidden('x4','x5')

kn.setForbidden('x6','x1')
kn.setForbidden('x6','x3')
kn.setForbidden('x6','x5')

# simultaneous effects
kn.setForbidden('x1','x3')
kn.setForbidden('x1','x5')
kn.setForbidden('x3','x1')
kn.setForbidden('x3','x5')
kn.setForbidden('x5','x1')
kn.setForbidden('x5','x3')

kn.setForbidden('x2','x4')
kn.setForbidden('x2','x6')
kn.setForbidden('x4','x2')
kn.setForbidden('x4','x6')
kn.setForbidden('x6','x2')
kn.setForbidden('x6','x4')

print('----------')
print('TETRAD knowledge')
print('----------')
print(kn)
#kn.addToTier(0, 'x1')
#kn.addToTier(0, 'x3')
#kn.addToTier(0, 'x5')

#kn.addToTier(1, 'x2')
#kn.addToTier(1, 'x4')
#kn.addToTier(1, 'x6')

#kn.setTierForbiddenWithin(0, False)
#kn.setTierForbiddenWithin(1, False)

FCI = ts.Fci(ts.test.MsepTest(dag))
FCI.setKnowledge(kn)
FCI.setCompleteRuleSetUsed(False)
FCI.setDoDiscriminatingPathRule(False)

pag = FCI.search()
print('----------')
print('TETRAD FCI SEARCH')
print('----------')

print(pag)

### with causal-learn 

print('----------')
print('FINISH')
print('----------')
'''






'''
G = nx.DiGraph()
G.add_nodes_from(np.arange(6))
G.add_edges_from([(0, 1), (2, 3), (4, 5), (2, 1), (2, 5)])

pag, _ = fci(np.empty(shape=(10**2, 6)), 
            d_separation, 
            true_dag=G,
            )
#print(pag)

nodes = pag.get_nodes()

kn = BackgroundKnowledge()
kn.add_node_to_tier(nodes[0], 0)
kn.add_node_to_tier(nodes[2], 0)
kn.add_node_to_tier(nodes[4], 0)

kn.add_node_to_tier(nodes[1], 1)
kn.add_node_to_tier(nodes[3], 1)
kn.add_node_to_tier(nodes[5], 1)

print(kn)

kn.add_required_by_node(nodes[0], nodes[1])
kn.add_required_by_node(nodes[2], nodes[3])
kn.add_required_by_node(nodes[4], nodes[5])

pag, _ = fci(np.empty(shape=(10**2, 6)), 
            d_separation, 
            true_dag=G,
            background_knowledge=kn,
            )
print('----------')
print('CAUSAL-LEARN')
print('----------')

print(pag)
'''