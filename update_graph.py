import numpy as np
from MH_functions import get_discriminating_path

def update_graph(pag):
    if np.any(pag != 0):
        p = pag.shape[0]
        old_pag = np.zeros((p,p))
        while np.any(old_pag != pag):
            old_pag = pag.copy()
            pag = apply_R1(pag)
            pag = apply_R2(pag)
            pag = apply_R4_new(pag)
            pag = apply_R8(pag)
    return pag

def apply_R4_new(pag):
    ind = np.column_stack(np.where((pag != 0) * (pag.T == 1)))
    while len(ind) > 0:
        b = ind[0, 0]
        c = ind[0, 1]
        ind = ind[1:]
        indA = np.where( (pag[b,:]==2)*(pag[:,b]!=0)*(pag[c,:]==3)*(pag[:,c]==2))[0]
        while(len(indA) > 0 and pag[c, b]==1):
            a = indA[0]
            indA[1:]
            done = False
            while(not done and pag[a, b] != 0 and pag[a, c] != 0 and pag[b,c] != 0):
                md_path = get_discriminating_path(pag, a, b, c)
                N_md = len(md_path)
                if N_md == 1:
                    done = True
                else:
                    print(f'Orient {b} --> {c}.')
                    pag[b,c] = 2
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
            print(f'Orient {b} --> {indC}.')
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
            pag[a, c] = 2
            print("Orient:",a,"->", indB, "*->",c,"or",a,"*->",indB, "->",c,"with",a, "*-o", c,"as:",a, "*->",c)
    return pag

def apply_R8(pag):
    ind = np.column_stack(np.where((pag == 2) * (pag.T == 1)))
    for i in range(len(ind)):
        a = ind[i,0]
        c = ind[i,1]  
        indB = np.where( (pag[:,a] == 3)*np.logical_or(pag[a,:] == 1, pag[a,:] == 2)*(pag[c,:] == 3)*(pag[:,c] == 2))[0]
        if len(indB) > 0:
            pag[c,a] = 3
            print(f'Orient {c} *-- {a}.')
    return pag


def apply_R10(pag):
    ind = np.column_stack(np.where((pag == 2) * (pag.T == 1)))
    while len(ind) > 0:
        a = ind[0, 0]
        c = ind[0, 1]
        ind = ind[1:]
        indB = list(np.where((pag[c,:]==3)*(pag[:,c]==2))[0])
        if len(indB) >= 2:
            counterB = 0
            while counterB < len(indB) and pag[c, a] == 1:
                counterB += 1
                b = indB[counterB]
                indD = set(indB).difference([b])
                counterD = 0
                while counterD < len(indD) and pag[c,a]==1:
                    counterD += 1
                    d = indD[counterD]
                    if ((pag[a, b]==1 or pag[a, b] == 2) and 
                        (pag[b, a]==1 or pag[b, a] == 3) and 
                        (pag[a, d]==1 or pag[a, d] == 2) and
                        (pag[d, a]==1 or pag[d, a] == 3) and (pag[d, b] == 0) and (pag[b, d] == 0) 
                        ):
                        print("Orient:", a, "->", c)
                        pag[c, a] = 3
    return pag