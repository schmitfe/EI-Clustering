import numpy as np
from itertools import product

def connecter(Q):
    all = list(np.arange(0,2*Q))
    ex = all[:Q]
    inh = all[Q:]
    dic = {}
    for e, number in enumerate(ex):
        dic[number] = inh[e]
    for i, number in enumerate(inh):
        dic[number] = ex[i]

    return dic

def connections(Q):
    def helper(Q):
        all = list(np.arange(0, 2 * Q))
        ex = all[:Q]
        inh = all[Q:]
        dic = {}
        for e, number in enumerate(ex):
            dic[number] = "E"
        for i, number in enumerate(inh):
            dic[number] = "I"

        return dic
    all = list(np.arange(0, 2 * Q))
    helper_lex = helper(Q)
    cluster = connecter(Q)
    dic = {}
    for i in all:
        for j in all:
            kind_i = helper_lex[i]
            kind_j = helper_lex[j]
            key = str(i).zfill(2) + str(j).zfill(2)
            dic[key] = str(kind_i) + str(kind_j)
            if cluster[i] == j or i == j:
                dic[key] = dic[key] + "_IN"
            else:
                dic[key] = dic[key] + "_OUT"
    return dic

def build(Q, values):
    matrix = np.zeros((2*Q, 2*Q))
    lexika = connections(Q)
    for i, j in product(range(2*Q), range(2*Q)):
        matrix[i,j] = values[str(lexika[str(i).zfill(2)+str(j).zfill(2)])]
    return matrix



