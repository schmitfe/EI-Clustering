import numpy as np
from sympy import Symbol


def builder(Q):
    pass

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
            key = str(i) + str(j)
            dic[key] = str(kind_i) + str(kind_j)
            if cluster[i] == j or i == j:
                dic[key] = dic[key] + "_IN"
            else:
                dic[key] = dic[key] + "_OUT"
    return dic

def build(Q, values):
    matrix = [[0 for i in range(2*Q)] for j in range(2*Q)]
    lexika = connections(Q)
    for i in range(2*Q):
        for j in range(2*Q):
            matrix[i][j] = values[str(lexika[str(i)+str(j)])]
    return matrix

def vector(Q):
    clusters = np.arange(1,2*Q+1)
    vektor = []
    for c in clusters:
        vektor.append(Symbol("v"+str(c)))
    return vektor

#v=dict(EE_IN="EE", EE_OUT="EE", IE_IN="IE", IE_OUT="IE", EI_IN="EI", EI_OUT="EI", II_IN="II",
#                 II_OUT="II")

#print(build(4,v))


