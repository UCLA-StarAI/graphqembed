from itertools import product, combinations
from collections import defaultdict
from sympy import symbols, Symbol

if __name__ == '__main__':
    terms = [] # Each term is a number, set pair
    cnf = [['R', symbols('b1')], ['S', symbols('b2')], ['Q', symbols('b3')]] # list of clauses
    # cnf = [['R', symbols('b1')], ['S', symbols('b2')]] # list of clauses
    dnf = [set(x) for x in product(*cnf)]

    for i in range(len(dnf)):
        for it in combinations(dnf, i+1):
            its = set.union(*it)
            strs = [x for x in its if type(x) == str]
            syms = [x for x in its if type(x) == Symbol]
            terms.append((pow(-1, i) * reduce(lambda x,y: x*y, syms, 1), strs))

    print terms

    sums = defaultdict(int)
    for t in terms:
        sums[tuple(t[1])] += t[0]

    print len(sums)
    print sums