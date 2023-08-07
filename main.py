from amgraph.data import plt
from amgraph import validator

'''
           nodes_num edges_num attrs_num homophily num_components assortativity density attr_sparsity
cora            2708     10556      1433   3537606             78       -0.0659  0.0029        0.0127  0.9  0.6
citeseer        3327      9104      3703  10150467            438        0.0484  0.0016        0.0085  0.8  0.45
computers      13752    491722       767   9490277            314       -0.0565  0.0052        0.3484  0.99 0.28
photo           7650    238162       745   4957745            136       -0.0449  0.0081        0.3474  0.99 0.4
steam           9944    533962       352   3406664           6741       -0.4584  0.0108        0.0239  0.98 0
pubmed         19717     88648       500   8901094              1       -0.0436  0.0005        0.1002  0.625  5:3
cs             18333    163788      6805 101206928              1        0.1126  0.0010        0.0088  0.050  1:19
arxiv         169343   2315598       128  16921272              1       -0.0431  0.0002        1.0000  0.375  3:5
products     2449029 123718024       100 200095968          52658       -0.0420  0.0000        0.9903
'''


if __name__ == '__main__':
    # plt.main()
    validator.main()
