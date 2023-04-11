from amgraph.data import plt
from amgraph import validator

'''
           homophily num_components assortativity
cora         15.3584             78       -0.0659
citeseer     25.6066            438        0.0484
computers   144.2162            314       -0.0565
photo       134.0178            136       -0.0449
steam         7.5387           6741       -0.4584
pubmed        0.1347              1       -0.0436
cs           41.6873              1        0.1126
arxiv         2.6927              1       -0.0431

normalized
           homophily num_components assortativity density
cora       1306.3541             78       -0.0659  0.0029
citeseer   3050.9373            438        0.0484  0.0016
computers   690.1016            314       -0.0565  0.0052
photo       648.0712            136       -0.0449  0.0081
steam       342.5850           6741       -0.4584  0.0108
pubmed      451.4426              1       -0.0436  0.0005
cs         5520.4785              1        0.1126  0.0010
arxiv        99.9231              1       -0.0431  0.0002

           nodes_num edges_num attrs_num homophily num_components assortativity density attr_sparsity
cora            2708     10556      1433    0.9116             78       -0.0659  0.0029        0.0127
citeseer        3327      9104      3703    0.8239            438        0.0484  0.0016        0.0085
computers      13752    491722       767    0.8997            314       -0.0565  0.0052        0.3484
photo           7650    238162       745    0.8699            136       -0.0449  0.0081        0.3474
steam           9944    533962       352    0.9733           6741       -0.4584  0.0108        0.0239
steam20         9944    197292       352    0.9759           8033       -0.4762  0.0040        0.0239
steam1          9944  10711922       352    0.9876            516       -0.1954  0.2167        0.0239
pubmed         19717     88648       500    0.9029              1       -0.0436  0.0005        0.1002  0.625  5:3
cs             18333    163788      6805    0.8112              1        0.1126  0.0010        0.0088  0.050  1:19
arxiv         169343   2315598       128    0.7806              1       -0.0431  0.0002        1.0000  0.375  3:5


           nodes_num edges_num attrs_num homophily num_components assortativity density attr_sparsity
cora            2708     10556      1433   3537606             78       -0.0659  0.0029        0.0127  0.9  0.6
citeseer        3327      9104      3703  10150467            438        0.0484  0.0016        0.0085  0.8  0.45
computers      13752    491722       767   9490277            314       -0.0565  0.0052        0.3484  0.99 0.28
photo           7650    238162       745   4957745            136       -0.0449  0.0081        0.3474  0.99 0.4
steam           9944    533962       352   3406664           6741       -0.4584  0.0108        0.0239  0.98 0
pubmed         19717     88648       500   8901094              1       -0.0436  0.0005        0.1002  0.625  5:3
cs             18333    163788      6805 101206928              1        0.1126  0.0010        0.0088  0.050  1:19
arxiv         169343   2315598       128  16921272              1       -0.0431  0.0002        1.0000  0.375  3:5
'''


if __name__ == '__main__':
    # plt.main()
    validator.main()
