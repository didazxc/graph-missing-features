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
'''


if __name__ == '__main__':
    # plt.main()
    validator.main()
