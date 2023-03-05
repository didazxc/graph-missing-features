from amgraph.data import plt
from amgraph import apa

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

           homophily num_components assortativity
cora       1306.3541             78       -0.0659
citeseer   3050.9368            438        0.0484
computers   690.1016            314       -0.0565
photo       648.0712            136       -0.0449
steam       342.5850           6741       -0.4584
pubmed      451.4426              1       -0.0436
cs         5520.4785              1        0.1126
arxiv        99.9231              1       -0.0431
'''


if __name__ == '__main__':
    plt.main()
    # apa.main()
