"""
Shortest path relation extraction
"""

import networkx as nx
import logging
import matplotlib.pyplot as plt
import en_core_web_md

from networkx.exception import NodeNotFound, NetworkXNoPath
from flair_embeddings import FlairEmbeddingModels

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ShortestPathRE:
    def __init__(self, embeddings_model=None, nlp=None):
        self.nlp = nlp
        self.embeddings_model = embeddings_model

    @classmethod
    def en_lang(cls):
        embeddings_model = FlairEmbeddingModels().en_lang()
        nlp = en_core_web_md.load()

        return cls(embeddings_model, nlp)

    def __build_undirected_graph(self, sentence, plot=False):
        doc = self.nlp(sentence)
        edges = []
        for token in doc:
            for child in token.children:
                # TODO indicate direction of the relationship - maybe with the help of the child token 's
                source = token.lower_
                sink = child.lower_

                edges.append((f'{source}',
                              f'{sink}'))

        graph = nx.Graph(edges)

        if plot:
            self.__plot_graph(graph)

        return graph

    @staticmethod
    def __plot_graph(graph):
        pos = nx.spring_layout(graph)  # positions for all nodes
        nx.draw_networkx_nodes(graph, pos, node_size=200)  # nodes
        nx.draw_networkx_edges(graph, pos, width=1)  # edges
        nx.draw_networkx_labels(graph, pos, font_size=12, font_family='sans-serif')  # labels

        plt.axis('off')  # disable axis plot
        plt.show()

    def search_shortest_dep_path(self, e1, e2, sentence, plot_graph=False):
        graph = self.__build_undirected_graph(sentence, plot_graph)
        shortest_path = None

        try:
            shortest_path = nx.shortest_path(graph, source=e1.lower(), target=e2.lower())
        except NodeNotFound as err:
            logger.warning(f'Node not found: {err} - Sentence: {sentence}; Entity1: {e1}; Entity2: {e2}')
        except NetworkXNoPath as err:
            logger.warning(f'Path not found: {err} - Sentence: {sentence}; Entity1: {e1}; Entity2: {e2}')

        return shortest_path

