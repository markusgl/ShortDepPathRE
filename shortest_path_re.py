"""
Shortest path relation extraction
"""

import networkx as nx
import logging
import matplotlib.pyplot as plt
import en_core_web_md
import de_core_news_sm

from networkx.exception import NodeNotFound, NetworkXNoPath

from flair_embeddings import FlairEmbeddingModels
from relation_types import RelationTypes

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ShortestPathRE:
    def __init__(self, me_list=None, embeddings_model=None, nlp=None, relationship_list=None):
        self.me_list = me_list
        self.nlp = nlp
        self.embeddings_model = embeddings_model
        self.relationship_list = relationship_list
        self.relation_types = RelationTypes()

    @classmethod
    def de_lang(cls):
        me_list = ['ich', 'mein', 'meine']
        embeddings_model = FlairEmbeddingModels().de_lang()
        nlp = de_core_news_sm.load()
        relationship_list = ['vater', 'mutter', 'sohn', 'tochter', 'bruder', 'schwester', 'enkel', 'enkelin',
                             'großvater', 'großmutter', 'ehemann', 'ehefrau', 'onkel', 'tante', 'freund']

        return cls(me_list, embeddings_model, nlp, relationship_list)

    @classmethod
    def en_lang(cls):
        me_list = ['i', 'my']
        embeddings_model = FlairEmbeddingModels().en_lang()
        nlp = en_core_web_md.load()

        relationship_list = ['father', 'mother', 'sister', 'brother', 'son', 'daughter', 'husband', 'wife',
                             'grandson', 'granddaughter', 'grandmother', 'grandfather', 'uncle', 'aunt', 'friend']

        return cls(me_list, embeddings_model, nlp, relationship_list)

    def search_shortest_dep_path(self, e1, e2, sentence, plot_graph=False):
        graph = self.__build_undirected_graph(sentence, plot_graph)
        shortest_path = None

        try:
            shortest_path = nx.shortest_path(graph, source=e1.lower(), target=e2.lower())
        except NodeNotFound as err:
            logger.warning(f'Node not found: {err}')
        except NetworkXNoPath as err:
            logger.warning(f'Path not found: {err}')

        return shortest_path

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

    def __measure_sp_rel_similarity(self, shortest_path):
        """
        Measures the cosine similarity between word embeddings
        :param shortest_path: dict of sp values
        :return: relation type with the highest score
        """
        relation = None
        highest_score = 0
        highest_rel = None
        threshold = 0.6

        for rel in self.relationship_list:
            try:
                # get word embeddings representation of shortest path and relation
                score = self.embeddings_model.n_similarity(shortest_path, [rel])
                logger.debug(f'{rel} {score}')
                if score > highest_score:
                    highest_score = score
                    highest_rel = rel
            except KeyError as err:
                logger.debug(err)

        if highest_score > threshold:
            logger.debug(f'Highest score for {shortest_path} - {highest_rel}, Score: {highest_score}')
            relation = self.relation_types.get_relation_type(highest_rel)

        return relation

