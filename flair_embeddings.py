"""
Handle Flair embeddings
(https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md)
"""

import re

from flair.data import Sentence
from flair.embeddings import WordEmbeddings, FlairEmbeddings, BertEmbeddings
from torch import nn


class FlairEmbeddingModels:
    def __init__(self, embeddings=None):
        self.embeddings = embeddings
        self.flair_embeddings = {}

    @classmethod
    def de_lang(cls):
        """
        Factory method for german embeddings
        """
        embeddings = WordEmbeddings('de')  # German FastText embeddings
        # embeddings = WordEmbeddings('de-crawl')  # German FastText embeddings trained over crawls
        #embeddings = BertEmbeddings('bert-base-multilingual-cased')

        return cls(embeddings)

    @classmethod
    def en_lang(cls):
        """
        Factory method for english embeddings
        """
        #embeddings = WordEmbeddings('en-glove')
        embeddings = WordEmbeddings('en-crawl')  # FastText embeddings over web crawls
        #embeddings = WordEmbeddings('en-news')
        #embeddings = FlairEmbeddings('news-forward')
        #embeddings = BertEmbeddings()

        return cls(embeddings)

    def get_word_embeddings(self, text):
        """
        get the glove word embedding representation of one or multiple words
        :param text: array of one or multiple words as string
        :return: sum of word embeddings inside text
        """
        #text = re.sub(r'\s{2,}', ' ', text)
        sent = ""
        for word in text:
            sent += word + " "

        sentence = Sentence(sent)
        self.embeddings.embed(sentence)

        words_embeddings = []
        for token in sentence:
            words_embeddings.append(token.embedding)

        return sum(words_embeddings)
