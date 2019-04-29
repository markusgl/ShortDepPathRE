"""
Read and pre process SemEval 2010 Task 8 data set
"""
import re
import pandas as pd
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from nltk.tokenize import WhitespaceTokenizer
from shortest_path_re import ShortestPathRE
from flair_embeddings import FlairEmbeddingModels

regex_no = r'\d'
tokenizer = WhitespaceTokenizer()
spre = ShortestPathRE().en_lang()


def extract_sentence(text):
    sentence_match = re.search(r'".*"', text)
    sentence = sentence_match.group()

    entity1_match = re.search(r'<e1>.*<\/e1>', sentence)
    entity1 = entity1_match.group()
    entity2_match = re.search(r'<e2>.*<\/e2>', sentence)
    entity2 = entity2_match.group()

    # clean up special characters
    sentence = re.sub('"|<(\/)?e\d>', '', sentence)
    e1 = re.sub(r'<(\/)?e\d>', '', entity1)
    e2 = re.sub(r'<(\/)?e\d>', '', entity2)

    return sentence, e1, e2


def get_relation_direction(label):
    reverse_match = r'\(e(2),e(1)\)'
    match_re = re.search(reverse_match, label)

    if match_re:
        return 'reverse'

    return 'norm'


def get_label_name(text):
    entities = r'(\(e(1|2),e(1|2)\))?\n'
    label = re.sub(entities, '', text)

    return label


def load_features(file_name):
    data_columns = ['sp', 'label']
    features = pd.DataFrame(columns=data_columns)

    with open(file_name, "r") as f:
        data = f.readlines()
        for i, row in enumerate(data):
            if len(row) > 1:
                sent_start = tokenizer.tokenize(row)

                if re.match(regex_no, sent_start[0]):
                    sentence, e1, e2 = extract_sentence(row)
                    label = get_label_name(data[i+1])
                    direction = get_relation_direction(label)

                    if direction == 'reverse':
                        sp = spre.search_shortest_dep_path(e1=e2, e2=e1, sentence=sentence, plot_graph=False)
                        if sp:
                            training_ex = pd.Series({'sp': sp, 'label': label})
                            features = features.append(training_ex, ignore_index=True)
                    else:
                        sp = spre.search_shortest_dep_path(e1=e1, e2=e2, sentence=sentence, plot_graph=False)
                        if sp:
                            training_ex = pd.Series({'sp': sp, 'label': label})
                            features = features.append(training_ex, ignore_index=True)

    return features


features = load_features('data/SemEval2010/train_small.txt')

embeddings = FlairEmbeddingModels().en_lang()
embedding_vectors = []
labels = []

for row in features.iterrows():

    sp_embeddings = embeddings.get_word_embeddings(row[1]['sp'])
    embedding_vectors.append(sp_embeddings)
    labels.append(row[1]['label'])


tuples = ()
for vector in embedding_vectors:
    if not tuples:
        tuples = (vector, )
    else:
        tuples = tuples + (vector, )

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X = np.vstack(tuples)
pca = PCA(n_components=2)
result = pca.fit_transform(X)


agglo_clustering = AgglomerativeClustering(linkage='ward', affinity='euclidean', n_clusters=9)
agglo_clustering.fit(X, labels)
plt.scatter(X[:, 0], X[:, 1], c=agglo_clustering.labels_, cmap='rainbow')
plt.show()
