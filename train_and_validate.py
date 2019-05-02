"""
Read and pre process SemEval 2010 Task 8 data set
"""
import re
import pandas as pd
import numpy as np
import joblib

from sklearn.svm import SVC
from nltk.tokenize import WhitespaceTokenizer
from shortest_path_re import ShortestPathRE
from flair_embeddings import FlairEmbeddingModels
from relation_types import RelationTypes

regex_no = r'\d'
tokenizer = WhitespaceTokenizer()
spre = ShortestPathRE().en_lang()
embeddings = FlairEmbeddingModels().en_lang()
rt = RelationTypes()


def replace_multi_word_entity(sentence, entity):
    """
    Replaces multi-word entities separated by a whitespace with an underscore
    """
    if sentence.find(entity) > -1:
        e_array = entity.split()
        e_new = e_array[0] + '_' + e_array[1]
        sentence = sentence.replace(entity, e_new)
    else:
        e_new = entity

    return sentence, e_new


def extract_sentence(text):
    sentence_match = re.search(r'".*"', text)
    sentence = sentence_match.group()

    entity1_match = re.search(r'<e1>.*<\/e1>', sentence)
    entity1 = entity1_match.group()
    entity2_match = re.search(r'<e2>.*<\/e2>', sentence)
    entity2 = entity2_match.group()

    # clean up special characters
    sentence = re.sub(r'"|<(\/)?e\d>', ' ', sentence)
    sentence = re.sub(r'\s{2,}', ' ', sentence)
    e1 = re.sub(r'<(\/)?e\d>', '', entity1)
    e2 = re.sub(r'<(\/)?e\d>', '', entity2)

    # search for multi word entities
    if len(e1.split()) > 1:
        sentence, e1 = replace_multi_word_entity(sentence, e1)
    if len(e2.split()) > 1:
        sentence, e2 = replace_multi_word_entity(sentence, e2)

    # replace hyphens with underscore to prevent tokenization
    sentence = sentence.replace('-', '_')
    e1 = e1.replace('-', '_')
    e2 = e2.replace('-', '_')

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


def load_training_data(file_name):
    data_columns = ['sp', 'label']
    features = pd.DataFrame(columns=data_columns)

    with open(file_name, "r") as f:
        data = f.readlines()
        for i, row in enumerate(data):
            if len(row) > 1:
                sent_start = tokenizer.tokenize(row)

                # check if sentence starts with a number to indicate a training sentence
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


def stack_embedding_tensors(embedding_vectors):
    tuples = ()
    for vector in embedding_vectors:
        if not tuples:
            tuples = (vector, )
        else:
            tuples = tuples + (vector, )

    X = np.vstack(tuples)

    return X


def get_embeddings(features):
    embedding_vectors = []
    labels = []

    for row in features.iterrows():
        sp_embeddings = embeddings.get_word_embeddings(row[1]['sp'])
        embedding_vectors.append(sp_embeddings)
        label = rt.label_to_number(row[1]['label'])
        labels.append(label)

    X = stack_embedding_tensors(embedding_vectors)

    return X, labels


def train_classifier(training_file):
    features = load_training_data(training_file)
    X, y = get_embeddings(features)#

    #clf = SGDClassifier(loss='squared_hinge', penalty='l1', alpha=1e-05, max_iter=100, tol=0.2)
    clf = SVC(kernel='rbf', C=100, gamma=0.01, decision_function_shape='ovo', probability=True)
    print(f'Start training...')
    clf.fit(X, y)
    joblib.dump(clf, 'models/svc_clf.joblib')


def validate_classifier():
    model = joblib.load('models/svc_clf.joblib')

    with open('data/SemEval2010/test.txt', 'r') as f:
        sentences = f.readlines()

    with open('data/SemEval2010/test_file_key.txt', 'r') as f:
        labels = f.readlines()

    positive_count = 0
    for i, row in enumerate(sentences):
        sentence, e1, e2 = extract_sentence(row)
        sp = spre.search_shortest_dep_path(e1, e2, sentence)

        if sp:
            sp_embeddings = embeddings.get_word_embeddings(sp)
            X_test = stack_embedding_tensors([sp_embeddings])
            actual_label = tokenizer.tokenize(labels[i])[1]

            result = model.predict(X_test)
            predicted_label = rt.number_to_label(result[0])

            if predicted_label == actual_label:
                positive_count += 1

    total_result = positive_count / len(sentences)
    print(f'Total result: {total_result}')


train_classifier('data/SemEval2010/train.txt')
validate_classifier()

