from shortest_path_re import ShortestPathRE


def test_extract_sentence():
    sentence = u'''This thesis defines the clinical_characteristics of amyloid disease.'''
    e1 = u'''thesis'''
    e2 = u'''clinical_characteristics'''
    spre = ShortestPathRE().en_lang()
    sp = spre.search_shortest_dep_path(e1=e1, e2=e2, sentence=sentence)
    assert sp == ['thesis', 'defines', 'clinical_characteristics']


