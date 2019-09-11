import os

import nltk
nltk.download('treebank')
nltk.download('tagsets')
from sklearn.linear_model import LogisticRegression

from features_builders import TerminalsFeatureBuilder
from sentences_file_reader import get_treebank
from util.transliteration import heb_tags, _trans_symbols


class Driver:
    _training_treebank_file = os.getcwd() + '/data/heb-ctrees.train'
    _gold_treebank_file = os.getcwd() + '/data/heb-ctrees.gold'
    _terminal_trainer = LogisticRegression(solver='sag', multi_class='multinomial')
    _eng_tag_set = list(nltk.data.load('help/tagsets/upenn_tagset.pickle').keys())
    _heb_tag_set = heb_tags
    _terminal_feature_builder = TerminalsFeatureBuilder(heb_tags, _trans_symbols)
    _terminal_feature_matrix = (list(), list())

    def _read_train_file(self):
        self._training_treebank = get_treebank(self._training_treebank_file)

    def _train(self):
        for tree in self._training_treebank:
            self._train_terminals(tree.pos())
            self._train_non_terminals(tree)

    def _train_terminals(self, terminal_nodes):
        for idx in range(len(terminal_nodes)):
            vec_features = self._terminal_feature_builder.create_features_list_for_node(terminal_nodes, idx)
            self._terminal_feature_matrix[0].append(vec_features)
            self._terminal_feature_matrix[1].append(terminal_nodes[idx][1])

    def _train_non_terminals(self, tree):
        pass


driver = Driver()
driver._read_train_file()
driver._train()
pass