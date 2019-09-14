import os

import nltk
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer

from util.inc_csr_matrix import IncrementalCSRMatrix
from vectorizer import TerminalVectorizer

nltk.download('treebank')
nltk.download('tagsets')
from sklearn.linear_model import LogisticRegression

from features_builders import TerminalsFeatureBuilder
from sentences_file_reader import get_treebank
from util.transliteration import heb_tags, _trans_symbols


class Driver:
    _training_treebank_file = os.getcwd() + '/data/heb-ctrees_min.train'
    _gold_treebank_file = os.getcwd() + '/data/heb-ctrees.gold'

    _terminal_vectorizer = TerminalVectorizer(get_treebank(_training_treebank_file))
    _terminal_trainer = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200)

    def drive(self):
        self._train_terminals()

    def _train_terminals(self):
        X, y = self._terminal_vectorizer.build_X_y(True)
        print("start training")
        self._terminal_trainer.fit(X, y)
        X, y = self._terminal_vectorizer.build_X_y(False)
        print("start test for terminals")
        print(self._terminal_trainer.score(X,y))
