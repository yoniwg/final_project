import os

import nltk

from vectorizer import TerminalVectorizer, RulesVectorizer

nltk.download('treebank')
nltk.download('tagsets')
from sklearn.linear_model import LogisticRegression

from sentences_file_reader import get_treebank


class Driver:
    _training_treebank_file = os.getcwd() + '/data/heb-ctrees.train'
    _gold_treebank_file = os.getcwd() + '/data/heb-ctrees.gold'

    _terminal_vectorizer = TerminalVectorizer()
    _terminal_trainer = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200)

    _rules_vectorizer = RulesVectorizer()
    _rules_trainer = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200, verbose=1)

    def drive(self):
        # self._train_terminals()
        self._train_rules()

    def _train_terminals(self):
        X, y = self._terminal_vectorizer.build_X_y(get_treebank(self._training_treebank_file))
        print("start training")
        self._terminal_trainer.fit(X, y)
        X, y = self._terminal_vectorizer.build_X_y(get_treebank(self._gold_treebank_file), False)
        print("start test for terminals")
        print(self._terminal_trainer.score(X,y))
        print(self._terminal_trainer.predict_proba(X[0:10]))

    def _train_rules(self):
        X, y = self._rules_vectorizer.build_X_y(get_treebank(self._training_treebank_file))
        print("start training")
        self._rules_trainer.fit(X, y)
        X, y = self._rules_vectorizer.build_X_y(get_treebank(self._gold_treebank_file), False)
        print("start test for rules")
        print(self._rules_trainer.score(X,y))
        print(self._terminal_trainer.predict_proba(X[0:10]))

