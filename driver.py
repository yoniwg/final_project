import os
from functools import reduce
from multiprocessing import Process
from threading import Thread

import nltk
from joblib import dump, load

from cky_parser import CKY_Parser
from util.transliteration import heb_tags
from vectorizer import TerminalVectorizer, RulesVectorizer

nltk.download('treebank')
nltk.download('tagsets')
from sklearn.linear_model import LogisticRegression

from sentences_file_reader import get_treebank


def prepare_training_tree(treebank):
    cnf_tree_bank_list = []
    terminals_tags_rules = set()
    for idx, tree in enumerate(treebank):
        tree.chomsky_normal_form(horzMarkov=2)
        tree.collapse_unary(collapsePOS = True)
        cnf_tree_bank_list.append(tree)
        for terminal,tag in tree.pos():
            terminals_tags_rules.add(tag)
    return cnf_tree_bank_list, list(sorted(terminals_tags_rules))

class Driver:
    TEST = False

    _training_treebank_file = os.getcwd() + '/data/heb-ctrees.train'
    _training_treebank_file_min = os.getcwd() + '/data/heb-ctrees_min.train'
    _gold_treebank_file = os.getcwd() + '/data/heb-ctrees.gold'


    def __init__(self) -> None:
        self._treebank, self._terms_tags_list = prepare_training_tree(get_treebank(self._training_treebank_file))
        self._gold_treebank, self._gold_terms_tags_list = prepare_training_tree(get_treebank(self._gold_treebank_file))
        assert any(map(lambda tag: tag in self._terms_tags_list, self._gold_terms_tags_list)), "No match of tag lists"
        self._terminal_vectorizer = TerminalVectorizer(self._terms_tags_list)
        self._terminal_trainer = LogisticRegression(solver='lbfgs', multi_class='multinomial',
                                                    max_iter=100, verbose=2, n_jobs=4)
        self._terminals_res_file = "data/term_train_res"

        self._rules_vectorizer = RulesVectorizer()
        self._rules_trainer = LogisticRegression(solver='lbfgs', multi_class='multinomial',
                                                 max_iter=500, verbose=2, n_jobs=4)
        self._rules_res_file = "data/rules_train_res"
        self._grammar_file = "data/grammar_res"

    def drive(self):
        train_terms = Process(target=self._train_terminals)
        # train_rules = Process(target=self._train_rules)
        # train_terms.start()
        # self._train_terminals()
        self._train_rules_grammar()
        # train_rules.start()
        # train_terms.join()
        # train_rules.join()
        self._train_from_files()
        self._test_gold()

    def _train_terminals(self):
        X, y = self._terminal_vectorizer.build_X_y(self._treebank)
        print("start training")
        self._terminal_trainer.fit(X, y)
        print("saving terminals training result to {}".format(self._terminals_res_file))
        dump(self._terminal_trainer, self._terminals_res_file)
        assert list(self._terminal_trainer.classes_) == self._terms_tags_list
        if self.TEST:
            print("start test for terminals")
            X, y = self._terminal_vectorizer.build_X_y(self._gold_treebank, False)
            print(self._terminal_trainer.score(X, y))

    def _train_rules(self):
        X, y = self._rules_vectorizer.build_X_y(self._treebank)
        print("start training")
        self._rules_trainer.fit(X, y)
        print("saving terminals training result to {}".format(self._terminals_res_file))
        dump(self._rules_trainer, self._rules_res_file)
        if self.TEST:
            print("start test for rules")
            X, y = self._rules_vectorizer.build_X_y(self._gold_treebank, False)
            print(self._rules_trainer.score(X, y))

    def _train_rules_grammar(self):
        self._grammar = nltk.induce_pcfg(nltk.Nonterminal('TOP'), reduce(lambda a,b:a+b, map(lambda t: t.productions(), self._treebank)))
        # dump(self._grammar, self._grammar_file)

    def _train_from_files(self):
        print("loading dump files")
        self._terminal_vectorizer.fit_from_file()
        self._terminal_trainer = load(self._terminals_res_file)
        if self.TEST:
            print("start test for terminals")
            X, y = self._terminal_vectorizer.build_X_y(self._gold_treebank, False)
            print(self._terminal_trainer.score(X, y))
        # self._rules_vectorizer.fit_from_file()
        # self._rules_trainer = load(self._rules_res_file)
        if self.TEST:
            print("start test for rules")
            X, y = self._rules_vectorizer.build_X_y(self._gold_treebank, False)
            print(self._rules_trainer.score(X, y))

    def _test_gold(self):
        cky_parser = CKY_Parser(self._terminal_trainer,
                                self._rules_trainer,
                                self._terminal_vectorizer,
                                self._rules_vectorizer,
                                self._grammar)
        for tree in get_treebank(self._gold_treebank_file):
            parsed_tree = cky_parser.parse(tree.leaves())
            parsed_tree.un_chomsky_normal_form()
            with open('data/out_gold', 'a') as fo:
                print(parsed_tree, file=fo)


