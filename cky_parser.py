import operator
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import thread
from itertools import product
from math import log
from typing import Type

from joblib import Parallel, delayed, dump, load
from nltk import Nonterminal
from nltk.tree import Tree
from sklearn.linear_model import LogisticRegression

from vectorizer import TerminalVectorizer, RulesVectorizer


class CKY_Parser:

    def __init__(self, terminal_trainer, rules_trainer, terminal_vecorizer, rules_vecorizer, grammar):
        self._terminal_trainer = terminal_trainer
        self._rules_trainer = rules_trainer
        self._terminal_vecorizer = terminal_vecorizer
        self._rules_vecorizer = rules_vecorizer
        self._terms_tags_list = terminal_trainer.classes_
        # self._rules_tags_list = rules_trainer.classes_
        self._grammar = grammar

    def create_tree(self, chart, chartI, chartJ):
        print("i={},j={}".format(chartI, chartJ))
        root_node = Tree('TOP', [])
        cell = chart[chartI][chartJ]
        maxProb = float("-inf")
        selected_rule = None
        for rule, info in cell.items():
            print("{}={}".format(rule, info))
            if info[0] > maxProb:
                maxProb = info[0]
                selected_rule = rule
        if selected_rule is None:
            return root_node

        self.add_node_to_tree(chart, selected_rule, cell, root_node)

        return root_node

    def add_node_to_tree(self, chart, selected_rule, next_cell, root):
        _, u_path, left, right = next_cell[selected_rule]

        new_node = Tree(selected_rule, [])
        root.append(new_node)
        for non_t in u_path:
            u_node = Tree(non_t[0], [])
            new_node.add_child(u_node)
            new_node = u_node

        left_rule, left_i, left_j = left
        if left_j == 0:
            new_node.append(Tree(left_rule, []))
            return

        next_left = chart[left_i][left_j]

        right_rule, right_i, right_j = right
        next_right = chart[right_i][right_j]

        self.add_node_to_tree(chart, left_rule, next_left, new_node)
        self.add_node_to_tree(chart, right_rule, next_right, new_node)

    def worker_job(self, chart, i, j):
        print('tid:{}, j={}'.format(threading.get_ident(),j))
        node = chart[i][j]
        for k in range(1, i):
            left_candidate = chart[k][j]
            right_candidate = chart[i - k][j + k]

            for l_non_t, (l_prob, _, ll_info, lr_info) in left_candidate.items():
                for r_non_t, (r_prob, _, rl_info, rr_info) in right_candidate.items():
                    print("{}x{}={}x{}".format(l_non_t, r_non_t, l_prob, r_prob))

                    # rules_res = self._get_rules_probs_log_reg(
                    #     l_non_t, r_non_t, ll_info[0], lr_info[0], rl_info[0], rr_info[0], k, i - k)
                    # leaves_prob = l_prob + r_prob
                    # for idx, prob in sorted(enumerate(rules_res), key=operator.itemgetter(1), reverse=True)[:5]:
                    #     self.fill_node(node, self._rules_tags_list[idx], prob, leaves_prob, (l_non_t, k, j),
                    #                    (r_non_t, i - k, j + k))

                    prods = self._get_rules_probs_grammar(l_non_t, r_non_t)

                    leaves_prob = l_prob + r_prob
                    for prod in prods:
                        self.fill_node(node, prod.lhs(), prod.prob(), leaves_prob, (l_non_t, k, j),
                                       (r_non_t, i - k, j + k))

        chart[i][j] = dict(sorted(node.items(), key=operator.itemgetter(1, 0), reverse=True)[:5])

    def _get_rules_probs_log_reg(self, l_non_t, r_non_t, ll, lr, rl, rr, l_lvs, r_lvs):
        vec = self._rules_vecorizer.build_one_vector_exp(
            l_non_t, r_non_t, ll, lr, rl, rr, l_lvs, r_lvs
        )
        return self._rules_trainer.predict_proba(vec)[0]

    def _get_rules_probs_grammar(self, l_non_t, r_non_t):
        l_NonT = Nonterminal(l_non_t)
        r_NonT = Nonterminal(r_non_t)
        return [
            prod for prod in self._grammar.productions(None, l_NonT)
            if prod.rhs() == (l_NonT, r_NonT)
        ]

    def parse(self, sentence):
        lengh = len(sentence)
        chart = [[dict() for j in range(lengh + 1)] for i in range(lengh + 1)]

        # init
        prev_probs = [0] * len(self._terms_tags_list)
        prev2_probs = [0] * len(self._terms_tags_list)
        for idx in range(lengh):
            terminal_vec = self._terminal_vecorizer.build_one_vector(sentence, idx, prev_probs, prev2_probs)
            terminal_res = self._terminal_trainer.predict_proba(terminal_vec)[0]
            prev2_probs = prev_probs
            prev_probs = terminal_res
            node = chart[1][idx + 1]
            for p_idx, prob in sorted(enumerate(terminal_res), key=operator.itemgetter(1), reverse=True)[:20]:
            # for p_idx, prob in enumerate(terminal_res):
                self.fill_node(node, self._terms_tags_list[p_idx], prob,
                               0, (sentence[idx], idx + 1, 0), ("", 0, 0))

        # main loop
        for i in range(2, lengh + 1):
            print('i={}'.format(i))
            Parallel(n_jobs=4, prefer='threads')(delayed(self.worker_job)(chart, i, j) for j in range(1, lengh + 2 - i))
        # create tree
        return self.create_tree(chart, lengh, 1)

    def fill_node(self, node, source_rule, prob, leaves_prob, l_child, r_child):
        new_prob = -log(prob) + leaves_prob
        if source_rule not in node or new_prob > node[source_rule][0]:
            node[source_rule] = (new_prob, tuple(), l_child, r_child)

    def parse2(self, sentence):
        table = [[defaultdict(float) for _ in sentence] for _ in sentence]
        nodes_table = [[defaultdict(lambda: Tree('')) for _ in sentence] for _ in sentence]
        prev_probs = [0] * len(self._terms_tags_list)
        prev2_probs = [0] * len(self._terms_tags_list)
        for j in range(0, len(sentence)):
            terminal_vec = self._terminal_vecorizer.build_one_vector(sentence, j, prev_probs, prev2_probs)
            terminal_res = self._terminal_trainer.predict_proba(terminal_vec)
            prev2_probs = prev_probs
            prev_probs = terminal_res
            for idx, prob in enumerate(terminal_res):
                table[j][j][self._terms_tags_list[idx]] = prob
                nodes_table[j - 1][j][sentence[j]] = Tree(sentence[j])
            for i in reversed(range(j)):
                for k in range(i, j):
                    for pair in set(product(table[i][k], table[k + 1][j])):
                        rules_probs = self._rules_trainer.predict_proba(self._rules_vecorizer.build_one_vector(
                            nodes_table[i][k][pair[0]], nodes_table[k + 1][j][pair[1]]))
                        for idx, prob in enumerate(rules_probs):
                            l_rule = self._rules_tags_list[idx]
                            new_prob = prob * table[i][k][pair[0]] * table[k + 1][j][pair[1]]
                            if table[i][j][l_rule] < new_prob:
                                table[i][j][l_rule] = new_prob
                                node = Tree(l_rule.label())
                                node.append(nodes_table[i][k][pair[0] if isinstance(pair[0], str) else pair[0].label()])
                                node.append(nodes_table[i][k][pair[1] if isinstance(pair[1], str) else pair[1].label()])
                                nodes_table[i][j][l_rule.label()] = node

        return nodes_table[0][len(sentence) - 1]['TOP']
