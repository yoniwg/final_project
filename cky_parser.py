import operator

from math import log
from time import time

from nltk import Nonterminal
from nltk.tree import Tree

from mode import TERMS_GRAMMAR_M, RUN_MODE, TERMS_RULES_M, PURE_CKY_M, UNARY_MODE, UNKOWN_MODE
from unariy_rules_handler import init_unaries_dict
from unknown_rules_handler import UNKNOWN_T


class CKY_Parser:

    def __init__(self, terminal_trainer, rules_trainer, terminal_vecorizer, rules_vecorizer, grammar):
        self._terminal_trainer = terminal_trainer
        self._rules_trainer = rules_trainer
        self._terminal_vecorizer = terminal_vecorizer
        self._rules_vecorizer = rules_vecorizer
        if RUN_MODE == TERMS_RULES_M or RUN_MODE == TERMS_GRAMMAR_M:
            self._terms_tags_list = terminal_trainer.classes_
        if RUN_MODE == TERMS_RULES_M:
            self._rules_tags_list = rules_trainer.classes_
        self._grammar = grammar
        if UNARY_MODE:
            self._unaries_dict = init_unaries_dict(grammar)

    def create_tree(self, chart, chartI, chartJ):
        root_node = Tree('TOP', [])
        cell = chart[chartI][chartJ]
        maxProb = float("-inf")
        selected_rule = None
        for rule, info in cell.items():
            if info[0] > maxProb:
                maxProb = info[0]
                selected_rule = rule
        if selected_rule is None:
            return root_node

        self.add_node_to_tree(chart, selected_rule, cell, root_node)

        return root_node

    def add_node_to_tree(self, chart, selected_rule, next_cell, root):
        _, u_path, left, right = next_cell[selected_rule]

        new_node = Tree(selected_rule.symbol(), [])
        root.append(new_node)
        for non_t in u_path:
            u_node = Tree(non_t.symbol(), [])
            new_node.append(u_node)
            new_node = u_node

        left_rule, left_i, left_j = left
        if left_j == 0:
            new_node.append(left_rule)
            return

        next_left = chart[left_i][left_j]

        right_rule, right_i, right_j = right
        next_right = chart[right_i][right_j]

        self.add_node_to_tree(chart, left_rule, next_left, new_node)
        self.add_node_to_tree(chart, right_rule, next_right, new_node)

    def _get_rules_probs_log_reg(self, l_non_t, r_non_t, ll, lr, rl, rr, l_lvs, r_lvs):
        vec = self._rules_vecorizer.build_one_vector_exp(
            l_non_t, r_non_t, ll, lr, rl, rr, l_lvs, r_lvs
        )
        return self._rules_trainer.predict_proba(vec)[0]

    def _get_rules_probs_grammar(self, l_non_t, r_non_t):
        return [
            prod for prod in self._grammar.productions(None, l_non_t)
            if prod.rhs() == (l_non_t, r_non_t)
        ]

    def parse(self, sentence):
        lengh = len(sentence)
        chart = [[dict() for j in range(lengh + 1)] for i in range(lengh + 1)]

        # init
        if RUN_MODE == PURE_CKY_M:
            for idx in range(lengh):

                terminal_res = [prod for prod in self._grammar.productions(None, sentence[idx])]
                node = chart[1][idx + 1]
                if not terminal_res and UNKOWN_MODE:
                    self.fill_node(node, UNKNOWN_T, log(0.0001), 0, (sentence[idx], idx + 1, 0), ("", 0, 0))
                else:
                    for prod in terminal_res:
                        self.fill_node(node, prod.lhs(), prod.prob(),
                                       0, (sentence[idx], idx + 1, 0), ("", 0, 0))
        else:
            prev_probs = [0] * len(self._terms_tags_list)
            prev2_probs = [0] * len(self._terms_tags_list)
            for idx in range(lengh):
                terminal_vec = self._terminal_vecorizer.build_one_vector(sentence, idx, prev_probs, prev2_probs)
                terminal_res = self._terminal_trainer.predict_proba(terminal_vec)[0]
                prev2_probs = prev_probs
                prev_probs = terminal_res
                node = chart[1][idx + 1]
                for p_idx, prob in enumerate(terminal_res):
                    self.fill_node_unaries(node, Nonterminal(self._terms_tags_list[p_idx]), log(prob),
                                   0, (sentence[idx], idx + 1, 0), ("", 0, 0))
        # main loop
        for i in range(2, lengh + 1):
            for j in range(1, lengh + 2 - i):
                node = chart[i][j]
                for k in range(1, i):
                    left_candidate = chart[k][j]
                    right_candidate = chart[i - k][j + k]
                    count = 0
                    for l_non_t, (l_prob, _, ll_info, lr_info) in left_candidate.items():
                        for r_non_t, (r_prob, _, rl_info, rr_info) in right_candidate.items():
                            count += 1
                            leaves_prob = l_prob + r_prob
                            if RUN_MODE == TERMS_RULES_M:
                                rules_res = self._get_rules_probs_log_reg(
                                    l_non_t, r_non_t, ll_info[0], lr_info[0], rl_info[0], rr_info[0], k, i - k)
                                for idx, prob in sorted(enumerate(rules_res), key=operator.itemgetter(1), reverse=True)[:30]:
                                    self.fill_node(node, Nonterminal(self._rules_tags_list[idx]), log(prob), leaves_prob, (l_non_t, k, j),
                                                   (r_non_t, i - k, j + k))
                            else:
                                prods = self._get_rules_probs_grammar(l_non_t, r_non_t)
                                for prod in prods:
                                    self.fill_node_unaries(node, prod.lhs(), prod.logprob(), leaves_prob, (l_non_t, k, j),
                                                   (r_non_t, i - k, j + k))
                chart[i][j] = dict(sorted(node.items(), key=operator.itemgetter(1, 0), reverse=True)[:100])

        # create tree
        return self.create_tree(chart, lengh, 1)

    def fill_node(self, node, source_rule, prob, leaves_prob, l_child, r_child):
        new_prob = prob + leaves_prob
        if source_rule not in node or new_prob > node[source_rule][0]:
            node[source_rule] = (new_prob, tuple(), l_child, r_child)

    def fill_node_unaries(self, node, source_rule, prob, leaves_prob, l_child, r_child):
        if not UNARY_MODE and source_rule in self._unaries_dict:
            for unary_top, (u_path, u_prob) in self._unaries_dict[source_rule].items():
                new_prob = leaves_prob + u_prob + prob
                if unary_top not in node or new_prob > node[unary_top][0]:
                    node[unary_top] = (new_prob, u_path, l_child, r_child)
        else:
            self.fill_node(node, source_rule, prob, leaves_prob, l_child, r_child)
