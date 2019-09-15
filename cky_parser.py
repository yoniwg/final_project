import operator
from typing import Type

from nltk.tree import Tree
from sklearn.linear_model import LogisticRegression

from vectorizer import TerminalVectorizer, RulesVectorizer


class CKY_Parser:
    _terminal_trainer: LogisticRegression
    _rules_trainer: LogisticRegression
    _terminal_vecorizer: TerminalVectorizer
    _rules_vecorizer: RulesVectorizer
    _tags_set: list

    def create_tree(self, chart, chartI, chartJ):
        root_node = Tree('TOP')
        cell = chart[chartI][chartJ]
        maxProb = float("-inf")
        selected_rule = None
        for rule, info in cell.items():
            if info[0] > maxProb:
                maxProb = info[0]
                selected_rule = rule
        if not selected_rule:
            root_node.append(Tree("")) # in order to have some child
            return root_node

        self.add_node_to_tree(chart, selected_rule, cell, root_node)

        return root_node

    def add_node_to_tree(self, chart, selected_rule, next_cell, root):
        _, u_path, left, right = next_cell[selected_rule]

        new_node = Tree(selected_rule[0])
        root.append(new_node)
        for non_t in u_path:
            u_node = Tree(non_t[0])
            new_node.add_child(u_node)
            new_node = u_node

        left_rule, left_i, left_j = left
        if left_j == 0:
            new_node.append(Tree(left_rule))
            return

        next_left = chart[left_i][left_j]

        right_rule, right_i, right_j = right
        next_right = chart[right_i][right_j]

        self.add_node_to_tree(chart, left_rule, next_left, new_node)
        self.add_node_to_tree(chart, right_rule, next_right, new_node)

    def parse(self, sentence):
        lengh = len(sentence)
        chart = [[dict() for j in range(lengh + 1)] for i in range(lengh + 1)]

        # init
        prev_probs = [0] * len(self._tags_set)
        prev2_probs = [0] * len(self._tags_set)
        for idx in range(lengh):
            terminal_vec = self._terminal_vecorizer.build_one_vector(sentence, idx, prev_probs, prev2_probs)
            terminal_res = self._terminal_trainer.predict_proba(terminal_vec)
            prev2_probs = prev_probs
            prev_probs = terminal_res
            node = chart[1][idx + 1]
            for prob_idx in range(len(terminal_res)):
                self.fill_node(node, self._tags_set[prob_idx], terminal_res[prob_idx],
                               0, (sentence[idx], idx + 1, 0), ("", 0, 0))

        # main loop
        for i in range(2, lengh + 1):
            for j in range(1, lengh + 2 - i):
                node = chart[i][j]
                for k in range(1, i):
                    firstNodeToCheck = chart[k][j]
                    secondNodeToCheck = chart[i - k][j + k]

                    for firstRule, (firstProb, _, _, _) in firstNodeToCheck.items():
                        for secondRule, (secondProb, _, _, _) in secondNodeToCheck.items():
                            rules_res = self._rules_trainer.predict_proba(self._rules_vecorizer.build_one_vector())
                            if rules_deriver is not None:
                                leavesProb = firstProb + secondProb
                                for source_rule, prob in rules_deriver.items():
                                    self.fill_node(node, source_rule, prob, leavesProb, (firstRule, k, j),
                                                   (secondRule, i - k, j + k))

                chart[i][j] = dict(sorted(node.items(), key=operator.itemgetter(1, 0), reverse=True)[:150])
        # create tree
        root_node = self.create_tree(chart, lengh, 1)

        self._transformer.detransform(root_node)
        remove_non_t_pref(root_node)
        return sequence_from_tree(root_node)

    def fill_node(self, node, source_rule, prob, leaves_prob, l_child, r_child):
        new_prob = prob + leaves_prob
        if source_rule not in node or new_prob > node[source_rule][0]:
            node[source_rule] = (new_prob, tuple(), l_child, r_child)