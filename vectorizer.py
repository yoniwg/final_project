from joblib import dump, load
from sklearn.feature_extraction import DictVectorizer

from features_builders import TerminalsFeatureBuilder, RulesFeatureBuilder
from util.transliteration import _trans_symbols

_terminals_vec_file = "data/term_train_vec"
_rules_vec_file = "data/rules_train_vec"


class TerminalVectorizer:

    def __init__(self, tags_list) -> None:
        self.tags_list = tags_list
        self._terminal_feature_builder = TerminalsFeatureBuilder(tags_list, _trans_symbols)
        self._terminal_feature_vectorizer = DictVectorizer()

    def _get_all_terminals_dicts(self):
        for idx, tree in enumerate(self._treebank):
            print("\rvectorizing tree #{}".format(idx), end='')
            for dic in self._get_one_tree_terminals_dicts(tree.pos()):
                yield dic
        print()
        print("fitting dicts -> vector")

    def _get_one_tree_terminals_dicts(self, terminal_nodes):
        for idx in range(len(terminal_nodes)):
            self._y.append(terminal_nodes[idx][1])
            dic = self._terminal_feature_builder.create_features_list_for_node(terminal_nodes, idx)
            yield dic

    def build_X_y(self, treebank, fit=True):
        self._treebank = treebank
        self._y = []
        self._X = None
        if fit:
            self._X = self._terminal_feature_vectorizer.fit_transform(self._get_all_terminals_dicts())
            dump(self._terminal_feature_vectorizer, _terminals_vec_file)
        else:
            self._X = self._terminal_feature_vectorizer.transform(self._get_all_terminals_dicts())
        print("Total vectors: {}x{}".format(len(self._y), self._X[0].shape[1]))
        return self._X, self._y

    def fit_from_file(self):
        self._terminal_feature_vectorizer = load(_terminals_vec_file)

    def build_one_vector(self, sentence, curr_idx, prev_probs, prev2_probs):
        return self._terminal_feature_vectorizer.transform(
            self._terminal_feature_builder.create_features_list_for_terminal(sentence, curr_idx, prev_probs, prev2_probs))


class RulesVectorizer:
    _rules_feature_builder = RulesFeatureBuilder()
    _rules_feature_vectorizer = DictVectorizer()

    def _get_all_rules_dicts(self):
        for idx, tree in enumerate(self._treebank):
            tree.chomsky_normal_form()
            tree.collapse_unary()
            print("\rvectorizing tree #{}".format(idx), end='')
            for dic in self._get_one_tree_rules_dicts(tree.subtrees(lambda t: len(t) == 2)):
                yield dic
        print()
        print("fitting dicts -> vector")

    def _get_one_tree_rules_dicts(self, subtrees):
        for tree in subtrees:
            self._y.append(tree.label())
            dic = self._rules_feature_builder.create_features_list_for_nodes(tree[0], tree[1])
            yield dic

    def build_X_y(self, treebank, fit=True):
        self._treebank = treebank
        self._y = []
        self._X = None
        if fit:
            self._X = self._rules_feature_vectorizer.fit_transform(self._get_all_rules_dicts())
            dump(self._rules_feature_vectorizer, _rules_vec_file)
        else:
            self._X = self._rules_feature_vectorizer.transform(self._get_all_rules_dicts())

        print("Total vectors: {}x{}".format(len(self._y), self._X[0].shape[1]))
        return self._X, self._y

    def fit_from_file(self):
        self._rules_feature_vectorizer = load(_rules_vec_file)

    def build_one_vector(self, l_node, r_node):
        return self._rules_feature_vectorizer.transform(
            self._rules_feature_builder.create_features_list_for_nodes(l_node, r_node))

    def build_one_vector_exp(self, l_non_t, r_non_t, ll, lr, rl, rr, l_lvs, r_lvs):
        if not lr:
            ll = ''
        if not rr:
            rl = ''
        return self._rules_feature_vectorizer.transform(
            self._rules_feature_builder.create_features_list_for_non_t(
                l_non_t, r_non_t, ll, lr, rl, rr, l_lvs, r_lvs
            )
        )
