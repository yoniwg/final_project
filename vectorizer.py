from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer

from features_builders import TerminalsFeatureBuilder, RulesFeatureBuilder
from util.transliteration import heb_tags, _trans_symbols


class TerminalVectorizer:
    # _eng_tag_set = list(nltk.data.load('help/tagsets/upenn_tagset.pickle').keys())
    _heb_tag_set = sorted(heb_tags)

    _terminal_feature_builder = TerminalsFeatureBuilder(heb_tags, _trans_symbols)
    _terminal_feature_vectorizer = DictVectorizer()
    _treebank = None
    _y = []
    _X = None


    def _get_all_terminals_dicts(self):
        for idx, tree in enumerate(self._treebank):
            print("\rvectorizing tree #{}".format(idx), end='')
            for dic in self._get_one_tree_terminals_dicts(tree.pos()):
                yield dic
        print()

    def _get_one_tree_terminals_dicts(self, terminal_nodes):
        for idx in range(len(terminal_nodes)):
            self._y.append(terminal_nodes[idx][1])
            yield self._terminal_feature_builder.create_features_list_for_node(terminal_nodes, idx)

    def build_X_y(self, treebank, fit=True):
        self._treebank = treebank
        self._y = []
        self._X = None
        if fit:
            X = self._terminal_feature_vectorizer.fit_transform(self._get_all_terminals_dicts())
        else:
            X = self._terminal_feature_vectorizer.transform(self._get_all_terminals_dicts())
        return X, self._y

    def build_one_vector(self, sentence, curr_idx, prev_probs, prev2_probs):
        return self._terminal_feature_vectorizer.transform(
            self._terminal_feature_builder.create_features_list_for_terminal(sentence, curr_idx, prev_probs, prev2_probs))


class RulesVectorizer:
    _rules_feature_builder = RulesFeatureBuilder()
    _rules_feature_vectorizer = DictVectorizer()
    _treebank = None
    _y = []
    _X = None


    def _get_all_rules_dicts(self):
        for idx, tree in enumerate(self._treebank):
            tree.chomsky_normal_form()
            tree.collapse_unary()
            print("\rvectorizing tree #{}".format(idx), end='')
            for dic in self._get_one_tree_rules_dicts(tree.subtrees(lambda t: len(t) == 2)): # 0 is for the sole child of TOP
                yield dic
        print()

    def _get_one_tree_rules_dicts(self, subtrees):
        for tree in subtrees:
            self._y.append(tree.label())
            yield self._rules_feature_builder.create_features_list_for_nodes(tree[0], tree[1])

    def build_X_y(self, treebank, fit=True):
        self._treebank = treebank
        self._y = []
        self._X = None
        if fit:
            X = self._rules_feature_vectorizer.fit_transform(self._get_all_rules_dicts())
        else:
            X = self._rules_feature_vectorizer.transform(self._get_all_rules_dicts())
        return X, self._y

    def build_one_vector(self, l_node, r_node):
        return self._rules_feature_vectorizer.transform(
            self._rules_feature_builder.create_features_list_for_nodes(l_node,r_node))
