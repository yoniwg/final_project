from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer

from features_builders import TerminalsFeatureBuilder
from util.transliteration import heb_tags, _trans_symbols


class TerminalVectorizer:
    # _eng_tag_set = list(nltk.data.load('help/tagsets/upenn_tagset.pickle').keys())
    _heb_tag_set = heb_tags

    _terminal_feature_builder = TerminalsFeatureBuilder(heb_tags, _trans_symbols)
    _terminal_feature_vectorizer = DictVectorizer()
    _y = []
    _X = None

    def __init__(self, treebank):
        self._treebank = treebank

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

    def build_X_y(self, fit):
        self._y = []
        self._X = None
        if fit:
            X = self._terminal_feature_vectorizer.fit_transform(self._get_all_terminals_dicts())
        else:
            X = self._terminal_feature_vectorizer.transform(self._get_all_terminals_dicts())
        return X, self._y