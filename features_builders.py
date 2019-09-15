import numpy

START_LABEL = '<S>'
END_LABEL = '<E>'


class TerminalsFeatureBuilder:
    def __init__(self, rules_list, delimiters_list, num_from_begin=20, num_from_end=20):
        self.rulesDict = {}
        for x in range(num_from_begin):
            self.rulesDict[str(x + 1) + "_FromBegin"] = 0
        for x in range(num_from_end):
            self.rulesDict[str(x + 1) + "_FromEnd"] = 0
        for i, val in enumerate(rules_list):
            self.rulesDict["Uncle_" + val] = 0
        for i, val in enumerate(rules_list):
            self.rulesDict["Uncle2_" + val] = 0

        self._max_begin = num_from_begin
        self._max_end = num_from_end
        self.rules_list = rules_list
        self._delims_set = set(delimiters_list)

    def create_features_list_for_node(self, terminals_list, node_index):
        uncle = None
        uncle2 = None
        if node_index > 0:
            terminal, rule = terminals_list[node_index - 1]
            uncle = rule

        if node_index > 1:
            terminal, rule = terminals_list[node_index - 2]
            uncle2 = rule

        pre_delimiter, post_delimiter = self._find_delimiters_place_around_index(node_index, terminals_list)

        merged_dict = self.rulesDict.copy()
        self._create_uncles_features(merged_dict, uncle, uncle2)
        self._create_position_features(merged_dict, node_index - pre_delimiter, post_delimiter - node_index)
        merged_dict.update(self._flat_word_features(list(map(lambda tup: tup[1], terminals_list)), node_index))

        return merged_dict

    def create_features_list_for_terminal(self, sentence, node_index, uncle_probs, uncle2_probs):

        pre_delimiter, post_delimiter = self._find_delimiters_place_around_index(node_index, sentence)

        merged_dict = self.rulesDict.copy()
        self._create_uncles_features(merged_dict, uncle_probs, uncle2_probs)
        self._create_position_features(merged_dict, node_index - pre_delimiter, post_delimiter - node_index)
        merged_dict.update(self._flat_word_features(sentence, node_index))

        return merged_dict

    def _create_uncles_features(self, dic, uncle, uncle2):
        if uncle is not None:
            dic["Uncle_" + uncle] = 1

        if uncle2 is not None:
            dic["Uncle2_" + uncle2] = 1

    def _create_uncles_probs_features(self, dic, uncle_probs, uncle2_probs):
        for idx, rule in enumerate(self.rules_list):
            dic["Uncle_" + rule] = uncle_probs[idx]
            dic["Uncle2_" + rule] = uncle2_probs[idx]

    def _create_position_features(self, dic, place_from_begin, place_from_end):

        if place_from_begin <= self._max_begin:
            dic[str(place_from_begin) + "_FromBegin"] = 1

        if place_from_end <= self._max_end:
            dic[str(place_from_end) + "_FromEnd"] = 1

    def _find_delimiters_place_around_index(self, index, terminals_list):
        before_index = -1
        after_index = len(terminals_list)

        for idx, val in enumerate(terminals_list):
            if isinstance(val, tuple):
                terminal, _ = val
            else:
                terminal = val
            if (idx < index) and self._is_delimiter(terminal):
                before_index = idx
            if (idx > index) and self._is_delimiter(terminal):
                after_index = idx
                return before_index, after_index

        return before_index, after_index

    def _is_delimiter(self, terminal):
        return terminal in self._delims_set

    def _flat_word_features(self, sentence, w_idx):
        cur_word = sentence[w_idx]
        if w_idx > 0:
            prev_word = sentence[w_idx - 1]
        else:
            prev_word = START_LABEL
        if w_idx < len(sentence) - 1:
            next_word = sentence[w_idx + 1]
        else:
            next_word = END_LABEL
        return {
            'index': w_idx,
            'is_first': w_idx == 0,
            'is_last': w_idx == len(sentence) - 1,
            'curr_is_lower': cur_word.islower(),
            'prev_is_lower': prev_word.islower(),
            'next_is_lower': next_word.islower(),
            'curr_is_upper': cur_word.isupper(),
            'prev_is_upper': prev_word.isupper(),
            'next_is_upper': next_word.isupper(),
            'first_upper': cur_word[0].isupper(),
            'prev_first_upper': prev_word[0].isupper(),
            'next_first_upper': next_word[0].isupper(),
            'is_digit': cur_word.isdigit(),
            'prev_is_digit': prev_word.isdigit(),
            'next_is_digit': next_word.isdigit(),
            'has_no_sign': cur_word.isalnum(),
            'prev_is_sign': all(map(lambda c: not c.isalnum(), prev_word)),
            'next_is_sign': all(map(lambda c: not c.isalnum(), next_word)),
            'prefix-1': cur_word[0],
            'prefix-2': cur_word[:2],
            'prefix-3': cur_word[:3],
            'suffix-1': cur_word[-1],
            'suffix-2': cur_word[-2:],
            'suffix-3': cur_word[-3:],
            'prev_prefix-1': prev_word[0],
            'prev_prefix-2': prev_word[:2],
            'prev_prefix-3': prev_word[:3],
            'prev_suffix-1': prev_word[-1],
            'prev_suffix-2': prev_word[-2:],
            'prev_suffix-3': prev_word[-3:],
            'next_prefix-1': next_word[0],
            'next_prefix-2': next_word[:2],
            'next_prefix-3': next_word[:3],
            'next_suffix-1': next_word[-1],
            'next_suffix-2': next_word[-2:],
            'next_suffix-3': next_word[-3:],
        }


class RulesFeatureBuilder:

    def create_features_list_for_nodes(self, left_node, right_node):
        left_rule = left_node.label()
        right_rule = right_node.label()

        left_left_rule = ''
        left_right_rule = ''
        if len(left_node) == 2:
            left_left_rule = left_node[0].label()
            left_right_rule = left_node[1].label()

        right_left_rule = ''
        right_right_rule = ''
        if len(right_node) == 2:
            right_left_rule = right_node[0].label()
            right_right_rule = right_node[1].label()

        left_height = left_node.height()
        right_height = right_node.height()

        return {
            'left-rule': left_rule,
            'right-rule': right_rule,
            'left_left_rule': left_left_rule,
            'left_right_rule': left_right_rule,
            'right_left_rule': right_left_rule,
            'right_right_rule':right_right_rule,
            'left_height': left_height,
            'right_height': right_height
        }

