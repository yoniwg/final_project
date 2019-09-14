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

        self.mNumFromBegin = num_from_begin
        self.mNumFromEnd = num_from_end
        self.mDelimitersList = delimiters_list

    def get_feature_num(self):
        return len(self.rulesDict)

    def flat_word_features(self, sentence, w_idx):
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
            # 'word': cur_word,
            'index': w_idx,
            'is_first': w_idx == 0,
            'is_last': w_idx == len(sentence) - 1,
            # 'prev_word': prev_word,
            # 'next_word': next_word,
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

    def create_features_list(self, uncle, uncle2, place_from_begin, place_from_end):
        dic = self.rulesDict.copy()

        if uncle is not None:
            dic["Uncle_" + uncle] = 1

        if uncle2 is not None:
            dic["Uncle2_" + uncle2] = 1

        if place_from_begin <= self.mNumFromBegin:
            dic[str(place_from_begin) + "_FromBegin"] = 1

        if place_from_end <= self.mNumFromEnd:
            dic[str(place_from_end) + "_FromEnd"] = 1

        return dic

    def create_features_list_for_node(self, terminals_list, node_index):
        uncle = None
        uncle2 = None
        if node_index > 0:
            terminal, rule = terminals_list[node_index - 1]
            uncle = rule

        if node_index > 1:
            terminal, rule = terminals_list[node_index - 2]
            uncle2 = rule

        pre_delimiter, post_delimiter = self.find_delimiters_place_around_inex(node_index, terminals_list)

        merged = self.create_features_list(uncle, uncle2, node_index - pre_delimiter, post_delimiter - node_index)
        merged.update(self.flat_word_features(list(map(lambda tup: tup[1], terminals_list)), node_index))

        return merged

    def find_delimiters_place_around_inex(self, index, terminals_list):
        before_index = -1
        after_index = len(terminals_list)

        for idx, val in enumerate(terminals_list):
            terminal, rule = val
            if (idx < index) and self.is_delimiter(terminal):
                before_index = idx
            if (idx > index) and self.is_delimiter(terminal):
                after_index = idx
                return before_index, after_index

        return before_index, after_index

    def is_delimiter(self, terminal):
        for word in self.mDelimitersList:
            if terminal == word:
                return True

        return False


class RulesFeatureBuilder:
    def __init__(self, rules_list, max_deep_from_end):
        self.rulesDict = {}

        for x in range(max_deep_from_end):
            self.rulesDict[str(x + 1) + "_FromLeftEnd"] = 0

        for x in range(max_deep_from_end):
            self.rulesDict[str(x + 1) + "_FromRightEnd"] = 0

        for i, val in enumerate(rules_list):
            self.rulesDict["Left_" + val] = 0

        for i, val in enumerate(rules_list):
            self.rulesDict["Right_" + val] = 0

        for i, val in enumerate(rules_list):
            self.rulesDict["Left_Left_" + val] = 0

        for i, val in enumerate(rules_list):
            self.rulesDict["Left_Right_" + val] = 0

        for i, val in enumerate(rules_list):
            self.rulesDict["Right_Left_" + val] = 0

        for i, val in enumerate(rules_list):
            self.rulesDict["Right_Right_" + val] = 0

        self.mMaxDeepFromEnd = max_deep_from_end

    def create_features_list(self, left_rule, right_rule, left_left_rule, left_right_rule, right_left_rule,
                             right_right_rule, left_deep_from_end, right_deep_from_end):
        dic = self.rulesDict.copy()

        if left_rule is not None:
            dic["Left_" + left_rule] = 1

        if right_rule is not None:
            dic["Right_" + right_rule] = 1

        if left_left_rule is not None:
            dic["Left_Left_" + left_left_rule] = 1

        if left_right_rule is not None:
            dic["Left_Right_" + left_right_rule] = 1

        if right_left_rule is not None:
            dic["Right_Left_" + right_left_rule] = 1

        if right_right_rule is not None:
            dic["Right_Right_" + right_right_rule] = 1

        if left_deep_from_end <= self.mMaxDeepFromEnd:
            dic[str(left_deep_from_end) + "_FromLeftEnd"] = 1

        if right_deep_from_end <= self.mMaxDeepFromEnd:
            dic[str(right_deep_from_end) + "_FromRightEnd"] = 1

        return dic.values()

    def create_features_list_for_nodes(self, left_node, right_node):
        left_rule = left_node.tag
        right_rule = right_node.tag

        left_left_rule = None
        left_right_rule = None
        if len(left_node.children) == 2:
            left_left_rule = left_node.children[0].tag
            left_right_rule = left_node.children[1].tag

        right_left_rule = None
        right_right_rule = None
        if len(right_node.children) == 2:
            right_left_rule = right_node.children[0].tag
            right_right_rule = right_node.children[1].tag

        left_deep_from_end = self.get_deep(left_node)
        right_deep_from_end = self.get_deep(right_node)

        return self.create_features_list(left_rule, right_rule, left_left_rule, left_right_rule, right_left_rule,
                                         right_right_rule, left_deep_from_end, right_deep_from_end)

    def get_deep(self, node):
        return 3
        # TODO YONI
