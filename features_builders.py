
class TerminalsHandler:
    def __init__(self, rules_list, delimiters_list, num_from_begin, num_from_end):
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

        return dic.values()

    def create_features_list_for_node(self, terminals_list, node_index):
        uncle = None
        uncle2 = None
        if node_index > 0:
            rule, terminal = terminals_list[node_index - 1]
            uncle = rule

        if node_index > 1:
            rule, terminal = terminals_list[node_index - 2]
            uncle2 = rule

        pre_delimiter, post_delimiter = self.find_delimiters_place_around_inex(node_index, terminals_list)

        return self.create_features_list(uncle, uncle2, node_index - pre_delimiter, post_delimiter - node_index)

    def create_terminals_list_for_tree(self, root):

        if root.tag != "TOP":
            print("ERROR: This isn't root")
            return None

        node = root.children[0]
        terminals_list = list()
        self.get_terminals(node, terminals_list)

        return terminals_list

    def get_terminals(self, node, terminals_list):
        if len(node.children) == 0:
            terminals_list.append((node.parent.tag, node.tag))
            return
        self.get_terminals(node.children[0], terminals_list)

        if len(node.children) == 2:
            self.get_terminals(node.children[1], terminals_list)

    def find_delimiters_place_around_inex(self, index, terminals_list):
        before_index = -1
        after_index = len(terminals_list)

        for idx, val in enumerate(terminals_list):
            rule, terminal = val
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


class RulesHandler:
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
