from typing import Iterable, List

from util.tree.builders import list_tree_from_sequence, node_tree_from_sequence
from util.tree.get_yield import get_yield
from nltk import Tree
from nltk.corpus import treebank


def get_flat_sentences(file) -> Iterable[List[str]]:
    with open(file, 'r') as fd:
        lines = fd.readlines()
        for line in lines:
            yield get_yield(list_tree_from_sequence(line))


def get_sentences_as_tree(file) -> Iterable[Tree]:
    with open(file, 'r') as train_set:
        for line in train_set:
            yield Tree.fromstring(line)


def get_treebank(file) -> treebank:
    return treebank.parsed_sents(file)