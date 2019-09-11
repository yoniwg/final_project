'''

A python script exemplifying the utility functions provided.

Usage
-----

In your Anaconda terminal or other terminal where your python environment is available:

    python -m util.example

'''

from util.tree.builders import list_tree_from_sequence
from util.tree.get_yield import get_yield
from util.transliteration import to_heb
from util.tree.builders import node_tree_from_sequence
from util.tree.node import Node
from pprint import pprint
bracketed_notation_tree = '(TOP (FRAG (CC W) (S (S (PP (IN EL) (NP (PRP KK))) (VP (VB TEID)) (NP (H H) (NN HISJWRIH))) (yyCM yyCM) (CC W) (S (VP (VB IEIDW)) (NP (NP (CDT EFRWT) (NP (H H) (NN LWXMIM))) (SBAR (REL F) (S (S (VP (VB NFARW)) (PP (IN B) (NP (H H) (NN XIIM)))) (CC W) (S (VP (VB HCLIXW)) (VP (VB LBCE) (NP (AT AT) (NP (NN MFIMTM)))) (PP (INP (IN EL) (IN AP)) (NP (NP (H H) (NN ABDWT)) (ADJP (H H) (JJ KBDWT)))))))))) (yyDOT yyDOT)))'

#bracketed_notation_tree = '(TOP (S (ADVP (RB RCUB))))'
#bracketed_notation_tree = '(1 (2 2-leaf) (3 3-leaf-1 3-leaf2 3-leaf-3))'

list_tree = list_tree_from_sequence(bracketed_notation_tree)
print()
print(f'original tree as given in bracketed notation:\n\n{bracketed_notation_tree}\n\n')

tree_yield = get_yield(list_tree)
print(f'the yield extracted from the tree:\n\n{tree_yield}\n\n')

de_transliterated = list(map(to_heb, tree_yield))
print(f'the yeild, de-transliterated:\n\n{de_transliterated}\n\n')

reversed = list(map(lambda morpheme: morpheme[::-1], de_transliterated))[::-1]
print(f'finally, reversed for easy reading in LTR applications:\n\n{reversed}\n\n')

print('\n-----------------------------------------------------------------\n\n')

node_tree = node_tree_from_sequence(bracketed_notation_tree)
print('all arcs of this tree:\n\n')
pprint(node_tree.get_downward_arcs())
