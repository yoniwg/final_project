from collections import defaultdict

import nltk
from nltk import ProbabilisticProduction

UNKNOWN_T = nltk.Nonterminal('UNKOWN')


def add_unknowns(grammar):
    terminal_derivation_non_t = set([prod.lhs() for prod in grammar.productions() if prod.is_lexical()])
    terminal_derivation_probs = defaultdict(int)
    for term_prod in terminal_derivation_non_t:
        terminal_derivation_probs[term_prod] = sum(prod.prob() for prod in grammar.productions(lhs=term_prod) if prod.is_lexical())
        if terminal_derivation_probs[term_prod] > 1:

            terminal_derivation_probs[term_prod] = 1
    total_non_t_rules = defaultdict(lambda: { 'l': defaultdict(lambda: [0, 0.0]), 'r': defaultdict(lambda: [0, 0.0])})
    for prod in grammar.productions():
        rule = prod.lhs()
        if len(prod.rhs()) > 1:
            l_non_t, r_non_t = prod.rhs()
            prob_l_derives_t = terminal_derivation_probs[l_non_t]
            prob_r_derives_t = terminal_derivation_probs[r_non_t]
            total_non_t_rules[rule]['l'][l_non_t][0] += 1
            total_non_t_rules[rule]['l'][l_non_t][1] += prod.prob() * prob_r_derives_t
            total_non_t_rules[rule]['r'][r_non_t][0] += 1
            total_non_t_rules[rule]['r'][r_non_t][1] += prod.prob() * prob_l_derives_t
    for non_t_rule, probs in total_non_t_rules.items():
        for l_prob, findings in probs['l'].items():
            if findings[1]:
                grammar._productions.append(ProbabilisticProduction(non_t_rule, (l_prob, UNKNOWN_T), prob=findings[1]/findings[0]))
        for r_prob, findings in probs['r'].items():
            if findings[1]:
                grammar._productions.append(ProbabilisticProduction(non_t_rule, (r_prob, UNKNOWN_T), prob=findings[1]/findings[0]))
    grammar._calculate_indexes()