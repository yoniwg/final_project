
def __find_unaries(grammar ,source_rule, unaries):
    path_tup, source_prob = unaries[source_rule]
    new_rules = set()
    prods = grammar.productions(rhs=source_rule)
    for prod in prods:
        if len(prod.rhs()) == 1 and prod.lhs().symbol() != 'TOP':
            prod_tag = prod.lhs()
            new_prob = source_prob + prod.logprob()
            if prod_tag not in path_tup and (prod_tag not in unaries or new_prob > unaries[prod_tag][1]):
                unaries[prod_tag] = (path_tup + (source_rule,), new_prob)
                new_rules.add(prod_tag)
    for rule in new_rules:
        __find_unaries(grammar, rule, unaries)
    return unaries


def init_unaries_dict(grammar):
    dict = {}
    for tag in grammar._categories:
        if tag.symbol() != 'TOP':
            unaries = __find_unaries(grammar, tag, {tag: (tuple(), 0)})
            if len(unaries) > 1:
                dict[tag] = unaries
    return dict
