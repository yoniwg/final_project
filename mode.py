from sys import argv

PURE_CKY_M = 'PURE_CKY_M'
TERMS_GRAMMAR_M = 'TERMS_GRAMMAR_M'
TERMS_RULES_M = 'TERMS_RULES_M'

TRAIN_ALL_M = 'TRAIN_ALL_M'
TRAIN_TERMS_M = 'TRAIN_TERMS_M'
TRAIN_RULES_M = 'TRAIN_RULES_M'
NO_TRAIN_M = 'NO_TRAIN_M'

if '--pure-cky' in argv:
    RUN_MODE = PURE_CKY_M
elif '--all-lr' in argv:
    RUN_MODE = TERMS_RULES_M
else:
    RUN_MODE = TERMS_GRAMMAR_M

if '--no-train' in argv:
    TRAIN_MODE = NO_TRAIN_M
elif '--train-terms' in argv:
    TRAIN_MODE = TRAIN_TERMS_M
elif '--train-rules' in argv:
    TRAIN_MODE = TRAIN_RULES_M
else:
    TRAIN_MODE = TRAIN_ALL_M

if '--percolate' in argv:
    UNARY_MODE = True
else:
    UNARY_MODE = False

if '--ignore_unknown' in argv:
    UNKOWN_MODE = False
else:
    UNKOWN_MODE = True
