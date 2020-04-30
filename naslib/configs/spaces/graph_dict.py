from naslib.search_spaces.core.operations import MixedOp
from naslib.search_spaces.nasbench1shot1.utils import PRIMITIVES

#TODO add attributes too
g = {0: {1: {'op': MixedOp(PRIMITIVES)},
         2: {}},
     1: {2: {'op': MixedOp(PRIMITIVES)}},
     2: {3: {'op': MixedOp(PRIMITIVES)}},
     3: {},
     4: {},
     5: {},
     6: {},
     7: {},
     8: {},
     9: {}
    }
