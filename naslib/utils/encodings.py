from enum import Enum


class EncodingType(Enum):
    ADJACENCY_ONE_HOT = "adjacency_one_hot"
    ADJACENCY_MIX = 'adjacency_mix'
    PATH = "path"
    GCN = "gcn"
    BONAS = "bonas"
    SEMINAS = "seminas"
    COMPACT = 'compact'
