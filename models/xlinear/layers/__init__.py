from models.xlinear.layers.low_rank import LowRank
from models.xlinear.layers.basic_att import BasicAtt
from models.xlinear.layers.sc_att import SCAtt

__factory = {
    'LowRank': LowRank,
    'BasicAtt': BasicAtt,
    'SCAtt': SCAtt,
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown layer:", name)
    return __factory[name](*args, **kwargs)