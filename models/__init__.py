from models.updown import UpDown
from models.xlan import XLAN
from models.xtransformer import XTransformer
from models.pure_transformer import RawTransformer, PureT
from models.m2_transformer import M2Transformer

__factory = {
    'UpDown': UpDown,
    'XLAN': XLAN,
    'XTransformer': XTransformer,
    'RawTransformer': RawTransformer,
    'PureT': PureT,
    'M2Transformer': M2Transformer,
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown caption model:", name)
    return __factory[name](*args, **kwargs)