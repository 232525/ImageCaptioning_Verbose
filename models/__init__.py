from models.xlan import XLAN
from models.xtransformer import XTransformer
from models.pure_transformer import PureT

__factory = {
    'XLAN': XLAN,
    'XTransformer': XTransformer,
    'PureT': PureT,
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown caption model:", name)
    return __factory[name](*args, **kwargs)