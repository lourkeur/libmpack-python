from hypothesis import given
from strategies import *

from mpack import Packer, Unpacker


@given(fixmap(keys=all_scalar, values=all_scalar))
def packer_correct(x):
    packed_obj, obj = x
    unpack = Unpacker()
    pack = Packer()
    assert pack(obj) == packed_obj
    assert unpack(packed_obj) == obj

packer_correct()
