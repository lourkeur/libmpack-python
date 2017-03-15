from hypothesis import given
from strategies import *

import mpack


@given(all_types)
def test_pack_unpack(x):
    packed_obj, obj = x
    unpacker = mpack.Unpacker()
    print("unpacking %r..." % packed_obj)
    unpacked_obj, n = unpacker(packed_obj)
    assert n == len(packed_obj)
    print("comparing...")
    assert unpacked_obj == obj
