from hypothesis import given
from strategies import *
import unittest

import mpack


class TestMpack(unittest.TestCase):
    @given(all_types)
    def test_pack_unpack(self, x):
        packed_obj, obj = x
        unpacker = mpack.Unpacker()
        print("unpacking %r..." % packed_obj)
        unpacked_obj, n = unpacker(packed_obj)
        self.assertEqual(n, len(packed_obj))
        print("comparing...")
        self.assertEqual(unpacked_obj, obj)
