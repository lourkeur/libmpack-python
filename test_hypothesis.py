from hypothesis import given
from strategies import *
import unittest

import mpack


class TestMpack(unittest.TestCase):
    @given(scalar_formats)
    def test_pack_unpack(self, x):
        packed_obj, obj = x
        unpacker = mpack.Unpacker()
        print("unpacking %r..." % packed_obj)
        unpacked_obj, n = unpacker(packed_obj)
        self.assertEqual(n, len(packed_obj))
        print("comparing...")
        self.assertEqual(unpacked_obj, obj)

    @unittest.skip("hangs and requires the process to be killed manually")
    def test_unpacking_c1(self):
        u = mpack.Unpacker()
        u(b"\xc1")
