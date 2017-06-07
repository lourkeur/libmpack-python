from hypothesis import given
import unittest

import mpack

from . import compat, strategies, statemachines


class TestMpack(unittest.TestCase, compat.SubTestMixin):
    @given(strategies.everything())
    def test_pack_unpack(self, x):
        packed_obj, obj = x
        with self.subTest(packed_obj=packed_obj, obj=obj):
            with self.subTest("unpack(packed_obj) == obj"):
                unpack = mpack.Unpacker(ext=strategies.ext_unpack)
                unpacked_obj, n = unpack(packed_obj)
                self.assertEqual(n, len(packed_obj))
                self.assertEqual(unpacked_obj, obj)
            with self.subTest("unpack(pack(unpack(packed_obj))) == obj"):
                pack = mpack.Packer()
                unpack = mpack.Unpacker(ext=strategies.ext_unpack)
                packed_obj = pack(unpacked_obj)
                unpacked_obj, n = unpack(packed_obj)
                self.assertEqual(n, len(packed_obj))
                self.assertEqual(unpacked_obj, obj)

    def test_unpacking_c1(self):
        unpack = mpack.Unpacker()
        with self.assertRaises(mpack.MpackException):
            unpack(b"\xc1")

    @unittest.skip('segfaults')
    def test_packing_with_ext_dict(self):
        pack = mpack.Packer(ext={})
        self.assertEqual(pack(None), b"\xc0")

    def test_unpacking_with_ext_dict(self):
        unpack = mpack.Unpacker(ext={})
        self.assertEqual(unpack(b"\xc0"), (None, 1))

TestMpackRPC = statemachines.RPCSession.TestCase


if __name__ == '__main__':
    unittest.main()
