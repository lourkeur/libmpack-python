from hypothesis import given
import strategies
import unittest

import mpack


class TestMpack(unittest.TestCase):
    @given(strategies.everything())
    def test_pack_unpack(self, x):
        packed_obj, obj = x
        with self.subTest(packed_obj=packed_obj, obj=obj):
            with self.subTest("unpack(packed_obj) == obj"):
                unpack = mpack.Unpacker()
                unpacked_obj, n = unpack(packed_obj)
                self.assertEqual(n, len(packed_obj))
                self.assertEqual(unpacked_obj, obj)
            with self.subTest("unpack(pack(unpack(packed_obj))) == obj"):
                pack, unpack = mpack.Packer(), mpack.Unpacker()
                packed_obj = pack(unpacked_obj)
                unpacked_obj, n = unpack(packed_obj)
                self.assertEqual(n, len(packed_obj))
                self.assertEqual(unpacked_obj, obj)

    @unittest.skip("hangs and requires the process to be killed manually")
    def test_unpacking_c1(self):
        u = mpack.Unpacker()
        u(b"\xc1")


if __name__ == '__main__':
    unittest.main()
