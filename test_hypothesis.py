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
                unpack = mpack.Unpacker(ext=strategies.ext_unpack)
                unpacked_obj, n = unpack(packed_obj)
                self.assertEqual(n, len(packed_obj))
                self.assertEqual(unpacked_obj, obj)
            with self.subTest("unpack(pack(unpack(packed_obj))) == obj"):
                pack = mpack.Packer(ext=strategies.ext_pack)
                unpack = mpack.Unpacker(ext=strategies.ext_unpack)
                packed_obj = pack(unpacked_obj)
                unpacked_obj, n = unpack(packed_obj)
                self.assertEqual(n, len(packed_obj))
                self.assertEqual(unpacked_obj, obj)

    @given(strategies.msg(types=('request',)))
    def test_unpack_request(self, x):
        # TODO: rules based stateful testing
        packed_msg, msg = x
        msg_type, msg_id, method, params = msg
        s = mpack.Session()
        self.assertEqual(s.receive(packed_msg), (len(packed_msg), msg_type, method, params, msg_id))

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

    def test_packing_none_with_ext(self):
        pack = mpack.Packer(ext=lambda x: (ord('c'), b'afebabe'))
        self.assertEqual(pack(None), b"\xc0")


if __name__ == '__main__':
    unittest.main()
