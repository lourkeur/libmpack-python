import os, unittest

import mpack


class TestMpack(unittest.TestCase):
    if not os.getenv("AVOID_SEGFAULT"):
        def test_packing_none_with_dict_ext(self):
            pack = mpack.Packer(ext={})
            self.assertEqual(pack(None), b"\xc0")

    def test_packing_1_with_dict_ext(self):
        pack = mpack.Packer(ext={})
        self.assertEqual(pack(1), b"\x01")

    def test_packing_none_with_list_ext(self):
        pack = mpack.Packer(ext=[])
        self.assertEqual(pack(None), b"\xc0")

    def test_unpacking_none_with_dict_ext(self):
        unpack = mpack.Unpacker(ext={})
        self.assertEqual(unpack(b"\xc0"), (None, 1))

    def test_unpacking_ext_with_dict_ext(self):
        unpack = mpack.Unpacker(ext={})
        self.assertEqual(unpack(b"\xd4\x01\x01"), (None, 3))

if __name__ == '__main__':
    unittest.main()
