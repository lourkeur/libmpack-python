import unittest

import mpack


class TestMpack(unittest.TestCase):
    def test_packing_none_with_ext(self):
        pack = mpack.Packer(ext=lambda x: (ord('c'), b'afebabe'))
        self.assertEqual(pack(None), b"\xc0")

    def test_packing_nones_with_ext(self):
        pack = mpack.Packer(ext=lambda x: (ord('c'), b'afebabe'))
        self.assertEqual(pack([None, None]), b"\x92\xc0\xc0")

    def test_packing_none_without_ext(self):
        pack = mpack.Packer()
        self.assertEqual(pack(None), b"\xc0")

    def test_packing_nones_without_ext(self):
        pack = mpack.Packer()
        self.assertEqual(pack([None, None]), b"\x92\xc0\xc0")

    def test_unpacking_none_with_ext(self):
        unpack = mpack.Unpacker(ext=lambda code, data: (code, data))
        self.assertEqual(unpack(b"\xc0"), (None, 1))


if __name__ == '__main__':
    unittest.main()
