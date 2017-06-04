import sys
import unittest

class SubTestMixin(object):
    if sys.version_info < (3,4):
        from contextlib import contextmanager
        @contextmanager
        def subTest(self, msg=None, **params):
            yield


if (3,) < sys.version_info < (3, 2):
    def callable(obj):
        return hasattr(obj, '__call__')
else:
    callable = callable


if not (3,) < sys.version_info < (3, 5):
    def bfmt(B, *args):
        return B % args
else:
    def bfmt(B, *args):
        acc = bytearray()
        args = iter(args)
        pos = 0
        for pos2 in iter(lambda: B.find(ord('%'), pos), -1):
            acc += B[pos:pos2]
            spec, pos = B[pos2+1], pos2 + 2
            if spec == ord('c'):
                acc.append(next(args))
            else:
                assert spec == ord('s')
                acc += next(args)
        acc += B[pos:]
        return bytes(acc)
