"""hypothesis strategies for testing libmpack.

Each strategy's goal is to generate a value suitable for a particular format,
pack it using a slow but easily verified implementation of MessagePack, and
then return the packed representation tupled with the original value.
"""

from hypothesis.strategies import (none, booleans, integers, lists,
        dictionaries, recursive, one_of, text, binary)


_all_scalar = []

def _scalar_st(source, pack):
    """helper to make a strategy for a non-recursive format.

    Also auto-collects it into `all_scalar` (which see).
    """
    st = source.map(lambda v: (pack(v), v))
    _all_scalar.append(st)
    return st

nil = _scalar_st(
        source=none(),
        pack=lambda v: b"\xc0")
bool = _scalar_st(
        source=booleans(),
        pack=lambda v: b"\xc3" if v else b"\xc2")
positive_fixnum = _scalar_st(
        source=integers(0, 127),
        pack=lambda v: b"%c" % v)
negative_fixnum = _scalar_st(
        source=integers(-32, -1),
        pack=lambda v: b"%c" % (v + 256 | 0xe0))


def _num_st(firstbyte, dtype):
    from hypothesis.extra.numpy import arrays
    return _scalar_st(
            source=arrays(dtype, ()),
            pack=lambda v: b"%c%s" % (firstbyte, v.tobytes()))

uint8, uint16, uint32, uint64 = (
        _num_st(0xcc + i, ">u%s" % 2**i) for i in range(4))
int8, int16, int32, int64 = (
        _num_st(0xd0 + i, ">i%s" % 2**i) for i in range(4))
float32, float64 = (
        _num_st(0xca + i, ">f%s" % 2**i) for i in range(2, 4))


def _int_tobytes(v, nbytes):
    import numpy
    return numpy.array(v, ">u%d" % nbytes).tobytes()

def _pack_bin(firstbyte, nbytes_len):
    def pack(v):
        _len = _int_tobytes(len(v), nbytes_len)
        return b"%c%s%s" % (firstbyte, _len, v)
    return pack

bin8, bin16, bin32 = (_scalar_st(
        source=binary(max_size=2**(8 * 2**i) - 1),
        pack=_pack_bin(0xc4 + i, 2**i))
    for i in range(3))


def _text(max_size):
    """makes unicode string with an utf-8 length of at most max_size.

    `hypothesis.strategies.text`'s `max_size` counts unicode *characters*, but we
    can't go over a certain number of *bytes*, so this wrapper makes up for it.
    `text` is used internally as an approximate.
    """
    return text(max_size=max_size).filter(
            lambda v: len(v.encode()) <= max_size)

fixstr = _scalar_st(
        source=_text(max_size=31),
        pack=lambda v: (b"%c%s" % (0xa0 | len(v.encode()), v.encode())))

def _pack_str(i):
    pack2 = _pack_bin(0xd9 + i, 2**i)
    def pack(v):
        return pack2(v.encode())
    return pack

str8, str16, str32 = (_scalar_st(
        source=_text(max_size=2**(8 * 2**i) - 1),
        pack=_pack_str(i))
    for i in range(3))


all_scalar = one_of(*_all_scalar)
"""aggregate of all non-recursive strategies"""

del _all_scalar  # just to be safe


def _concat_elements(l):
    if len(l) == 0:
        return b"", []
    else:
        packed_vs, vs = zip(*l)
        return b"".join(packed_vs), list(vs)

def _lists(_max_size):
    """`hypothesis.strategies.lists`-like strategy factory.

    `_max_size` is the physical maximum element count for the MessagePack
    container and will be used to cap the passed `max_size`.
    """
    def st(elements=all_scalar, *, max_size=_max_size, **kwargs):
        max_size = min(max_size, _max_size)
        source = lists(elements, max_size=max_size, **kwargs)
        return source.map(_concat_elements)
    return st

def fixarray(*args, **kwargs):
    source = _lists(_max_size=15)(*args, **kwargs)
    return source.map(lambda packed_vs, vs:
            b"%c%s" % (0x90 | len(vs), packed_vs))

def _array_st(i):
    def st(*args, **kwargs):
        source = _lists(_max_size=2**(8 * 2**i) - 1)(*args, **kwargs)
        return source.map(lambda packed_vs, vs:
                b"%c%s%s" % (0xdc + i, _int_tobytes(len(vs), 2**i), packed_vs))
    return st

array16, array32 = (_array_st(i) for i in (1, 2))

def all_arrays(*args, **kwargs):
    return one_of(a(*args, **kwargs) for a in (fixarray, array16, array32))


def _concat_items(d):
    packed_items, items = _concat_elements(
            [(packed_key + packed_val, (key, val))
                for (packed_key, key), (packed_val, val) in v.items()])
    return packed_items, dict(items)

def _dictionaries(_max_size):
    """`_lists`, mutatis mutandis"""
    def st(keys=all_scalar, values=all_scalar, *, max_size=_max_size, **kwargs):
        max_size = min(max_size, _max_size)
        source = dictionaries(keys, values, max_size=max_size, **kwargs)
        return source.map(_concat_elements)
    return st

def fixmap(*args, **kwargs):
    source = _dictionaries(_max_size=15)(*args, **kwargs)
    return source.map(lambda packed_d, d:
            b"%c%s" % (0x80 | len(d), packed_d))

def _map_st(i):
    def st(*args, **kwargs):
        source = _dictionaries(_max_size=2**(8 * 2**i) - 1)(*args, **kwargs)
        return source.map(lambda packed_d, d:
                b"%c%s%s" % (0xde + i, _int_tobytes(len(d), 2**i), packed_d))
    return st

map16, map32 = (_map_st(i) for i in (1, 2))

def all_maps(*args, **kwargs):
    return one_of(m(*args, **kwargs) for m in (fixmap, map16, map32))


all_types = recursive(all_scalar, lambda s:
        fixarray(s) | array16(s) | array32(s) |
        fixmap(all_scalar, s) | map16(all_scalar, s) | map32(all_scalar, s))
