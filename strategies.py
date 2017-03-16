"""hypothesis strategies for testing libmpack.

Each strategy's goal is to generate a value suitable for a particular format,
pack it using a slow but easily verified implementation of MessagePack, and
then return the packed representation tupled with the original value.
"""

from hypothesis.strategies import (assume, composite, none, booleans, integers, lists,
        dictionaries, recursive, one_of, text, binary)
from hypothesis.searchstrategy import SearchStrategy
import numpy


@composite
def nil(draw):
    draw(none())
    return b"\xc0", None

@composite
def bool(draw):
    v = draw(booleans())
    return b"%c" % (0xc2 + v), v


def _limit_values(min_value, max_value, kwargs):
    try:
        kwargs["min_value"] = max(kwargs["min_value"], min_value)
    except KeyError:
        kwargs["min_value"] = min_value
    try:
        kwargs["max_value"] = min(kwargs["max_value"], max_value)
    except KeyError:
        kwargs["max_value"] = max_value

@composite
def positive_fixnum(draw, **kwargs):
    _limit_values(0, 127, kwargs)
    v = draw(integers(**kwargs))
    return b"%c" % v, v

@composite
def negative_fixnum(draw, **kwargs):
    _limit_values(-32, -1, kwargs)
    v = draw(integers(**kwargs))
    return b"%c" % (v + 256 | 0xe0), v

def _num_tobytes(dtype, v):
    return numpy.array(v, dtype).tobytes()

def _do_num(draw, dtype, firstbyte, postpack=lambda v: v):
    from hypothesis.extra.numpy import arrays
    v = draw(arrays(dtype, ()))
    return b"%c%s" % (firstbyte, _num_tobytes(dtype, v)), postpack(v)

@composite
def uint8(draw):
    return _do_num(draw, ">u1", 0xcc)
@composite
def uint16(draw):
    return _do_num(draw, ">u2", 0xcd)
@composite
def uint32(draw):
    return _do_num(draw, ">u4", 0xce)
@composite
def uint64(draw):
    return _do_num(draw, ">u8", 0xcf)

@composite
def int8(draw):
    return _do_num(draw, ">i1", 0xd0)
@composite
def int16(draw):
    return _do_num(draw, ">i2", 0xd1)
@composite
def int32(draw):
    return _do_num(draw, ">i4", 0xd2)
@composite
def int64(draw):
    return _do_num(draw, ">i8", 0xd3)


class _Nan(object):
    """It's equal to this number that's not even equal to itself.

    This greatly simplifies things downstream.
    """
    def __eq__(self, other):
        if not numpy.isreal(other):
            return NotImplemented
        return numpy.isnan(other)

    def __ne__(self, other):
        if not numpy.isreal(other):
            return NotImplemented
        return not numpy.isnan(other)

    def __hash__(self):
        return hash(numpy.nan)

_nan = _Nan()
def _float_postpack(v):
    return _nan if numpy.isnan(v) else v

@composite
def float32(draw):
    return _do_num(draw, ">f4", 0xca, _float_postpack)
@composite
def float64(draw):
    return _do_num(draw, ">f8", 0xcb, _float_postpack)


def _do_bin(draw, dtype, firstbyte, binary=binary, prepack=lambda v: v):
    v = draw(binary(max_size=numpy.iinfo(dtype).max))
    data = prepack(v)
    return b"%c%s%s" % (firstbyte, _num_tobytes(dtype, len(data)), data), v

@composite
def bin8(draw):
    return _do_bin(draw, ">u1", 0xc4)
@composite
def bin16(draw):
    return _do_bin(draw, ">u2", 0xc5)
@composite
def bin32(draw):
    return _do_bin(draw, ">u4", 0xc6)


@composite
def fixstr(draw):
    v = draw(text(max_size=31))
    data = v.encode("utf-8")
    assume(len(data) < 31)  # a unicode character can be many bytes, so there's a small chance that we might overrun.
    return b"%c%s" % (0xa0 | len(data), data), v

def _str_prepack(dtype):
    def f(v):
        data = v.encode("utf-8")
        assume(len(data) < numpy.iinfo(dtype).max)
        return data
    return f

@composite
def str8(draw):
    return _do_bin(draw, ">u1", 0xd9, binary=text, prepack=_str_prepack(">u1"))
@composite
def str16(draw):
    return _do_bin(draw, ">u2", 0xda, binary=text, prepack=_str_prepack(">u2"))
@composite
def str32(draw):
    return _do_bin(draw, ">u4", 0xdb, binary=text, prepack=_str_prepack(">u4"))


scalar_formats = one_of(
        nil(),
        bool(),
        positive_fixnum(),
        negative_fixnum(),
        uint8(),
        uint16(),
        uint32(),
        uint64(),
        int8(),
        int16(),
        int32(),
        int64(),
        float32(),
        float64(),
        bin8(),
        bin16(),
        bin32(),
        fixstr(),
        str8(),
        str16(),
        str32(),
        )


def _concat_elements(l):
    packed_vs, vs = bytearray(), []
    for pv, v in l:
        packed_vs += pv
        vs.append(v)
    return bytes(packed_vs), vs

def _concat_items(d):
    packed_items, items = _concat_elements(
            (packed_key + packed_val, (key, val))
                for (packed_key, key), (packed_val, val) in d.items())
    return packed_items, items
