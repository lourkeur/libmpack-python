"""hypothesis strategies for testing libmpack.

Each strategy's goal is to generate a value suitable for a particular format,
pack it using a slow but easily verified implementation of MessagePack, and
then return the packed representation tupled with the original value.
"""

from hypothesis.strategies import (assume, composite, none, booleans, integers, lists,
        dictionaries, recursive, one_of, text, binary)
import numpy


@composite
def nil(draw, none=none):
    draw(none())
    return b"\xc0", None

@composite
def bool(draw, booleans=booleans):
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
def positive_fixnum(draw, integers=integers, **kwargs):
    _limit_values(0, 127, kwargs)
    v = draw(integers(**kwargs))
    return b"%c" % v, v

@composite
def negative_fixnum(draw, integers=integers, **kwargs):
    _limit_values(-32, -1, kwargs)
    v = draw(integers(**kwargs))
    return b"%c" % (v + 256 | 0xe0), v

def _num_tobytes(dtype, v):
    return numpy.array(v, dtype).tobytes()

def _num_max(dtype):
    return numpy.iinfo(dtype).max

def _do_num(draw, dtype, firstbyte, kwargs, postpack=lambda v: v):
    from hypothesis.extra.numpy import arrays
    if "min_value" in kwargs or "max_value" in kwargs:
        raise NotImplementedError("custom limits")
    v = draw(arrays(dtype, ()))
    return b"%c%s" % (firstbyte, _num_tobytes(dtype, v)), postpack(v)

@composite
def uint8(draw, **kwargs):
    return _do_num(draw, ">u1", 0xcc, kwargs)
@composite
def uint16(draw, **kwargs):
    return _do_num(draw, ">u2", 0xcd, kwargs)
@composite
def uint32(draw, **kwargs):
    return _do_num(draw, ">u4", 0xce, kwargs)
@composite
def uint64(draw, **kwargs):
    return _do_num(draw, ">u8", 0xcf, kwargs)

@composite
def int8(draw, **kwargs):
    return _do_num(draw, ">i1", 0xd0, kwargs)
@composite
def int16(draw, **kwargs):
    return _do_num(draw, ">i2", 0xd1, kwargs)
@composite
def int32(draw, **kwargs):
    return _do_num(draw, ">i4", 0xd2, kwargs)
@composite
def int64(draw, **kwargs):
    return _do_num(draw, ">i8", 0xd3, kwargs)


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
def float32(draw, **kwargs):
    return _do_num(draw, ">f4", 0xca, kwargs, _float_postpack)
@composite
def float64(draw, **kwargs):
    return _do_num(draw, ">f8", 0xcb, kwargs, _float_postpack)


def _limit_size(max_size, kwargs):
    try:
        kwargs["max_size"] = min(kwargs["max_size"], max_size)
    except KeyError:
        kwargs["max_size"] = max_size
    kwargs.setdefault("average_size", min(20, max_size))  # general tweak to avoid Hypothesis buffer overruns.


def _do_bin(draw, dtype, firstbyte, kwargs, prepack=lambda v: v):
    _limit_size(_num_max(dtype), kwargs)
    v = draw(kwargs.pop("binary", binary)(**kwargs))
    data = prepack(v)
    return b"%c%s%s" % (firstbyte, _num_tobytes(dtype, len(data)), data), v

@composite
def bin8(draw, **kwargs):
    return _do_bin(draw, ">u1", 0xc4, kwargs)
@composite
def bin16(draw, **kwargs):
    return _do_bin(draw, ">u2", 0xc5, kwargs)
@composite
def bin32(draw, **kwargs):
    return _do_bin(draw, ">u4", 0xc6, kwargs)


@composite
def fixstr(draw, **kwargs):
    _limit_size(31, kwargs)
    v = draw(text(**kwargs))
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
def str8(draw, **kwargs):
    kwargs["binary"] = kwargs.pop("text", text)
    return _do_bin(draw, ">u1", 0xd9, kwargs, _str_prepack(">u1"))
@composite
def str16(draw, **kwargs):
    kwargs["binary"] = kwargs.pop("text", text)
    return _do_bin(draw, ">u2", 0xda, kwargs, _str_prepack(">u2"))
@composite
def str32(draw, **kwargs):
    kwargs["binary"] = kwargs.pop("text", text)
    return _do_bin(draw, ">u4", 0xdb, kwargs, _str_prepack(">u4"))


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
