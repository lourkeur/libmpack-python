"""hypothesis strategies for testing libmpack.

Each strategy's goal is to generate a value suitable for a particular format,
pack it using a slow but easily verified implementation of MessagePack, and
then return the packed representation tupled with the original value.
"""

from hypothesis.strategies import (assume, composite, none, booleans, integers, lists,
        dictionaries, recursive, one_of, text, binary)
from hypothesis.extra.numpy import arrays

import collections
import numpy


@composite
def nil(draw, none=none):
    draw(none())
    return b"\xc0", None

@composite
def boolean(draw, booleans=booleans):
    v = draw(booleans())
    return b"%c" % (0xc2 + v), bool(v)

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
        if numpy.dtype(type(other)).kind != 'f':
            return NotImplemented
        return numpy.isnan(other)

    def __ne__(self, other):
        if numpy.dtype(type(other)).kind != 'f':
            return NotImplemented
        return not numpy.isnan(other)

    def __hash__(self):
        return hash(numpy.nan)

    def __repr__(self):
        return "_Nan()"

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


all_scalar = one_of(
        nil(),
        boolean(),
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


class _Sequence(tuple):
    """mpack yields lists, but we don't wan't to yield those because they are not hashable."""
    def __eq__(self, other):
        try:
            other = tuple(other)
        except TypeError:
            return NotImplemented
        return super(_Sequence, self).__eq__(other)

    def __ne__(self, other):
        try:
            other = tuple(other)
        except TypeError:
            return NotImplemented
        return super(_Sequence, self).__ne__(other)

    def __hash__(self):
        return super(_Sequence, self).__hash__()

def _concat_elements(l):
    packed_vs, vs = bytearray(), []
    for packed_v, v in l:
        packed_vs += packed_v
        vs.append(v)
    return bytes(packed_vs), _Sequence(vs)


@composite
def fixarray(draw, lists=lists, **kwargs):
    _limit_size(15, kwargs)
    kwargs.setdefault("elements", all_scalar)
    l = draw(lists(**kwargs))
    data, v = _concat_elements(l)
    return b"%c%s" % (0x90 | len(v), data), v

def _do_array(draw, dtype, firstbyte, kwargs):
    _limit_size(_num_max(dtype), kwargs)
    kwargs.setdefault("elements", all_scalar)
    l = draw(kwargs.pop("lists", lists)(**kwargs))
    data, v = _concat_elements(l)
    return b"%c%s%s" % (firstbyte, _num_tobytes(dtype, len(v)), data), v

@composite
def array16(draw, **kwargs):
    return _do_array(draw, ">u2", 0xdc, kwargs)
@composite
def array32(draw, **kwargs):
    return _do_array(draw, ">u4", 0xdd, kwargs)


class _KeyWrapper(collections.namedtuple('_key_wrapper', 'packed v')):
    def __eq__(self, other):
        if not isinstance(other, _KeyWrapper):
            return NotImplemented
        return self.v == other.v

    def __ne__(self, other):
        if not isinstance(other, _KeyWrapper):
            return NotImplemented
        return self.v != other.v

    def __hash__(self):
        return hash(self.v)

def _wrap_key(k):
    v, packed = k
    return _KeyWrapper(v, packed)

def _concat_items(d):
    packed_items, items = _concat_elements(
            (packed_key + packed_val, (key, val))
                for (packed_key, key), (packed_val, val) in d.items())
    return packed_items, dict(items)

@composite
def fixmap(draw, dictionaries=dictionaries, **kwargs):
    _limit_size(15, kwargs)
    kwargs.setdefault("keys", all_scalar)
    kwargs["keys"] = kwargs["keys"].map(_wrap_key)
    kwargs.setdefault("values", all_scalar)
    d = draw(dictionaries(**kwargs))
    data, v = _concat_items(d)
    return b"%c%s" % (0x80 | len(v), data), v

def _do_map(draw, dtype, firstbyte, kwargs):
    _limit_size(_num_max(dtype), kwargs)
    kwargs.setdefault("keys", all_scalar)
    kwargs["keys"] = kwargs["keys"].map(_wrap_key)
    kwargs.setdefault("values", all_scalar)
    d = draw(kwargs.pop("dictionaries", dictionaries)(**kwargs))
    data, v = _concat_items(d)
    return b"%c%s%s" % (firstbyte, _num_tobytes(dtype, len(v)), data), v

@composite
def map16(draw, **kwargs):
    return _do_map(draw, ">u2", 0xde, kwargs)
@composite
def map32(draw, **kwargs):
    return _do_map(draw, ">u4", 0xdf, kwargs)
