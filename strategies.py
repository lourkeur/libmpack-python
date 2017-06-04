"""hypothesis strategies for testing libmpack.

Each strategy's goal is to generate a value suitable for a particular format,
pack it using a slow but easily verified implementation of MessagePack, and
then return the packed representation tupled with the original value.
"""

from hypothesis.strategies import assume, composite, just, none, booleans, integers, lists, tuples, dictionaries, recursive, sampled_from, text, binary
from hypothesis.extra.numpy import arrays

import collections
import numpy


@composite
def nil(draw):
    return b"\xc0", None

@composite
def boolean(draw, values=booleans()):
    v = draw(values)
    return b"%c" % (0xc2 + v), bool(v)

@composite
def positive_fixnum(draw, values=integers(0, 127).map(numpy.int8), prepack=lambda v: v):
    v = draw(values)
    return b"%c" % prepack(v), v

@composite
def negative_fixnum(draw, values=integers(-32, -1).map(numpy.int8)):
    v = draw(values)
    return b"%c" % (256 + v | 0xe0), numpy.int8(v)

def _num_tobytes(dtype, v):
    return numpy.array(v, dtype).tobytes()

def _num_max(dtype):
    return numpy.iinfo(dtype).max

def _do_num(draw, dtype, firstbyte, postpack=lambda v: v):
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
def all_uint(draw):
    return draw(uint8() | uint16() | uint32() | uint64())

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
@composite
def all_int(draw):
    return draw(int8() | int16() | int32() | int64())


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
        return 'nan'

nan = _Nan()
def _float_postpack(v):
    return nan if numpy.isnan(v) else v

@composite
def float32(draw):
    return _do_num(draw, ">f4", 0xca, _float_postpack)
@composite
def float64(draw):
    return _do_num(draw, ">f8", 0xcb, _float_postpack)
@composite
def all_float(draw):
    return draw(float32() | float64())


# general tweak to avoid Hypothesis buffer overruns.
AVERAGE_BIN_SIZE = 20

def _do_bin(dtype, firstbyte, binary=binary, prepack=lambda v: v):
    @composite
    def _(draw, contents=binary(average_size=AVERAGE_BIN_SIZE, max_size=_num_max(dtype))):
        data = v = draw(contents)
        assume(len(data) <= _num_max(dtype))
        return b"%c%s%s" % (firstbyte, _num_tobytes(dtype, len(data)), data), v
    return _

bin8 = _do_bin(">u1", 0xc4)
bin16 = _do_bin(">u2", 0xc5)
bin32 = _do_bin(">u4", 0xc6)

@composite
def all_bin(draw, **kwargs):
    return draw(bin8(**kwargs) | bin16(**kwargs) | bin32(**kwargs))


@composite
def fixstr(draw, contents):
    v = draw(contents)
    data = v.encode("utf-8")
    assume(len(data) <= 31)
    return b"%c%s" % (0xa0 | len(data), data), v

def _do_str(draw, dtype, firstbyte, contents):
    return _do_bin(draw, dtype, firstbyte, binary=text, prepack=lambda v: v.encode("utf-8"))

str8 = _do_str(">u1", 0xd9)
str16 = _do_str(">u2", 0xda)
str32 = _do_str(">u4", 0xdb)

@composite
def all_str(draw, **kwargs):
    return draw(fixstr(**kwargs) | str8(**kwargs) | str16(**kwargs) | str32(**kwargs))


class _ExtBase(object):
    # code = ...
    def __init__(self, data):
        assert hasattr(self, 'code')
        self.data = data

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.data == other.data

    def __ne__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.data != other.data

    def __hash__(self):
        return hash((self.code, self.data))

ext_classes = [type('_Ext%d' % i, (_ExtBase,), {'code': i}) for i in range(128)]
def ext_pack(ext):
    return ext.code, ext.data
def ext_unpack(code, data):
    return ext_classes[code](data)
_default_extcodes = integers(0, 127)

def _do_fixext(size, firstbyte):
    @composite
    def _(draw, extcodes=_default_extcodes, payloads=binary(min_size=size, max_size=size)):
        code = draw(extcodes)
        data = draw(payloads)
        assume(len(data) == size)
        return b"%c%c%s" % (firstbyte, code, data), ext_unpack(code, data)
    return _

fixext1 = _do_fixext(1, 0xd4)
fixext2 = _do_fixext(2, 0xd5)
fixext4 = _do_fixext(4, 0xd6)
fixext8 = _do_fixext(8, 0xd7)
fixext16 = _do_fixext(16, 0xd8)

def _do_ext(dtype, firstbyte, kwargs):
    @composite
    def _(draw, extcodes=_default_extcodes, payloads=binary(average_size=AVERAGE_BIN_SIZE, max_size=_num_max(dtype))):
        code = draw(extcodes)
        data = draw(payloads)
        assume(len(data) <= _num_max(dtype))
        return b"%c%s%c%s" % (firstbyte, _num_tobytes(dtype, len(data)), code, data), ext_unpack(code, data)
    return _
ext8 = _do_ext('>u1', 0xc7, kwargs)
ext16 = _do_ext('>u2', 0xc8, kwargs)
ext32 = _do_ext('>u4', 0xc9, kwargs)

@composite
def ext8(draw, **kwargs):
    return _do_ext(draw, '>u1', 0xc7, kwargs)
@composite
def ext16(draw, **kwargs):
    return _do_ext(draw, '>u2', 0xc8, kwargs)
@composite
def ext32(draw, **kwargs):
    return _do_ext(draw, '>u4', 0xc9, kwargs)

@composite
def all_ext(draw, **kwargs):
    return draw(fixext1(**kwargs) | fixext2(**kwargs) | fixext4(**kwargs) | fixext8(**kwargs) | fixext16(**kwargs) | ext8(**kwargs) | ext16(**kwargs) | ext32(**kwargs))


@composite
def all_scalar(draw, boolean=boolean(), positive_fixnum=positive_fixnum(), negative_fixnum=negative_fixnum(), all_uint=all_uint(), all_int=all_int(), all_float=all_float(), all_bin=all_bin(), all_str=all_str(), all_ext=all_ext()):
    return draw(nil() | boolean | positive_fixnum | negative_fixnum | all_uint | all_int | all_float | all_bin | all_str)


def _concat_elements(l):
    packed_vs, vs = bytearray(), []
    for packed_v, v in l:
        packed_vs += packed_v
        vs.append(v)
    return bytes(packed_vs), vs

_AVERAGE_ARRAY_SIZE = 6

@composite
def fixarray(draw, lists=lists, **kwargs):
    _limit_size(15, _AVERAGE_ARRAY_SIZE, kwargs)
    kwargs.setdefault("elements", all_scalar())
    l = draw(lists(**kwargs))
    data, v = _concat_elements(l)
    return b"%c%s" % (0x90 | len(v), data), v

def _do_array(draw, dtype, firstbyte, kwargs):
    _limit_size(_num_max(dtype), _AVERAGE_ARRAY_SIZE, kwargs)
    kwargs.setdefault("elements", all_scalar())
    l = draw(kwargs.pop("lists", lists)(**kwargs))
    data, v = _concat_elements(l)
    return b"%c%s%s" % (firstbyte, _num_tobytes(dtype, len(v)), data), v

@composite
def array16(draw, **kwargs):
    return _do_array(draw, ">u2", 0xdc, kwargs)

@composite
def array32(draw, **kwargs):
    return _do_array(draw, ">u4", 0xdd, kwargs)

@composite
def all_array(draw, **kwargs):
    return draw(fixarray(**kwargs) | array16(**kwargs) | array32(**kwargs))


class KeyWrapper(collections.namedtuple('_KeyWrapper', 'packed v')):
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

def _hashable(k):
    try: hash(k)
    except TypeError:
        return False
    else:
        return True

def _concat_items(d):
    packed_items, items = _concat_elements(
            (packed_key + packed_val, (key, val))
                for (packed_key, key), (packed_val, val) in d.items())
    return packed_items, dict(items)

def _prepare_keys(kwargs):
    keys = kwargs.setdefault("keys", all_scalar())
    kwargs["keys"] = keys.filter(_hashable).map(_wrap_key)

_AVERAGE_MAP_SIZE = 3

@composite
def fixmap(draw, dictionaries=dictionaries, **kwargs):
    _limit_size(15, _AVERAGE_MAP_SIZE, kwargs)
    _prepare_keys(kwargs)
    kwargs.setdefault("values", all_scalar)
    d = draw(dictionaries(**kwargs))
    data, v = _concat_items(d)
    return b"%c%s" % (0x80 | len(v), data), v

def _do_map(draw, dtype, firstbyte, kwargs):
    _limit_size(_num_max(dtype), _AVERAGE_MAP_SIZE, kwargs)
    _prepare_keys(kwargs)
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

@composite
def all_map(draw, **kwargs):
    return draw(fixmap(**kwargs) | map16(**kwargs) | map32(**kwargs))

@composite
def everything(draw):
    return draw(recursive(all_scalar(), lambda S: all_array(elements=S) | all_map(values=S)))


_msg_types = 'request', 'response', 'notification'

@composite
def msg(draw, types=_msg_types, msg_id=uint32(), method=all_str(), params=all_array(), has_error=booleans(), errors=everything(), results=everything()):
    packed_t, t = draw(positive_fixnum(values=sampled_from(types), prepack=lambda t: _msg_types.index(t)))
    if t == 'request':
        payload_tail = msg_id, method, params
    elif t == 'response':
        if draw(has_error):
            payload_tail = msg_id, errors, nil()
        else:
            payload_tail = msg_id, nil(), results
    elif t == 'notification':
        payload_tail = method, params
    else:
        raise ValueError("Invalid message type", t)
    return draw(fixarray(lists=lambda **kwargs: tuples(just((packed_t, t)), *payload_tail)))
