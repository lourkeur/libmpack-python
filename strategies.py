"""hypothesis strategies for testing libmpack.

Each strategy's goal is to generate a value suitable for a particular format,
pack it using a slow but easily verified implementation of MessagePack, and
then return the packed representation tupled with the original value.
"""

from hypothesis.strategies import assume, composite, just, none, booleans, integers, lists, tuples, dictionaries, recursive, sampled_from, text, binary
from hypothesis.extra.numpy import arrays

import collections
import functools
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


def _limit_size(max_size, average_size, kwargs):
    if 'max_size' in kwargs:
        max_size = min(kwargs['max_size'], max_size)
    kwargs['max_size'] = max_size
    kwargs['average_size'] = min(average_size, max_size)

# general tweak to avoid Hypothesis buffer overruns.
AVERAGE_BIN_SIZE = 20

def _do_bin(draw, dtype, firstbyte, kwargs, prepack=lambda v: v):
    _limit_size(_num_max(dtype), _AVERAGE_STR_SIZE, kwargs)
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
def all_bin(draw, *args, **kwargs):
    return draw(one_of(b(*args, **kwargs) for b in (bin8, bin16, bin32)))


@composite
def fixstr(draw, **kwargs):
    _limit_size(31, _AVERAGE_STR_SIZE, kwargs)
    v = draw(kwargs.pop('text', text)(**kwargs))
    data = v.encode("utf-8")
    assume(len(data) <= 31)
    return b"%c%s" % (0xa0 | len(data), data), v

def _str_prepack(dtype):
    def f(v):
        data = v.encode("utf-8")
        assume(len(data) <= _num_max(dtype))
        return data
    return f

def _do_str(draw, dtype, firstbyte, kwargs):
    kwargs["binary"] = kwargs.pop("text", text)
    return _do_bin(draw, dtype, firstbyte, kwargs, _str_prepack(dtype))

@composite
def str8(draw, **kwargs):
    return _do_str(draw, ">u1", 0xd9, kwargs)
@composite
def str16(draw, **kwargs):
    return _do_str(draw, ">u2", 0xda, kwargs)
@composite
def str32(draw, **kwargs):
    return _do_str(draw, ">u4", 0xdb, kwargs)

@composite
def all_str(draw, *args, **kwargs):
    return draw(one_of(s(*args, **kwargs) for s in (fixstr, str8, str16, str32)))


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

def _do_fixext(draw, size, firstbyte, kwargs):
    code = draw(kwargs.pop('extcodes', integers(0, 127)))
    data = draw(kwargs.pop("binary", binary)(min_size=size, max_size=size))
    return b"%c%c%s" % (firstbyte, code, data), ext_unpack(code, data)

@composite
def fixext1(draw, **kwargs):
    return _do_fixext(draw, 1, 0xd4, kwargs)
@composite
def fixext2(draw, **kwargs):
    return _do_fixext(draw, 2, 0xd5, kwargs)
@composite
def fixext4(draw, **kwargs):
    return _do_fixext(draw, 4, 0xd6, kwargs)
@composite
def fixext8(draw, **kwargs):
    return _do_fixext(draw, 8, 0xd7, kwargs)
@composite
def fixext16(draw, **kwargs):
    return _do_fixext(draw, 16, 0xd8, kwargs)

def _do_ext(draw, dtype, firstbyte, kwargs):
    code = draw(kwargs.pop('extcodes', integers(0, 127)))
    _limit_size(_num_max(dtype), _AVERAGE_STR_SIZE, kwargs)
    data = draw(kwargs.pop("binary", binary)(**kwargs))
    return b'%c%s%c%s' % (firstbyte, _num_tobytes(dtype, len(data)), code, data), ext_unpack(code, data)

@composite
def all_ext(draw, *args, **kwargs):
    return draw(one_of(e(*args, **kwargs) for e in (ext8, ext16, ext32)))


@composite
def all_scalar(draw, boolean=boolean(), positive_fixnum=positive_fixnum(), negative_fixnum=negative_fixnum(), all_uint=all_uint(), all_int=all_int(), all_float=all_float(), all_bin=all_bin(), all_str=all_str(), fixext1=fixext1(), fixext2=fixext2(), fixext4=fixext4(), fixext8=fixext8(), fixext16=fixext16(), all_ext=all_ext()):
    return draw(one_of(nil(), boolean, positive_fixnum, negative_fixnum, all_uint, all_int, all_float, all_bin, all_str, fixext1, fixext2, fixext4, fixext8, fixext16, all_ext))


class PartialStrategy:
    def __init__(self, *args, **kwargs):
        self.wrapped = functools.partial(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.wrapped(*args, **kwargs)

    def __str__(self):
        return self.wrapped().__str__()


def _concat_elements(l):
    packed_vs, vs = bytearray(), []
    for packed_v, v in l:
        packed_vs += packed_v
        vs.append(v)
    return bytes(packed_vs), vs

AVERAGE_ARRAY_SIZE = 6

@PartialStrategy
@composite
def array_contents(  # arglist kept in sync with on hypothesis.strategies.list
        draw,
        elements=all_scalar(),
        min_size=None,
        average_size=None,
        max_size=None,
        unique_by=None,
        unique=False,
        *,
        hard_max_size=None,
        ):
    assert hard_max_size is not None
    if max_size is None or max_size > hard_max_size:
        max_size = hard_max_size
    if average_size is None:
        average_size = AVERAGE_ARRAY_SIZE
    return draw(lists(elements, min_size, average_size, max_size, unique_by, unique))


@composite
def fixarray(draw, array_contents=array_contents()):
    l = draw(array_contents(hard_max_size=15))
    data, v = _concat_elements(l)
    return b"%c%s" % (0x90 | len(v), data), v

def _do_array(draw, dtype, firstbyte, array_contents=array_contents):
    hard_max_size = _num_max(dtype)
    l = draw(array_contents(hard_max_size=hard_max_size))
    data, v = _concat_elements()
    return b"%c%s%s" % (firstbyte, _num_tobytes(dtype, len(v)), data), v

@composite
def array16(draw, **kwargs):
    return _do_array(draw, ">u2", 0xdc, **kwargs)

@composite
def array32(draw, **kwargs):
    return _do_array(draw, ">u4", 0xdd, **kwargs)

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

def _concat_items(d):
    packed_items, items = _concat_elements(
            (packed_key + packed_val, (key, val))
                for (packed_key, key), (packed_val, val) in d.items())
    return packed_items, dict(items)

AVERAGE_MAP_SIZE = 3

@PartialStrategy
@composite
def map_contents(  # arglist kept in sync with on hypothesis.strategies.dictionaries
        draw,
        keys=all_scalar(),
        values=all_scalar(),
        dict_class=collections.OrderedDict,
        min_size=None,
        average_size=None,
        max_size=None,
        *,
        hard_max_size=None,
        ):
    assert hard_max_size is not None
    if max_size is None or max_size > hard_max_size:
        max_size = hard_max_size
    if average_size is None:
        average_size = AVERAGE_MAP_SIZE
    return draw(dictionaries(keys.map(_KeyWrapper), values, dict_class, min_size, average_size, max_size))

@composite
def fixmap(draw, map_contents=map_contents()):
    d = draw(map_contents(hard_max_size=15))
    data, v = _concat_items(d)
    return b"%c%s" % (0x80 | len(v), data), v

def _do_map(draw, dtype, firstbyte, kwargs):
    hard_max_size = _num_max(dtype)
    d = draw(map_contents(hard_max_size=hard_max_size))
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
    return draw(recursive(all_scalar(), lambda S: all_array(array_contents(elements=S)) | all_map(map_contents(values=S))))


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
    return draw(fixarray(tuples(just((packed_t, t)), *payload_tail)))
