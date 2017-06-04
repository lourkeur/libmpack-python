"""hypothesis strategies for testing libmpack.

Each strategy's goal is to generate a value suitable for a particular format,
pack it using a slow but easily verified implementation of MessagePack, and
then return the packed representation tupled with the original value.
"""

from hypothesis.strategies import assume, composite, just, none, booleans, integers, lists, tuples, dictionaries, recursive, one_of, sampled_from, text, binary
from hypothesis.extra.numpy import arrays

import collections
import functools
import numpy
import re

from compat import bfmt, callable


@composite
def nil(draw):
    return b"\xc0", None

@composite
def boolean(draw, values=booleans()):
    v = draw(values)
    return bfmt(b"%c", 0xc2 + v), bool(v)

@composite
def positive_fixnum(draw, values=integers(0, 127).map(numpy.int8), prepack=lambda v: v):
    v = draw(values)
    return bfmt(b"%c", prepack(v)), v

@composite
def negative_fixnum(draw, values=integers(-32, -1).map(numpy.int8)):
    v = draw(values)
    return bfmt(b"%c", 0xe0 | 256 + v), numpy.int8(v)

def _num_tobytes(dtype, v):
    return numpy.array(v, dtype).tobytes()

def _num_max(dtype):
    return numpy.iinfo(dtype).max

def _do_num(draw, dtype, firstbyte, postpack=lambda v: v):
    v = draw(arrays(dtype, ()))
    return bfmt(b"%c%s", firstbyte, _num_tobytes(dtype, v)), postpack(v)

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


def _limit_size(
        min_size, average_size, max_size, *,  # user controlled
        hard_max_size=None, hard_size=None, default_average_size=None,  # strategy controlled
        ):
    if min_size is not None and max_size is not None and min_size > max_size:
        raise ValueError("min_size > max_size")
    if average_size is not None:
        if max_size is not None and average_size > max_size:
            raise ValueError("average_size > max_size")
        if min_size is not None and average_size < min_size:
            raise ValueError("average_size < min_size")
    else:
        average_size = default_average_size
    if hard_size is not None:
        assert hard_max_size is None
        min_size, max_size = hard_size, hard_size
    else:
        assert hard_max_size is not None
        if max_size is None or max_size > hard_max_size:
            max_size = hard_max_size
        if min_size is not None and min_size > max_size:
            min_size = max_size
    if (min_size is not None and average_size is not None
        and not min_size <= average_size <= max_size):
        # Our logic broke the average_size invariant.
        # Fix silently so we're more robust.
        average_size = None
    return min_size, average_size, max_size


class _CurriedStrategy:
    def __init__(self, *args, **kwargs):
        self.wrapped = functools.partial(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.wrapped(*args, **kwargs)

    def __str__(self):
        return re.sub(r"(?<=\()[^(]*(?=\)$)", lambda m: m[0] + ', ...' if m[0] else '...', self.wrapped().__str__())

def _curried_strategy(f):
    return functools.partial(_CurriedStrategy, f)


# general tweak to avoid Hypothesis buffer overruns.
AVERAGE_BIN_SIZE = 20

@_curried_strategy
@composite
def payloads(  # arglist kept in sync with hypothesis.strategies.binary
        draw,
        min_size=None,
        average_size=None,
        max_size=None,
        hard_max_size=None,
        hard_size=None,
        ):
    sizes = _limit_size(
            min_size, average_size, max_size,
            hard_max_size=hard_max_size,
            hard_size=hard_size,
            default_average_size=AVERAGE_BIN_SIZE,
            )
    return draw(binary(*sizes))

def _do_bin(draw, dtype, firstbyte, payloads, prepack=lambda v: v):
    hard_max_size = _num_max(dtype)
    if callable(payloads):
        payloads = payloads(hard_max_size=hard_max_size)
    v = draw(payloads)
    data = prepack(v)
    return bfmt(b"%c%s%s", firstbyte, _num_tobytes(dtype, len(data)), data), v

@composite
def bin8(draw, payloads=payloads()):
    return _do_bin(draw, ">u1", 0xc4, payloads)
@composite
def bin16(draw, payloads=payloads()):
    return _do_bin(draw, ">u2", 0xc5, payloads)
@composite
def bin32(draw, payloads=payloads()):
    return _do_bin(draw, ">u4", 0xc6, payloads)

@composite
def all_bin(draw, *args, **kwargs):
    return draw(one_of(b(*args, **kwargs) for b in (bin8, bin16, bin32)))


AVERAGE_TEXT_SIZE = AVERAGE_BIN_SIZE

@_curried_strategy
@composite
def payloads_text(  # arglist kept in sync with hypothesis.strategies.text
        draw,
        alphabet=None,
        min_size=None,
        average_size=AVERAGE_BIN_SIZE,
        max_size=None,
        hard_max_size=None,
        ):
    sizes = _limit_size(
            min_size, average_size, max_size,
            hard_max_size=hard_max_size,
            default_average_size=AVERAGE_TEXT_SIZE,
            )
    return draw(text(alphabet, *sizes))

@composite
def fixstr(draw, payloads_text=payloads_text()):
    if callable(payloads_text):
        payloads_text = payloads_text(hard_max_size=31)
    v = draw(payloads_text)
    data = v.encode("utf-8")
    assume(len(data) <= 31)
    return bfmt(b"%c%s", 0xa0 | len(data), data), v

def _str_prepack(v):
    return v.encode("utf-8")

def _do_str(draw, dtype, firstbyte, payloads_text):
    return _do_bin(draw, dtype, firstbyte, payloads_text, _str_prepack)

@composite
def str8(draw, payloads_text=payloads_text()):
    return _do_str(draw, ">u1", 0xd9, payloads_text)
@composite
def str16(draw, payloads_text=payloads_text()):
    return _do_str(draw, ">u2", 0xda, payloads_text)
@composite
def str32(draw, payloads_text=payloads_text()):
    return _do_str(draw, ">u4", 0xdb, payloads_text)

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

_default_extcodes = integers(0, 127)

def _do_fixext(draw, size, firstbyte, extcodes, payloads):
    if callable(payloads):
        payloads = payloads(hard_size=size)
    code, data = draw(extcodes), draw(payloads)
    assume(len(data) == size)
    return bfmt(b"%c%c%s", firstbyte, code, data), ext_unpack(code, data)

@composite
def fixext1(draw, extcodes=_default_extcodes, payloads=payloads()):
    return _do_fixext(draw, 1, 0xd4, extcodes, payloads)
@composite
def fixext2(draw, extcodes=_default_extcodes, payloads=payloads()):
    return _do_fixext(draw, 2, 0xd5, extcodes, payloads)
@composite
def fixext4(draw, extcodes=_default_extcodes, payloads=payloads()):
    return _do_fixext(draw, 4, 0xd6, extcodes, payloads)
@composite
def fixext8(draw, extcodes=_default_extcodes, payloads=payloads()):
    return _do_fixext(draw, 8, 0xd7, extcodes, payloads)
@composite
def fixext16(draw, extcodes=_default_extcodes, payloads=payloads()):
    return _do_fixext(draw, 16, 0xd8, extcodes, payloads)

@composite
def ext8(draw, extcodes=_default_extcodes, payloads=payloads()):
    return _do_ext(draw, '>u1', 0xc7, extcodes, payloads)
@composite
def ext16(draw, extcodes=_default_extcodes, payloads=payloads()):
    return _do_ext(draw, '>u2', 0xc8, extcodes, payloads)
@composite
def ext32(draw, extcodes=_default_extcodes, payloads=payloads()):
    return _do_ext(draw, '>u4', 0xc9, extcodes, payloads)

def _do_ext(draw, dtype, firstbyte, extcodes, payloads):
    hard_max_size = _num_max(dtype)
    if callable(payloads):
        payloads = payloads(hard_max_size=hard_max_size)
    code, data = draw(extcodes), draw(payloads)
    assume(len(data) <= hard_max_size)
    return bfmt(b'%c%s%c%s', firstbyte, _num_tobytes(dtype, len(data)), code, data), ext_unpack(code, data)

@composite
def all_ext(draw, *args, **kwargs):
    return draw(one_of(e(*args, **kwargs) for e in (fixext1, fixext2, fixext4, fixext8, fixext16, ext8, ext16, ext32)))


@composite
def all_scalar(draw, boolean=boolean(), positive_fixnum=positive_fixnum(), negative_fixnum=negative_fixnum(), all_uint=all_uint(), all_int=all_int(), all_float=all_float(), all_bin=all_bin(), all_str=all_str(), all_ext=all_ext()):
    return draw(one_of(nil(), boolean, positive_fixnum, negative_fixnum, all_uint, all_int, all_float, all_bin, all_str))


def _concat_elements(l):
    acc1, acc2 = bytearray(), []
    for el in l:
        acc1 += el[0]
        acc2.append(el[1])
    return bytes(acc1), acc2

AVERAGE_ARRAY_SIZE = 6

@_curried_strategy
@composite
def array_contents(  # arglist kept in sync with on hypothesis.strategies.list
        draw,
        elements=all_scalar(),
        min_size=None,
        average_size=None,
        max_size=None,
        unique_by=None,
        unique=False,
        hard_max_size=None,
        ):
    sizes = _limit_size(
        min_size, average_size, max_size,
        hard_max_size=hard_max_size,
        default_average_size=AVERAGE_ARRAY_SIZE,
        )
    return draw(lists(elements, *sizes, unique_by=unique_by, unique=unique))


@composite
def fixarray(draw, array_contents=array_contents()):
    if callable(array_contents):
        array_contents = array_contents(hard_max_size=15)
    l = draw(array_contents)
    assume(len(l) <= 15)
    data, v = _concat_elements(l)
    return bfmt(b"%c%s", 0x90 | len(v), data), v

def _do_array(draw, dtype, firstbyte, array_contents):
    hard_max_size = _num_max(dtype)
    if callable(array_contents):
        array_contents = array_contents(hard_max_size=hard_max_size)
    l = draw(array_contents)
    assume(len(l) <= hard_max_size)
    data, v = _concat_elements(l)
    return bfmt(b"%c%s%s", firstbyte, _num_tobytes(dtype, len(v)), data), v

@composite
def array16(draw, array_contents=array_contents()):
    return _do_array(draw, ">u2", 0xdc, array_contents)

@composite
def array32(draw, array_contents=array_contents()):
    return _do_array(draw, ">u4", 0xdd, array_contents)

@composite
def all_array(draw, *args, **kwargs):
    return draw(one_of(a(*args, **kwargs) for a in (fixarray, array16, array32)))


class KeyWrapper(collections.namedtuple('KeyWrapper', 'packed v')):
    def __eq__(self, other):
        if not isinstance(other, KeyWrapper):
            return NotImplemented
        return self.v == other.v

    def __ne__(self, other):
        if not isinstance(other, KeyWrapper):
            return NotImplemented
        return self.v != other.v

    def __hash__(self):
        return hash(self.v)

def _concat_items(d):
    data, v = _concat_elements((k[0] + v[0], (k[1], v[1])) for k, v in d.items())
    return data, type(d)(v)

AVERAGE_MAP_SIZE = 3

@_curried_strategy
@composite
def map_contents(  # arglist kept in sync with on hypothesis.strategies.dictionaries
        draw,
        keys=all_scalar(),
        values=all_scalar(),
        dict_class=collections.OrderedDict,
        min_size=None,
        average_size=None,
        max_size=None,
        hard_max_size=None,
        ):
    sizes = _limit_size(
            min_size, average_size, max_size,
            hard_max_size=hard_max_size,
            default_average_size=AVERAGE_MAP_SIZE,
            )
    return draw(dictionaries(keys.map(lambda k: KeyWrapper(*k)), values, dict_class, *sizes))

@composite
def fixmap(draw, map_contents=map_contents()):
    if callable(map_contents):
        map_contents = map_contents(hard_max_size=15)
    d = draw(map_contents)
    assume(len(d) <= 15)
    data, v = _concat_items(d)
    return bfmt(b"%c%s", 0x80 | len(v), data), v

def _do_map(draw, dtype, firstbyte, map_contents):
    hard_max_size = _num_max(dtype)
    if callable(map_contents):
        map_contents = map_contents(hard_max_size=hard_max_size)
    d = draw(map_contents)
    assume(len(d) <= hard_max_size)
    data, v = _concat_items(d)
    return bfmt(b"%c%s%s", firstbyte, _num_tobytes(dtype, len(v)), data), v

@composite
def map16(draw, map_contents=map_contents()):
    return _do_map(draw, ">u2", 0xde, map_contents)
@composite
def map32(draw, map_contents=map_contents()):
    return _do_map(draw, ">u4", 0xdf, map_contents)

@composite
def all_map(draw, *args, **kwargs):
    return draw(one_of(m(*args, **kwargs) for m in (fixmap, map16, map32)))

@composite
def everything(draw):
    return draw(recursive(
        all_scalar(), lambda S:
            all_array(array_contents(elements=S)) |
            all_map(map_contents(values=S))))  # lists and dicts are unhashable


_msg_types = 'request', 'response', 'notification'

@composite
def msg(draw, types=_msg_types, msg_id=uint32(), method=all_str(), params=all_array(array_contents(everything())), has_error=booleans(), errors=everything(), results=everything()):
    packed_t, t = draw(positive_fixnum(values=sampled_from(types), prepack=lambda t: _msg_types.index(t)))
    if t == 'request':
        contents_tail = msg_id, method, params
    elif t == 'response':
        if draw(has_error):
            contents_tail = msg_id, errors, nil()
        else:
            contents_tail = msg_id, nil(), results
    elif t == 'notification':
        contents_tail = method, params
    else:
        raise ValueError("Invalid message type", t)
    return draw(fixarray(tuples(just((packed_t, t)), *contents_tail)))
